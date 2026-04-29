"""Benchmark tinker08_mass_function: analytical vs emulated,
CPU vs GPU, JIT vs no-JIT.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import colossus.cosmology.cosmology as cc
from colossus.lss import mass_function

import halox

jax.config.update("jax_enable_x64", True)


cosmo = halox.cosmology.Planck18()
emu = halox.emus.SigmaMEmulator()
M = jnp.logspace(13, 15, 256)  # h^-1 Msun
z = jnp.linspace(0, 1, 256)
N_WARMUP = 3
N_REPEAT = 21
N_K_INT = 500


def bench(fn, n_warmup=N_WARMUP, n_repeat=N_REPEAT, reducer="median"):
    """Return wall-clock time in seconds for calling fn().

    ``reducer="median"`` is robust to OS noise (use for no-JIT rows where
    Python dispatch dominates). ``reducer="min"`` reports the steady-state
    floor (use for JIT rows where the floor is the real device time).
    """
    for _ in range(n_warmup):
        result = fn()
        jax.block_until_ready(result)

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    if reducer == "min":
        return min(times)
    return sorted(times)[n_repeat // 2]  # median


def hmf_grid(M, z, cosmo, emu=None):
    """Computes dn/dlnM for all (M, z) pairs.
    Returns shape (len(z), len(M))."""
    return jax.vmap(
        lambda z_i: halox.hmf.tinker08_mass_function(
            M, z_i, cosmo, emu=emu, n_k_int=N_K_INT
        )
    )(z)


devices = {}
devices["CPU"] = jax.devices("cpu")[0]
try:
    devices["GPU"] = jax.devices("gpu")[0]
except RuntimeError:
    print(r"/!\ No GPU found - GPU rows will be skipped.")

def _to_device(tree, dev):
    """device_put every array leaf in a pytree onto ``dev``."""
    if tree is None:
        return None
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, dev), tree)


results = {}  # (device_name, jit_label) -> {analytical: time, emulated: time}

for dev_name, dev in devices.items():
    # Move inputs and any closed-over array state onto the target device once,
    # so jit dispatch compiles for ``dev`` and no implicit H2D copies happen
    # inside the timed loop.
    M_dev = jax.device_put(M, dev)
    z_dev = jax.device_put(z, dev)
    cosmo_dev = _to_device(cosmo, dev)
    emu_dev = _to_device(emu, dev)

    for _emu, col_label in [(None, "Analytical"), (emu_dev, "Emulated")]:
        # --- plain (no JIT) ---
        # nb: vmap still traces, but nothing is compiled ahead of time;
        # each call re-traces and compiles the inner vmap.
        def _call_plain(_emu=_emu):
            return hmf_grid(M_dev, z_dev, cosmo_dev, emu=_emu)

        # Sanity-check device placement on the first run of each combo.
        _probe = _call_plain()
        jax.block_until_ready(_probe)
        assert dev in _probe.devices(), (
            f"expected result on {dev}, got {_probe.devices()}"
        )

        key_plain = (dev_name, "No JIT")
        results.setdefault(key_plain, {})[col_label] = bench(
            _call_plain, reducer="min"
        )

        # --- JIT-compiled ---
        # Wrap the full grid computation in jit so tracing happens once.
        _hmf_jit = jax.jit(
            lambda M_, z_, _emu=_emu: hmf_grid(M_, z_, cosmo_dev, emu=_emu)
        )

        def _call_jit(_fn=_hmf_jit):
            return _fn(M_dev, z_dev)

        key_jit = (dev_name, "JIT")
        results.setdefault(key_jit, {})[col_label] = bench(
            _call_jit, reducer="min"
        )


# --- Colossus (CPU, no JIT) ---
cc.setCosmology("planck18")
M_np = np.asarray(M)
z_np = np.asarray(z)


def _call_colossus():
    out = np.empty((z_np.size, M_np.size))
    for i, zi in enumerate(z_np):
        out[i] = mass_function.massFunction(
            M_np,
            float(zi),
            mdef="200c",
            model="tinker08",
            q_in="M",
            q_out="dndlnM",
        )
    return out


colossus_time = bench(_call_colossus)

# --- Sanity check: halox analytical vs colossus (tolerance matches tests) ---
_halox_ref = np.asarray(hmf_grid(M, z, cosmo, emu=None))
_colossus_ref = _call_colossus()
_disc = _halox_ref / _colossus_ref - 1.0
_max_disc = float(np.max(np.abs(_disc)))
assert _max_disc < 2e-2, (
    f"halox analytical vs colossus disagreement too large: "
    f"max={_max_disc:.3e}"
)
print(f"halox vs colossus max relative discrepancy: {_max_disc:.3e}")


def fmt(t):
    if t < 1e-3:
        return f"{t * 1e6:.1f} µs"
    if t < 1:
        return f"{t * 1e3:.2f} ms"
    return f"{t:.3f} s"


# --- Markdown ---
lines = [
    "| Device | JIT | Analytical | Emulated |",
    "|--------|-----|------------|----------|",
]
for (dev_name, jit_label), cols in sorted(results.items()):
    jit_yn = "Yes" if jit_label == "JIT" else "No"
    lines.append(
        f"| {dev_name} | {jit_yn} | {fmt(cols['Analytical'])} "
        f"| {fmt(cols['Emulated'])} |"
    )

lines.append(f"| CPU | No | Colossus: {fmt(colossus_time)} | |")

table = "\n".join(lines)
print(table)

with open("benchmark_hmf_results.md", "w") as f:
    f.write("# HMF Benchmark Results\n\n")
    f.write(table + "\n")

# --- CSV ---
with open("benchmark_hmf_results.csv", "w") as f:
    f.write("name,time_s\n")
    for method, col_label in [
        ("Analytical", "Analytical"),
        ("Emulator", "Emulated"),
    ]:
        for dev_name in devices:
            for jit_label in ["No JIT", "JIT"]:
                t = results[(dev_name, jit_label)][col_label]
                f.write(f"{method}; {dev_name}; {jit_label},{t}\n")
    f.write(f"colossus,{colossus_time}\n")

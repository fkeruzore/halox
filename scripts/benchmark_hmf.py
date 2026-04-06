"""Benchmark tinker08_mass_function: analytical vs emulated,
CPU vs GPU, JIT vs no-JIT.
"""

import time

import jax
import jax.numpy as jnp

import halox

jax.config.update("jax_enable_x64", True)


cosmo = halox.cosmology.Planck18()
emu = halox.emus.SigmaMEmulator()
M = jnp.logspace(13, 15, 256)  # h^-1 Msun
z = jnp.linspace(0, 2, 16)
N_WARMUP = 1
N_REPEAT = 21


def bench(fn, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    """Return median wall-clock time in seconds for calling fn()."""
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
    return sorted(times)[n_repeat // 2]  # median


def hmf_grid(M, z, cosmo, emu=None):
    """Computes dn/dlnM for all (M, z) pairs.
    Returns shape (len(z), len(M))."""
    return jax.vmap(
        lambda z_i: halox.hmf.tinker08_mass_function(
            M, z_i, cosmo, emu=emu
        )
    )(z)


devices = {}
devices["CPU"] = jax.devices("cpu")[0]
try:
    devices["GPU"] = jax.devices("gpu")[0]
except RuntimeError:
    print(r"/!\ No GPU found - GPU rows will be skipped.")

results = {}  # (device_name, jit_label) -> {analytical: time, emulated: time}

for dev_name, dev in devices.items():
    for _emu, col_label in [(None, "Analytical"), (emu, "Emulated")]:
        # --- plain (no JIT) ---
        # nb: vmap still traces, but nothing is compiled ahead of time;
        # each call re-traces and compiles the inner vmap.
        def _call_plain(_emu=_emu, _dev=dev):
            with jax.default_device(_dev):
                return hmf_grid(M, z, cosmo, emu=_emu)

        key_plain = (dev_name, "No JIT")
        results.setdefault(key_plain, {})[col_label] = bench(_call_plain)

        # --- JIT-compiled ---
        # Wrap the full grid computation in jit so tracing happens once.
        _hmf_jit = jax.jit(
            lambda M_, z_, _emu=_emu: hmf_grid(
                M_, z_, cosmo, emu=_emu
            )
        )

        def _call_jit(_dev=dev, _fn=_hmf_jit):
            with jax.default_device(_dev):
                return _fn(M, z)

        key_jit = (dev_name, "JIT")
        results.setdefault(key_jit, {})[col_label] = bench(_call_jit)


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


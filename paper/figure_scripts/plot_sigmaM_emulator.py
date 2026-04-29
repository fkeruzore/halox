"""Validate the sigma(M) emulator against analytic halox and colossus."""

import jax
import jax.numpy as jnp
import numpy as np
from colossus.lss import peaks, mass_function
import colossus.cosmology.cosmology as cc

from halox import cosmology, hmf, lss
from halox.emus import SigmaMEmulator

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

jax.config.update("jax_enable_x64", True)

plt.style.use(["seaborn-v0_8-darkgrid", "petroff10"])
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})
def lighten_color(color, amount=0.5):
    c = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*c)
    l = 1 - (1 - l) * (1 - amount)
    return colorsys.hls_to_rgb(h, l, s)

# --- Cosmologies ---
cosmo_planck = cosmology.Planck18()
cosmo_lows8 = cosmology.Planck18(
    h=0.682,
    Omega_c=0.305 - 0.0473,
    Omega_b=0.0473,
    sigma8=0.76,
    n_s=0.965,
)

cc.addCosmology(
    "my_planck18",
    params={
        "flat": True,
        "H0": 100 * cosmo_planck.h,
        "Om0": cosmo_planck.Omega_m,
        "Ob0": cosmo_planck.Omega_b,
        "sigma8": cosmo_planck.sigma8,
        "ns": cosmo_planck.n_s,
    },
)
cc.setCosmology("my_planck18")
cc.addCosmology(
    "low_s8",
    params={
        "flat": True,
        "H0": 100 * cosmo_lows8.h,
        "Om0": cosmo_lows8.Omega_m,
        "Ob0": cosmo_lows8.Omega_b,
        "sigma8": cosmo_lows8.sigma8,
        "ns": cosmo_lows8.n_s,
    },
)

# --- Emulator ---
emu = SigmaMEmulator()

# --- Mass array ---
masses = jnp.logspace(11, 16, 200)
masses_np = np.array(masses)

# --- Cases: (label, color, redshift, halox_cosmo, colossus_cosmo_name) ---
cases = [
    (r"Planck18, $z=0$", "C0", 0.0, cosmo_planck, "my_planck18"),
    (r"Planck18, $z=1$", "C1", 1.0, cosmo_planck, "my_planck18"),
    (r"Low-S8, $z=0$", "C2", 0.0, cosmo_lows8, "low_s8"),
]

# --- Compute ---
results = []
for label, color, z, cosmo_h, cosmo_c_name in cases:
    sig_analytic = lss.sigma_M(masses, z, cosmo_h)
    sig_emulated = lss.sigma_M(masses, z, cosmo_h, emu=emu)
    cosmo_col = cc.setCosmology(cosmo_c_name)
    R_col = peaks.lagrangianR(masses_np)
    sig_colossus = cosmo_col.sigma(R_col, z=z)
    results.append(
        (
            label,
            color,
            np.array(sig_analytic),
            np.array(sig_emulated),
            sig_colossus,
        )
    )

# --- Plot ---
fig, (ax_top, ax_bot) = plt.subplots(
    2,
    1,
    gridspec_kw={"height_ratios": [2, 1]},
    sharex=True,
    # figsize=(8, 6),
)

for label, color, sig_a, sig_e, sig_c in results:
    ax_top.loglog(
        masses_np,
        sig_a,
        color=color,
        ls="--",
        lw=2.0,
        label=f"{label}, analytic",
    )
    ax_top.loglog(
        masses_np,
        sig_e,
        color=lighten_color(color, 1/3),
        ls=":",
        lw=2.5,
        label=f"{label}, emulator",
    )
    ax_top.loglog(
        masses_np,
        sig_c,
        color=lighten_color(color, -1/3),
        ls="-",
        lw=1.5,
        zorder=0,
        label=f"{label}, colossus",
    )

    diff_a = (sig_a / sig_c - 1) * 100
    diff_e = (sig_e / sig_c - 1) * 100
    ax_bot.semilogx(masses_np, diff_a, color=color, ls="--", lw=2.0)
    ax_bot.semilogx(masses_np, diff_e, color=lighten_color(color, 1/3), ls=":", lw=2.5)

ax_top.set_ylabel(r"$\sigma(M, z)$")
ax_top.legend(fontsize=7, ncol=3)

ax_bot.axhline(0, color="k", ls=":", lw=0.8)
ax_bot.set_ylabel("Difference [%]")
ax_bot.set_ylim(-0.5, 0.5)
ax_bot.set_xlabel(r"$M$ [$h^{-1}\,M_\odot$]")

fig.tight_layout()
fig.savefig("../sigmaM_emulator_validation.png", dpi=500)

# ===== HMF plot =====
delta_c = 200.0

hmf_results = []
for label, color, z, cosmo_h, cosmo_c_name in cases:
    hmf_analytic = hmf.tinker08_mass_function(masses, z, cosmo_h, delta_c)
    hmf_emulated = hmf.tinker08_mass_function(
        masses, z, cosmo_h, delta_c, emu=emu
    )
    cc.setCosmology(cosmo_c_name)
    hmf_colossus = mass_function.massFunction(
        masses_np,
        z,
        mdef=f"{delta_c:.0f}c",
        model="tinker08",
        q_in="M",
        q_out="dndlnM",
    )
    hmf_results.append(
        (
            label,
            color,
            np.array(hmf_analytic),
            np.array(hmf_emulated),
            hmf_colossus,
        )
    )

fig2, (ax_top2, ax_bot2) = plt.subplots(
    2,
    1,
    gridspec_kw={"height_ratios": [2, 1]},
    sharex=True,
)

for label, color, hmf_a, hmf_e, hmf_c in hmf_results:
    ax_top2.loglog(
        masses_np,
        hmf_a,
        color=color,
        ls="--",
        lw=2.0,
        label=f"{label}, analytic",
    )
    ax_top2.loglog(
        masses_np,
        hmf_e,
        color=lighten_color(color, 1/3),
        ls=":",
        lw=2.5,
        label=f"{label}, emulator",
    )
    ax_top2.loglog(
        masses_np,
        hmf_c,
        color=lighten_color(color, -1/3),
        ls="-",
        lw=1.5,
        zorder=0,
        label=f"{label}, colossus",
    )

    diff_a = (hmf_a / hmf_c - 1) * 100
    diff_e = (hmf_e / hmf_c - 1) * 100
    ax_bot2.semilogx(masses_np, diff_a, color=color, ls="--", lw=2.0)
    ax_bot2.semilogx(masses_np, diff_e, color=lighten_color(color, 1/3), ls=":", lw=2.5)

ax_top2.set_ylabel(r"${\rm d}n/{\rm d}\ln M$ [$h^{3}\,\mathrm{Mpc}^{-3}$]")
ax_top2.legend(fontsize=7, ncol=3)

ax_bot2.axhline(0, color="k", ls=":", lw=0.8)
ax_bot2.set_ylabel("Difference [%]")
ax_bot2.set_ylim(-7.5, 7.5)
ax_bot2.set_xlabel(r"$M$ [$h^{-1}\,M_\odot$]")

fig2.tight_layout()
fig2.savefig("../hmf_emulator_validation.png", dpi=500)
plt.show()

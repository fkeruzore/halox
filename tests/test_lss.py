import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
from colossus.halo.mass_so import densityThreshold
from colossus.lss import peaks
import colossus.cosmology.cosmology as cc
import halox

jax.config.update("jax_enable_x64", True)

test_mzs = jnp.array(
    [
        [1e15, 0.0],
        [1e14, 0.0],
        [1e13, 0.0],
        [1e14, 1.0],
        [1e13, 1.0],
        [1e14, 2.0],
        [1e13, 0.0],
    ]
)
test_deltas = [200.0, 500.0]
test_cosmos = {
    "Planck18": [halox.cosmology.Planck18(), "planck18"],
    "70_0.3": [
        jc.Cosmology(0.25, 0.05, 0.7, 0.97, 0.8, 0.0, -1.0, 0.0),
        "70_0.3",
    ],
}
cc.addCosmology(
    cosmo_name="70_0.3",
    params=dict(
        flat=True,
        H0=70.0,
        Om0=0.3,
        Ob0=0.05,
        de_model="lambda",
        sigma8=0.8,
        ns=0.97,
    ),
)

sigma_R = jax.jit(halox.lss.sigma_R)


@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_lagrangian_R(cosmo_name, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    ms = test_mzs[:, 0]
    R_c = peaks.lagrangianR(ms)
    R_h = halox.lss.mass_to_lagrangian_radius(ms, cosmo_j)  # cMpc

    if return_vals:
        return R_h, R_c
    discrepancy = R_h / R_c - 1.0
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 5e-3, (
        f"Bias in lagrangianR: avg={avg_disc:.3e}, max={max_disc:.3e}"
    )


@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_sigma_R_z(cosmo_name, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    R_c = peaks.lagrangianR(ms)
    sigma_c = jnp.array(
        [cosmo_c.sigma(R_c[i], z=zs[i]) for i in range(len(test_mzs))]
    )

    R_h = halox.lss.mass_to_lagrangian_radius(ms, cosmo_j)  # Mpc
    sigma_h = jnp.array(
        [sigma_R(R_h[i], zs[i], cosmo_j) for i in range(len(test_mzs))]
    )

    if return_vals:
        return sigma_h, sigma_c
    discrepancy = sigma_h / sigma_c - 1.0
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 1e-2, (
        f"Bias in sigma: avg={avg_disc:.3e}, max={max_disc:.3e}"
    )


@pytest.mark.parametrize("delta_c", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_overdensity_c_to_m(delta_c, cosmo_name, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    zs = test_mzs[:, 1]

    d_c = jnp.array(
        [
            densityThreshold(zs[i], f"{delta_c:.0f}c") / cosmo_c.rho_m(zs[i])
            for i in range(len(test_mzs))
        ]
    )
    d_h = jnp.array(
        [
            halox.lss.overdensity_c_to_m(delta_c, zs[i], cosmo_j)
            for i in range(len(test_mzs))
        ]
    )

    if return_vals:
        return d_h, d_c
    discrepancy = d_h / d_c - 1.0
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 5e-3, (
        f"Bias in delta_m: avg={avg_disc:.3e}, max={max_disc:.3e}"
    )


if __name__ == "__main__":
    cosmo_j, cosmo_c = test_cosmos["70_0.3"]
    cosmo_c = cc.setCosmology(cosmo_c)
    R_h, R_c = test_lagrangian_R("70_0.3", return_vals=True)
    print(R_h / R_c)

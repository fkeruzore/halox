import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
from colossus.halo.mass_so import densityThreshold
from colossus.lss import mass_function, peaks
import colossus.cosmology.cosmology as cc
import halox

jax.config.update("jax_enable_x64", True)

rtol = 1e-2
test_deltas = [200.0, 500.0]
test_cosmos = {
    "Planck18": [halox.cosmology.Planck18, "planck18"],
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
G = halox.cosmology.G
ms, zs = jnp.logspace(13, 15, 3), jnp.linspace(0, 2, 3)


@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_lagrangian_R(cosmo_name, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    R_c = peaks.lagrangianR(ms)  # mass in Msun -> radius in cMpc
    R_h = halox.hmf.mass_to_lagrangian_radius(ms, cosmo_j)  # cMpc

    if return_vals:
        return R_h, R_c
    assert jnp.allclose(R_c, R_h, rtol=rtol), (
        f"Different lagrangianR: {R_c} != {R_h}"
    )


@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_sigma_R_z(cosmo_name, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    R_c = peaks.lagrangianR(ms)  # mass in Msun -> radius in cMpc
    sigma_c = jnp.array([cosmo_c.sigma(R_c, z=z) for z in zs])

    R_h = halox.hmf.mass_to_lagrangian_radius(ms, cosmo_j)  # Mpc
    sigma_h = jax.vmap(lambda z: halox.hmf.sigma_R(R_h, z, cosmo_j))(zs)

    if return_vals:
        return sigma_h, sigma_c
    assert jnp.allclose(sigma_c, sigma_h, rtol=rtol), (
        f"Different sigma: {sigma_c} != {sigma_h}"
    )


@pytest.mark.parametrize("delta_c", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_overdensity_c_to_m(delta_c, cosmo_name, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    d_c = jnp.array(
        [densityThreshold(z, f"{delta_c:.0f}c") / cosmo_c.rho_m(z) for z in zs]
    )
    d_h = jax.vmap(
        lambda z: halox.hmf.overdensity_c_to_m(delta_c, z, cosmo_j)
    )(zs)

    if return_vals:
        return d_h, d_c
    assert jnp.allclose(d_c, d_h, rtol=rtol), (
        f"Different delta_m: avg ratio={jnp.mean(d_h / d_c)} ({d_c} != {d_h})"
    )


@pytest.mark.parametrize("delta_c", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_tinker08_f_sigma(delta_c, cosmo_name, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    f_c = jnp.array(
        [
            mass_function.massFunction(
                ms,
                z,
                mdef=f"{delta_c:.0f}c",
                model="tinker08",
                q_in="M",
                q_out="f",
            )
            for z in zs
        ]
    )
    f_h = jax.vmap(
        lambda z: halox.hmf.tinker08_f_sigma(
            ms, z, cosmo=cosmo_j, delta_c=delta_c
        )
    )(zs)

    if return_vals:
        return f_h, f_c
    assert jnp.allclose(f_c, f_h, rtol=rtol), (
        f"Different hmf: avg ratio={jnp.mean(f_h / f_c)} ({f_c} != {f_h})"
    )


if __name__ == "__main__":
    cosmo_j, cosmo_c = test_cosmos["70_0.3"]
    cosmo_c = cc.setCosmology(cosmo_c)
    f_h, f_c = test_tinker08_f_sigma(500.0, "70_0.3", return_vals=True)
    print(f_h / f_c)

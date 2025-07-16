import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
import halox.cosmology as hc
import astropy.cosmology as ac

jax.config.update("jax_enable_x64", True)

test_cosmos = {
    "Planck15": [jc.Planck15(), ac.Planck15],
    "Planck18": [hc.Planck18, ac.Planck18],
    "70_0.3": [
        jc.Cosmology(0.25, 0.05, 0.7, 0.97, 0.8, 0.0, -1.0, 0.0),
        ac.FlatLambdaCDM(70.0, 0.3, Ob0=0.05),
    ],
}

test_zs = jnp.linspace(0, 3, 13)


@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_cosmo_params(cosmo_name):
    cosmo_j, cosmo_a = test_cosmos[cosmo_name]
    assert jnp.isclose(
        cosmo_j.h, cosmo_a.h, atol=1e-4
    ), "Cosmologies have different h"
    assert jnp.isclose(
        cosmo_j.Omega_b, cosmo_a.Ob0, atol=1e-4
    ), "Cosmologies have different Ob"
    assert jnp.isclose(
        cosmo_j.Omega_c, cosmo_a.Odm0, atol=1e-4
    ), "Cosmologies have different Ocdm"
    assert jnp.isclose(
        cosmo_j.Omega_m, cosmo_a.Om0, atol=1e-4
    ), "Cosmologies have different Om"
    assert jnp.isclose(
        cosmo_j.Omega_k, cosmo_a.Ok0, atol=1e-4
    ), "Cosmologies have different Om"


@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_hubble_parameter(cosmo_name):
    cosmo_j, cosmo_a = test_cosmos[cosmo_name]
    H_j = hc.hubble_parameter(test_zs, cosmo_j)
    H_a = cosmo_a.H(test_zs).to("km s-1 Mpc-1").value
    assert jnp.allclose(H_j, jnp.array(H_a), rtol=5e-3), (
        f"Different H({test_zs}): {H_j} != {H_a}" f"(ratio: {H_j / H_a})"
    )


@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_critical_density(cosmo_name):
    cosmo_j, cosmo_a = test_cosmos[cosmo_name]
    rhoc_j = hc.critical_density(test_zs, cosmo_j)
    rhoc_a = cosmo_a.critical_density(test_zs).to("Msun Mpc-3").value
    assert jnp.allclose(rhoc_j, jnp.array(rhoc_a), rtol=1e-2), (
        f"Different rhoc({test_zs}): {rhoc_j} != {rhoc_a}"
        f"(ratio: {rhoc_j / rhoc_a})"
    )


@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_volume_element(cosmo_name):
    cosmo_j, cosmo_a = test_cosmos[cosmo_name]
    dV_j = hc.differential_comoving_volume(test_zs, cosmo_j)
    dV_a = cosmo_a.differential_comoving_volume(test_zs).to("Mpc3 sr-1").value
    assert jnp.allclose(dV_j, jnp.array(dV_a), rtol=1e-2), (
        f"Different dV({test_zs}): {dV_j} != {dV_a}" f"(ratio: {dV_j / dV_a})"
    )

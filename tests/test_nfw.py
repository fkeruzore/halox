import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
from colossus.halo import profile_nfw
import colossus.cosmology.cosmology as cc
import halox

jax.config.update("jax_enable_x64", True)

rtol = 1e-2
test_halos = {
    "him_loz": {"M": 1e15, "c": 4.0, "z": 0.1},
    "lom_hiz": {"M": 1e14, "c": 5.5, "z": 1.0},
}
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
G = halox.cosmology.G


@pytest.mark.parametrize("halo_name", test_halos.keys())
@pytest.mark.parametrize("delta", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_density(halo_name, delta, cosmo_name):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    cosmo_c = cc.setCosmology(cosmo_c)
    nfw_h = halox.nfw.NFWHalo(m_delta, c_delta, z, cosmo_j, delta=delta)
    nfw_c = profile_nfw.NFWProfile(
        M=m_delta,
        c=c_delta,
        z=z,
        mdef=f"{delta:.0f}c",
    )

    rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
    rho_c = nfw_c.density(rs * 1000) * 1e9
    rho_h = nfw_h.density(rs)
    assert jnp.allclose(jnp.array(rho_c), rho_h, rtol=rtol), (
        f"Different rho({rs}): {rho_c} != {rho_h}"
    )


@pytest.mark.parametrize("halo_name", test_halos.keys())
@pytest.mark.parametrize("delta", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_enclosed_mass(halo_name, delta, cosmo_name):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    cosmo_c = cc.setCosmology(cosmo_c)
    nfw_h = halox.nfw.NFWHalo(m_delta, c_delta, z, cosmo_j, delta=delta)
    nfw_c = profile_nfw.NFWProfile(
        M=m_delta,
        c=c_delta,
        z=z,
        mdef=f"{delta:.0f}c",
    )

    rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
    mass_c = nfw_c.enclosedMass(rs * 1000)
    mass_h = nfw_h.enclosed_mass(rs)
    assert jnp.allclose(jnp.array(mass_c), mass_h, rtol=rtol), (
        f"Different M(<{rs}): {mass_c} != {mass_h}"
    )


@pytest.mark.parametrize("halo_name", test_halos.keys())
@pytest.mark.parametrize("delta", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_potential(halo_name, delta, cosmo_name):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    cosmo_c = cc.setCosmology(cosmo_c)
    nfw_h = halox.nfw.NFWHalo(m_delta, c_delta, z, cosmo_j, delta=delta)
    nfw_c = profile_nfw.NFWProfile(
        M=m_delta,
        c=c_delta,
        z=z,
        mdef=f"{delta:.0f}c",
    )

    rs = jnp.logspace(-2, 1, 6)  # Mpc

    _r0 = nfw_c.par["rhos"] * 1e9  # Msun Mpc-3
    _rs = nfw_c.par["rs"] / 1e3  # Mpc

    phi_c = -4 * jnp.pi * G * _r0 * _rs**3 * jnp.log(1 + rs / _rs) / rs
    phi_h = nfw_h.potential(rs)
    assert jnp.allclose(jnp.array(phi_c), phi_h, rtol=rtol), (
        f"Different phi({rs}): {phi_c} != {phi_h}"
    )


@pytest.mark.parametrize("halo_name", test_halos.keys())
@pytest.mark.parametrize("delta", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_circular_velocity(halo_name, delta, cosmo_name):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    cosmo_c = cc.setCosmology(cosmo_c)
    nfw_h = halox.nfw.NFWHalo(m_delta, c_delta, z, cosmo_j, delta=delta)
    nfw_c = profile_nfw.NFWProfile(
        M=m_delta,
        c=c_delta,
        z=z,
        mdef=f"{delta:.0f}c",
    )

    rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
    vc_c = nfw_c.circularVelocity(rs * 1000)
    vc_h = nfw_h.circular_velocity(rs)
    assert jnp.allclose(jnp.array(vc_c), vc_h, rtol=rtol), (
        f"Different v_circ({rs}): {vc_c} != {vc_h}"
    )

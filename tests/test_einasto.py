import jax
import pytest
import numpy as np
import jax.numpy as jnp
import jax_cosmo as jc
from colossus.halo import profile_einasto, mass_defs
import colossus.cosmology.cosmology as cc
from scipy.integrate import quad
import halox

# Note, current alpha is calculated relying on a virial mass input,
# currently this is still implemented for coverage, but it must be
# understood that this current test is not necessarily physical

# TODO:
# velociity dispersion and surface density tests after those features are added

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

@jax.jit
def a_from_nu(m_delta, z, cosmo_j):
    return halox.halo.einasto.a_from_nu(m_delta, z, cosmo_j, n_k_int=200)

@pytest.mark.parametrize("halo_name", test_halos.keys())
@pytest.mark.parametrize("delta", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_density(halo_name, delta, cosmo_name, return_vals: bool = False):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    cosmo_c = cc.setCosmology(cosmo_c)
    alpha = a_from_nu(m_delta, z, cosmo_j)
    ein_h = halox.halo.einasto.EinastoHalo(
        m_delta, c_delta, z, alpha=alpha, cosmo=cosmo_j, delta=delta
    )
    ein_c = profile_einasto.EinastoProfile(
        M=m_delta, c=c_delta, z=z, mdef=f"{delta:.0f}c", alpha=alpha
    )

    rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
    res_c = (
        ein_c.density(rs * 1000) * 1e9
    )  # 1e9 is a valid unit conversion from ~1/kpc^3 to ~1/Mpc^3
    res_h = ein_h.density(rs)

    if return_vals:
        return res_h, res_c

    assert jnp.allclose(jnp.array(res_c), res_h, rtol=rtol), (
        f"Different rho({rs}): {res_c} != {res_h}"
    )


@pytest.mark.parametrize("halo_name", test_halos.keys())
@pytest.mark.parametrize("delta", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_enclosed_mass(
    halo_name, delta, cosmo_name, return_vals: bool = False
):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    cosmo_c = cc.setCosmology(cosmo_c)
    alpha = a_from_nu(m_delta, z, cosmo_j)
    ein_h = halox.halo.einasto.EinastoHalo(
        m_delta, c_delta, z, alpha=alpha, cosmo=cosmo_j, delta=delta
    )
    ein_c = profile_einasto.EinastoProfile(
        M=m_delta, c=c_delta, z=z, mdef=f"{delta:.0f}c", alpha=alpha
    )

    rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
    res_c = ein_c.enclosedMass(rs * 1000)
    res_h = ein_h.enclosed_mass(rs)

    if return_vals:
        return res_h, res_c

    assert jnp.allclose(jnp.array(res_c), res_h, rtol=rtol), (
        f"Different M(<{rs}): {res_c} != {res_h}"
    )


@pytest.mark.parametrize("halo_name", test_halos.keys())
@pytest.mark.parametrize("delta", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_potential(halo_name, delta, cosmo_name, return_vals: bool = False):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    def einasto_potential_numeric(r, halo, r_max):
        """
        halo: a colossus halo
        """
        G = 4.30091e-6  # (kpc/h) (km/s)^2 / (Msun/h)
        r = np.atleast_1d(r)

        def outer_term(r):
            integrand = lambda rp: halo.enclosedMass(rp) / rp**2
            return np.array([quad(integrand, ri, r_max)[0] for ri in r])

        # phi = -G * (halo.enclosedMass(r) / r + outer_term(r))
        phi = -G * outer_term(r)
        return phi

    cosmo_c = cc.setCosmology(cosmo_c)
    alpha = a_from_nu(m_delta, z, cosmo_j)
    ein_h = halox.halo.einasto.EinastoHalo(
        m_delta, c_delta, z, alpha=alpha, cosmo=cosmo_j, delta=delta
    )
    ein_c = profile_einasto.EinastoProfile(
        M=m_delta, c=c_delta, z=z, mdef=f"{delta:.0f}c", alpha=alpha
    )

    rs = jnp.logspace(-2, 1, 6)  # Mpc
    f = 1e4
    res_c = einasto_potential_numeric(
        rs * 1000, ein_c, r_max=f * ein_c.RDelta(z, mdef=f"{delta:.0f}c")
    )
    res_h = ein_h.potential(rs)

    if return_vals:
        return res_h, res_c

    assert jnp.allclose(jnp.array(res_c), res_h, rtol=rtol), (
        f"Different phi({rs}): {res_c} != {res_h}"
    )


@pytest.mark.parametrize("halo_name", test_halos.keys())
@pytest.mark.parametrize("delta", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_circular_velocity(
    halo_name, delta, cosmo_name, return_vals: bool = False
):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    cosmo_c = cc.setCosmology(cosmo_c)
    alpha = a_from_nu(m_delta, z, cosmo_j)
    ein_h = halox.halo.einasto.EinastoHalo(
        m_delta, c_delta, z, alpha=alpha, cosmo=cosmo_j, delta=delta
    )
    ein_c = profile_einasto.EinastoProfile(
        M=m_delta, c=c_delta, z=z, mdef=f"{delta:.0f}c", alpha=alpha
    )

    rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
    res_c = ein_c.circularVelocity(rs * 1000)
    res_h = ein_h.circular_velocity(rs)

    if return_vals:
        return res_h, res_c

    assert jnp.allclose(jnp.array(res_c), res_h, rtol=rtol), (
        f"Different v_circ({rs}): {res_c} != {res_h}"
    )


@jax.jit
def halox_convert_delta(m_delta, c_delta, z, cosmo_j, delta_in, delta_out):
    alpha = a_from_nu(m_delta, z, cosmo_j)
    ein_h = halox.halo.EinastoHalo(
        m_delta, c_delta, z, alpha=alpha, cosmo=cosmo_j, delta=delta_in
    )
    return jnp.squeeze(jnp.array(ein_h.to_delta(delta_out)))


@pytest.mark.parametrize("halo_name", test_halos.keys())
@pytest.mark.parametrize("delta_in", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_convert_delta(
    halo_name, delta_in, cosmo_name, return_vals: bool = False
):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    cosmo_c = cc.setCosmology(cosmo_c)

    delta_out = 500.0 if delta_in == 200.0 else 200.0
    res_c = jnp.array(
        mass_defs.changeMassDefinition(
            m_delta,
            c_delta,
            z,
            mdef_in=f"{delta_in:.0f}c",
            mdef_out=f"{delta_out:.0f}c",
            profile="nfw",
        )
    )
    res_c = res_c.at[1].divide(1e3)

    res_h = halox_convert_delta(
        m_delta, c_delta, z, cosmo_j, delta_in, delta_out
    )

    if return_vals:
        return res_h, res_c

    assert jnp.allclose(res_c, res_h, rtol=rtol), (
        f"Different results: {res_c} != {res_h}"
    )

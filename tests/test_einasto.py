import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
from colossus.halo import profile_einasto, mass_defs
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
def test_density(halo_name, delta, cosmo_name, return_vals: bool = False):
    halo = test_halos[halo_name]
    m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]

    cosmo_c = cc.setCosmology(cosmo_c)
    alpha = halox.einasto.a_from_nu(m_delta, z, cosmo_j)
    ein_h = halox.einasto.EinastoHalo(m_delta, c_delta, z, alpha = alpha, cosmo = cosmo_j, delta=delta)
    ein_c = profile_einasto.EinastoProfile(
        M=m_delta,
        c=c_delta,
        z=z,
        mdef=f"{delta:.0f}c",
        alpha = alpha
    )

    rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
    res_c = ein_c.density(rs * 1000) * 1e9
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
    alpha = halox.einasto.a_from_nu(m_delta, z, cosmo_j)
    ein_h = halox.einasto.EinastoHalo(m_delta, c_delta, z, alpha = alpha, cosmo = cosmo_j, delta=delta)
    ein_c = profile_einasto.EinastoProfile(
        M=m_delta,
        c=c_delta,
        z=z,
        mdef=f"{delta:.0f}c",
        alpha = alpha
    )

    rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
    res_c = ein_c.enclosedMass(rs * 1000)
    res_h = ein_h.enclosed_mass(rs)

    if return_vals:
        return res_h, res_c

    assert jnp.allclose(jnp.array(res_c), res_h, rtol=rtol), (
        f"Different M(<{rs}): {res_c} != {res_h}"
    )


# profiley to get this to work
# @pytest.mark.parametrize("halo_name", test_halos.keys())
# @pytest.mark.parametrize("delta", test_deltas)
# @pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
# def test_potential(halo_name, delta, cosmo_name, return_vals: bool = False):
#     halo = test_halos[halo_name]
#     m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
#     cosmo_j, cosmo_c = test_cosmos[cosmo_name]

#     cosmo_c = cc.setCosmology(cosmo_c)
#     alpha = halox.einasto.a_from_nu(m_delta, z, cosmo_j)
#     ein_h = halox.einasto.EinastoHalo(m_delta, c_delta, z, alpha = alpha, cosmo = cosmo_j, delta=delta)
#     ein_c = profile_einasto.EinastoProfile(
#         M=m_delta,
#         c=c_delta,
#         z=z,
#         mdef=f"{delta:.0f}c",
#         alpha = alpha
#     )

#     rs = jnp.logspace(-2, 1, 6)  # Mpc

#     _r0 = ein_c.par["rhos"] * 1e9  # Msun Mpc-3
#     _rs = ein_c.par["rs"] / 1e3  # Mpc

#     res_c = -4 * jnp.pi * G * _r0 * _rs**3 * jnp.log(1 + rs / _rs) / rs
#     res_h = ein_h.potential(rs)

#     if return_vals:
#         return res_h, res_c

#     assert jnp.allclose(jnp.array(res_c), res_h, rtol=rtol), (
#         f"Different phi({rs}): {res_c} != {res_h}"
#     )


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
    alpha = halox.einasto.a_from_nu(m_delta, z, cosmo_j)
    ein_h = halox.einasto.EinastoHalo(m_delta, c_delta, z, alpha = alpha, cosmo = cosmo_j, delta=delta)
    ein_c = profile_einasto.EinastoProfile(
        M=m_delta,
        c=c_delta,
        z=z,
        mdef=f"{delta:.0f}c",
        alpha = alpha
    )

    rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
    res_c = ein_c.circularVelocity(rs * 1000)
    res_h = ein_h.circular_velocity(rs)

    if return_vals:
        return res_h, res_c

    assert jnp.allclose(jnp.array(res_c), res_h, rtol=rtol), (
        f"Different v_circ({rs}): {res_c} != {res_h}"
    )


# @pytest.mark.parametrize("halo_name", test_halos.keys())
# @pytest.mark.parametrize("delta", test_deltas)
# @pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
# def test_velocity_dispersion(
#     halo_name, delta, cosmo_name, return_vals: bool = False
# ):
#     halo = test_halos[halo_name]
#     m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
#     cosmo_j, cosmo_c = test_cosmos[cosmo_name]

#     cosmo_c = cc.setCosmology(cosmo_c)
#     ein_h = halox.einasto.EinastoHalo(m_delta, c_delta, z, cosmo_j, delta=delta)
#     ein_c = profile_einasto.EinastoProfile(
#         M=m_delta,
#         c=c_delta,
#         z=z,
#         mdef=f"{delta:.0f}c",
#     )

#     rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc

#     x = rs / (ein_c.par["rs"] * 1e-3)
#     g_x = (jnp.log(1 + x) - x / (1 + x)) / x**2
#     gc = jnp.log(1 + c_delta) - c_delta / (1 + c_delta)
#     res_c = jnp.sqrt(
#         G
#         * m_delta
#         * gc
#         * g_x
#         / (ein_c.RDelta(z, f"{delta:.0f}c") * 1e-3 * x * (1 + x) ** 2)
#     )
#     res_h = ein_h.velocity_dispersion(rs)

#     if return_vals:
#         return res_h, res_c

#     assert jnp.allclose(jnp.array(res_c), res_h, rtol=rtol), (
#         f"Different sigma_v({rs}): {res_c} != {res_h}"
#     )


# @pytest.mark.parametrize("halo_name", test_halos.keys())
# @pytest.mark.parametrize("delta", test_deltas)
# @pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
# def test_surface_density(
#     halo_name, delta, cosmo_name, return_vals: bool = False
# ):
#     halo = test_halos[halo_name]
#     m_delta, c_delta, z = halo["M"], halo["c"], halo["z"]
#     cosmo_j, cosmo_c = test_cosmos[cosmo_name]

#     cosmo_c = cc.setCosmology(cosmo_c)
#     ein_h = halox.einasto.EinastoHalo(m_delta, c_delta, z, cosmo_j, delta=delta)
#     ein_c = profile_einasto.EinastoProfile(
#         M=m_delta,
#         c=c_delta,
#         z=z,
#         mdef=f"{delta:.0f}c",
#     )

#     rs = jnp.logspace(-2, 1, 6)  # h-1 Mpc
#     res_c = ein_c.surfaceDensity(rs * 1000) * 1e6
#     res_h = ein_h.surface_density(rs)

#     if return_vals:
#         return res_h, res_c

#     assert jnp.allclose(jnp.array(res_c), res_h, rtol=rtol), (
#         f"Different sigma({rs}): {res_c} != {res_h}"
#     )  # E501: ignore


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
    alpha = halox.einasto.a_from_nu(m_delta, z, cosmo_j)
    ein_h = halox.einasto.EinastoHalo(m_delta, c_delta, z, alpha = alpha, cosmo = cosmo_j, delta=delta_in)

    delta_out = 500.0 if delta_in == 200.0 else 200.0
    res_h = jnp.squeeze(jnp.array(ein_h.to_delta(delta_out)))
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

    if return_vals:
        return res_h, res_c

    assert jnp.allclose(res_c, res_h, rtol=rtol), (
        f"Different results: {res_c} != {res_h}"
    )

import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
import colossus.halo.concentration as ccon
import colossus.cosmology.cosmology as cc
import halox
from halox.halo.cMrelation import (
    duffy08,
    klypin11,
    prada12,
    child18all,
    child18relaxed,
)

jax.config.update("jax_enable_x64", True)

# TODO: Check prada12 for small calculation error on the percent level,
# see if we can change this to get it in allignment

rtol = 1e-2

test_mzs = jnp.array(
    [
        [1e15, 0.0],
        [1e14, 0.0],
        [1e13, 0.0],
        [1e12, 0.0],
        [1e14, 1.0],
        [1e13, 1.0],
        [1e12, 1.0],
        [1e14, 2.0],
        [1e13, 2.0],
        [1e12, 2.0],
    ]
)
test_deltas = [200.0, 500.0]
test_cosmos = {
    # "Planck18": [halox.cosmology.Planck18(), "planck18"],
    "Planck18": [
        jc.Cosmology(
            h=0.6766,
            Omega_b=0.049,
            Omega_c=0.2621,
            Omega_k=0.0,
            w0=-1.0,
            wa=0.0,
            n_s=0.9665,
            sigma8=0.8102,
        ),
        "planck18",
    ],
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


@pytest.mark.parametrize("m_delta, z", test_mzs)
def test_duffy08(m_delta, z, return_vals=False):
    c_h = duffy08()(M=m_delta, z=z)
    c_c = ccon.modelDuffy08(m_delta, z, mdef="200c")
    if return_vals:
        return c_h, c_c

    assert jnp.isclose(jnp.atleast_1d(c_h), jnp.atleast_1d(c_c[0])), (
        f"duffy08 c-M relation not consistent, colossus to halox; \
            {c_c[0]} != {c_h}"
    )


@pytest.mark.parametrize("m_delta, z", test_mzs)
def test_klypin11(m_delta, z, return_vals=False):
    c_h = klypin11()(M=m_delta)
    c_c = ccon.modelKlypin11(m_delta, z)
    if return_vals:
        return c_h, c_c

    assert jnp.isclose(jnp.atleast_1d(c_h), jnp.atleast_1d(c_c[0])), (
        f"klypin11 c-M relation not consistent, colossus to halox; \
            {c_c[0]} != {c_h}"
    )


@pytest.mark.parametrize("cosmo", test_cosmos)
def test_prada12(cosmo, return_vals=False):
    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    model = prada12(cosmo=test_cosmos[cosmo][0])
    p12 = jax.jit(lambda m, z: model(m, z=z))
    c_h = jnp.array([p12(ms[i], zs[i]) for i in range(len(test_mzs))])

    cc.setCosmology(test_cosmos[cosmo][1])
    c_c = jnp.array(
        [ccon.modelPrada12(ms[i], zs[i]) for i in range(len(test_mzs))]
    )

    if return_vals:
        return c_h, c_c

    assert jnp.allclose(
        jnp.atleast_1d(c_h), jnp.atleast_1d(c_c), rtol=1e-3, atol=0.0
    ), (
        f"prada12 c-M relation not consistent, colossus to halox; \
            {c_c} != {c_h}"
    )


@pytest.mark.parametrize("cosmo", test_cosmos)
def test_child18all(cosmo, return_vals=False):
    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]
    model = child18all(cosmo=test_cosmos[cosmo][0])
    c18all = jax.jit(lambda m, z: model(m, z=z))
    c_h = jnp.array([c18all(ms[i], zs[i]) for i in range(len(test_mzs))])

    cc.setCosmology(test_cosmos[cosmo][1])
    c_c = jnp.array(
        [
            ccon.modelChild18(ms[i], zs[i], halo_sample="individual_all")
            for i in range(len(test_mzs))
        ]
    )

    if return_vals:
        return c_h, c_c

    assert jnp.allclose(
        jnp.atleast_1d(c_h[:, 0]),
        jnp.atleast_1d(c_c[:, 0]),
        rtol=1e-3,
        atol=0.0,
    ), (
        f"child18all c-M relation not consistent, colossus to halox; \
            {c_c[0:3, 0]} != {c_h[0:3, 0]}"
    )


@pytest.mark.parametrize("cosmo", test_cosmos)
def test_child18relaxed(cosmo, return_vals=False):
    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    model = child18relaxed(cosmo=test_cosmos[cosmo][0])

    c18rel = jax.jit(lambda m, z: model(m, z=z))

    c_h = jnp.array([c18rel(ms[i], zs[i]) for i in range(len(test_mzs))])

    cc.setCosmology(test_cosmos[cosmo][1])
    c_c = jnp.array(
        [
            ccon.modelChild18(ms[i], zs[i], halo_sample="individual_relaxed")
            for i in range(len(test_mzs))
        ]
    )

    if return_vals:
        return c_h, c_c

    assert jnp.allclose(
        jnp.atleast_1d(c_h[:, 0]),
        jnp.atleast_1d(c_c[:, 0]),
        rtol=1e-3,
        atol=0.0,
    ), (
        f"child18relaxed c-M relation not consistent, colossus to halox; \
            {c_c[0]} != {c_h}"
    )

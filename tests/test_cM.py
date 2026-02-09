import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
import colossus.halo.concentration as ccon
import colossus.cosmology.cosmology as cc
import halox
from halox.halo.cMrelation import duffy08, klypin11, prada12, child18all, child18relaxed


jax.config.update("jax_enable_x64", True)

rtol = 1e-2

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

G = halox.cosmology.G

@pytest.mark.parametrize("m_delta, z", test_mzs)
def test_duffy08(m_delta, z, return_vals=False): 
    c_h = duffy08()(M=m_delta, z=z)
    c_c = ccon.modelDuffy08(m_delta,z,mdef=f"200c")
    if return_vals:
        return  c_h, c_c
    
    assert jnp.isclose(jnp.atleast_1d(c_h),jnp.atleast_1d(c_c[0])), (
        f"duffy08 c-M relation not consistent, colossus to halox; {c_c[0]} != {c_h}"
    )

@pytest.mark.parametrize("m_delta, z", test_mzs)
def test_klypin11(m_delta, z, return_vals=False): 
    c_h = klypin11()(M=m_delta)
    c_c = ccon.modelKlypin11(m_delta, z)
    if return_vals:
        return  c_h, c_c
    
    assert jnp.isclose(jnp.atleast_1d(c_h),jnp.atleast_1d(c_c[0])), (
        f"klypin11 c-M relation not consistent, colossus to halox; {c_c[0]} != {c_h}"
    )

@pytest.mark.parametrize("m_delta, z", test_mzs)
@pytest.mark.parametrize("cosmo", test_cosmos)
def test_child18all(m_delta, z, cosmo, return_vals=False): 
    c_h = child18all()(M=m_delta, z=z, cosmo = test_cosmos[cosmo][0])
    c_c = ccon.modelChild18(m_delta, z, halo_sample = "individual_all")
    if return_vals:
        return  c_h, c_c
    
    assert jnp.isclose(jnp.atleast_1d(c_h),jnp.atleast_1d(c_c[0])), (
        f"child18all c-M relation not consistent, colossus to halox; {c_c[0]} != {c_h}"
    )

# @pytest.mark.parametrize("m_delta, z", test_mzs)
# @pytest.mark.parametrize("cosmo", test_cosmos)
# def test_child18relaxed(m_delta, z, cosmo, return_vals=False): 
#     c_h = child18relaxed()(M=m_delta, z=z, cosmo = cosmo)
#     c_c = ccon.modelChild18(m_delta, z)
#     if return_vals:
#         return  c_h, c_c
    
#     assert jnp.isclose(jnp.atleast_1d(c_h),jnp.atleast_1d(c_c[0])), (
#         f"child18relaxed c-M relation not consistent, colossus to halox; {c_c[0]} != {c_h}"
#     )
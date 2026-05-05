import jax
import jax.numpy as jnp
import halox

jax.config.update("jax_enable_x64", True)

# Shared setup for emulator tests
_cosmo = halox.cosmology.Planck18()
_emu = halox.emus.SigmaMEmulator()
_M = jnp.float64(1e14)
_z = jnp.float64(0.5)
_Ms = jnp.array([1e13, 1e14, 1e15])
_zs = jnp.array([0.0, 0.5, 2.0])


def test_convert_delta_parallel():
    m_delta, c_delta, z = (
        jnp.array([1e15, 1e14]),
        jnp.array([4.0, 5.5]),
        jnp.array([0.1, 1.0]),
    )
    cosmo_j = halox.cosmology.Planck18()

    delta_in, delta_out = 200.0, 500.0

    def convert_delta(m_delta, c_delta, z):
        nfw = halox.nfw.NFWHalo(m_delta, c_delta, z, cosmo_j, delta=delta_in)
        return nfw.to_delta(delta_out)

    convert_deltas = jax.vmap(convert_delta)
    res = jnp.array(convert_deltas(m_delta, c_delta, z))  # M, R, c
    assert jnp.all(jnp.isfinite(res)), f"Infinite predictions: {res}"


# --- pytree tests ---


def test_emulator_pytree_roundtrip():
    leaves, treedef = jax.tree_util.tree_flatten(_emu)
    emu2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert emu2.n_layers == _emu.n_layers
    assert jnp.allclose(emu2(_M, _z, _cosmo), _emu(_M, _z, _cosmo))


# --- JIT tests ---


def test_jit_sigma_M_emu():
    f = jax.jit(lambda m, z: halox.lss.sigma_M(m, z, _cosmo, emu=_emu))
    res = f(_M, _z)
    assert jnp.all(jnp.isfinite(res))


def test_jit_peak_height_emu():
    f = jax.jit(lambda m, z: halox.lss.peak_height(m, z, _cosmo, emu=_emu))
    res = f(_M, _z)
    assert jnp.all(jnp.isfinite(res))


def test_jit_tinker08_mass_function_emu():
    f = jax.jit(
        lambda m, z: halox.hmf.tinker08_mass_function(
            m, z, cosmo=_cosmo, emu=_emu
        )
    )
    res = f(_M, _z)
    assert jnp.all(jnp.isfinite(res))


def test_jit_tinker10_bias_emu():
    f = jax.jit(
        lambda m, z: halox.bias.tinker10_bias(m, z, cosmo=_cosmo, emu=_emu)
    )
    res = f(_M, _z)
    assert jnp.all(jnp.isfinite(res))


# --- vmap tests ---


def test_vmap_sigma_M_emu():
    f = jax.vmap(lambda m, z: halox.lss.sigma_M(m, z, _cosmo, emu=_emu))
    res = f(_Ms, _zs)
    assert jnp.all(jnp.isfinite(res))


def test_vmap_peak_height_emu():
    f = jax.vmap(lambda m, z: halox.lss.peak_height(m, z, _cosmo, emu=_emu))
    res = f(_Ms, _zs)
    assert jnp.all(jnp.isfinite(res))


def test_vmap_tinker08_mass_function_emu():
    f = jax.vmap(
        lambda m, z: halox.hmf.tinker08_mass_function(
            m, z, cosmo=_cosmo, emu=_emu
        )
    )
    res = f(_Ms, _zs)
    assert jnp.all(jnp.isfinite(res))


def test_vmap_tinker10_bias_emu():
    f = jax.vmap(
        lambda m, z: halox.bias.tinker10_bias(m, z, cosmo=_cosmo, emu=_emu)
    )
    res = f(_Ms, _zs)
    assert jnp.all(jnp.isfinite(res))


# --- grad tests ---


def test_grad_sigma_M_emu():
    f = jax.grad(lambda m: halox.lss.sigma_M(m, _z, _cosmo, emu=_emu))
    res = f(_M)
    assert jnp.all(jnp.isfinite(res))


def test_grad_peak_height_emu():
    f = jax.grad(lambda m: halox.lss.peak_height(m, _z, _cosmo, emu=_emu))
    res = f(_M)
    assert jnp.all(jnp.isfinite(res))

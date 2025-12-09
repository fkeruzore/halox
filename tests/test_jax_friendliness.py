import jax
import jax.numpy as jnp
import halox

jax.config.update("jax_enable_x64", True)


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

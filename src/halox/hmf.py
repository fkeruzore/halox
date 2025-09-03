import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
from . import cosmology, lss


def _tinker08_parameters(
    z: ArrayLike,
    cosmo: jc.Cosmology,
    delta_c: float = 200.0,
) -> Array:
    """Get Tinker08 mass function parameters for given overdensity.

    Parameters
    ----------
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology
    delta_c : float
        Overdensity threshold, default 200.0

    Returns
    -------
    Array
        Parameters [A, a, b, c] for Tinker08 mass function
    """
    # Table 2 from Tinker et al. 2008 - exact values
    delta_vals = jnp.array(
        [200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0]
    )
    A_vals = jnp.array(
        [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260]
    )
    a_vals = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
    b_vals = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
    c_vals = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])

    z = jnp.asarray(z)
    # Critical to mean overdensity
    delta_m = lss.overdensity_c_to_m(delta_c, z, cosmo)

    # Use linear interpolation in log space
    A_0 = jnp.interp(delta_m, delta_vals, A_vals)
    a_0 = jnp.interp(delta_m, delta_vals, a_vals)
    b_0 = jnp.interp(delta_m, delta_vals, b_vals)
    c_0 = jnp.interp(delta_m, delta_vals, c_vals)

    # Apply redshift evolution
    A_z = A_0 * (1.0 + z) ** (-0.14)
    a_z = a_0 * (1.0 + z) ** (-0.06)
    alpha = 10 ** (-1 * (0.75 / jnp.log10(delta_m / 75)) ** 1.2)
    b_z = b_0 * (1.0 + z) ** (-alpha)

    return jnp.array([A_z, a_z, b_z, c_0])


def tinker08_f_sigma(
    M: ArrayLike,
    z: ArrayLike,
    cosmo: jc.Cosmology,
    delta_c: float = 200.0,
) -> Array:
    """Tinker08 multiplicity function :math:`f(\\sigma)`.

    Parameters
    ----------
    M : Array
        Halo mass [h-1 Msun]
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology
    delta_c : float
        Overdensity threshold, default 200.0

    Returns
    -------
    Array
        Mass function multiplicity
    """
    M = jnp.asarray(M)
    z = jnp.asarray(z)
    sigma = lss.sigma_M(M, z, cosmo)
    A, a, b, c = _tinker08_parameters(z, cosmo, delta_c)
    return A * ((b / sigma) ** a + 1.0) * jnp.exp(-c / sigma**2)


def tinker08_mass_function(
    M: ArrayLike,
    z: ArrayLike,
    cosmo: jc.Cosmology,
    delta_c: float = 200.0,
) -> Array:
    """Tinker08 halo mass function :math:`dn/d\\ln M`.

    Parameters
    ----------
    M : Array
        Halo mass [h-1 Msun]
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology, default Planck18
    delta_c : float
        Overdensity threshold, default 200.0

    Returns
    -------
    Array
        Mass function [h3 Mpc-3]
    """
    M = jnp.atleast_1d(M)
    z = jnp.asarray(z)

    # Background density
    rho_m = cosmo.Omega_m * cosmology.critical_density(0.0, cosmo)

    # Multiplicity function with redshift evolution
    f_sigma = tinker08_f_sigma(M, z, cosmo, delta_c)

    # Use autodiff to compute d ln sigma / dM
    d_ln_sigma_inv = jax.grad(
        lambda M: jnp.log(1.0 / lss.sigma_M(M, z, cosmo))
    )

    dn_dm = f_sigma * (rho_m / M) * jax.vmap(d_ln_sigma_inv)(M)

    return jnp.squeeze(M * dn_dm)

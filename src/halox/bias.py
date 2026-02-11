from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
from . import lss


def _tinker10_parameters(
    z: ArrayLike,
    cosmo: jc.Cosmology,
    delta_c: float = 200.0,
) -> Array:
    """Get Tinker10 halo bias parameters for given overdensity.

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
        Parameters [A, a, B, C] for Tinker10 bias function
    """
    z = jnp.asarray(z)

    # Convert critical to mean overdensity
    delta_m = lss.overdensity_c_to_m(delta_c, z, cosmo)

    # y parameter from log10(Delta_halo)
    y = jnp.log10(delta_m)

    # Parameter calculations following Tinker et al. 2010
    A = 1.0 + 0.24 * y * jnp.exp(-((4.0 / y) ** 4))
    a = 0.44 * y - 0.88
    B = 0.183
    C = 0.019 + 0.107 * y + 0.19 * jnp.exp(-((4.0 / y) ** 4))

    return jnp.array([A, a, B, C])


def tinker10_bias(
    M: ArrayLike,
    z: ArrayLike,
    cosmo: jc.Cosmology,
    delta_c: float = 200.0,
    delta_sc: float = 1.686,
    n_k_int: int = 5000,
) -> Array:
    """Tinker10 halo bias function.

    Linear halo bias as calibrated by Tinker et al. 2010.

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
    delta_sc : float
        Spherical collapse threshold, default 1.686
    n_k_int : int
        Number of k-space integration points for :math:`\\sigma(R,z)`,
        default 5000

    Returns
    -------
    Array
        Linear halo bias
    """
    M = jnp.asarray(M)
    z = jnp.asarray(z)

    # Calculate peak height nu = delta_sc / sigma(M,z)
    nu = lss.peak_height(M, z, cosmo, n_k_int=n_k_int)

    # Get parameters
    A, a, B, C = _tinker10_parameters(z, cosmo, delta_c)

    # Fixed parameters
    b = 1.5
    c = 2.4

    # Tinker10 bias formula
    bias = 1.0 - A * (nu**a) / (nu**a + delta_sc**a) + B * nu**b + C * nu**c

    return bias

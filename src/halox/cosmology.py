from functools import partial
from jax import Array
import jax.numpy as jnp
import jax_cosmo as jc


G = 4.30091727e-9  # km^2 Mpc Msun^-1 s^-2

# Planck 2018 cosmology parameters
Planck18 = partial(
    jc.Cosmology,
    Omega_c=0.26069,
    Omega_b=0.04897,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,
)()


def hubble_parameter(z: Array, cosmo: jc.Cosmology):
    """Computes the Hubble parameter :math:`H(z)` at a given redshift
    for a given cosmology.

    Parameters
    ----------
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology

    Returns
    -------
    Array
        Hubble parameter at z [km s-1 Mpc-1]
    """
    a = jc.utils.z2a(z)
    return cosmo.h * jc.background.H(cosmo, a)


def critical_density(z: Array, cosmo: jc.Cosmology):
    """Computes the Universe critical density :math:`\\rho_c(z)` at a
    given redshift for a given cosmology.

    Parameters
    ----------
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology

    Returns
    -------
    Array
        Critical density at z [Msun Mpc-3]
    """
    return (3 * hubble_parameter(z, cosmo) ** 2) / (8 * jnp.pi * G)

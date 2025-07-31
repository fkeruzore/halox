from functools import partial
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.background as jcb


G = 4.30091727e-9  # km^2 Mpc Msun^-1 s^-2
c = 299_792.458  # km s^-1

# Planck 2018 cosmology parameters
Planck18 = partial(
    jc.Cosmology,
    Omega_c=round(0.11933 / 0.6766**2, 5),
    Omega_b=round(0.02242 / 0.6766**2, 5),
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,
)


def hubble_parameter(z: ArrayLike, cosmo: jc.Cosmology) -> Array:
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
    z = jnp.asarray(z)
    a = 1.0 / (1.0 + z)
    return cosmo.h * jcb.H(cosmo, a)


def critical_density(z: ArrayLike, cosmo: jc.Cosmology) -> Array:
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
        Critical density at z [h2 Msun Mpc-3]
    """
    z = jnp.asarray(z)
    rho_c = (3 * hubble_parameter(z, cosmo) ** 2) / (8 * jnp.pi * G)
    return rho_c / (cosmo.h**2)


def differential_comoving_volume(z: ArrayLike, cosmo: jc.Cosmology) -> Array:
    """Computes the differential comoving volume element per solid
    angle, :math:`{\\rm d}V_c / {\\rm d}\\Omega {\\rm d}z`, at a given
    redshift for a given cosmology.

    Parameters
    ----------
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology

    Returns
    -------
    Array
        Differential comoving volume element at z [h-3 Mpc3 sr-1]
    """
    z = jnp.asarray(z)
    a = 1.0 / (1.0 + z)
    hubble_dist = c / 100  # h-1 Mpc
    ang_dist = jcb.angular_diameter_distance(cosmo, a)  # h-1 Mpc
    return (
        hubble_dist
        * (1.0 + z) ** 2
        * (ang_dist**2)
        / jnp.sqrt(jcb.Esqr(cosmo, a))
    )  # h-3 Mpc3 sr-1

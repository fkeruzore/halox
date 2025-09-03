import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
from . import cosmology


# jax-cosmo power spectra differ from colossus at the 0.3% level, which
# results in %-level discrepancies in HMF predictions. This fudge factor
# solves that.
_jax_cosmo_pk_corr = 1.0 / 1.0030


def mass_to_lagrangian_radius(M: ArrayLike, cosmo: jc.Cosmology) -> Array:
    """Convert mass to Lagrangian radius.

    Computes the radius of a sphere containing mass M at the mean matter
    density of the universe at z=0.

    Parameters
    ----------
    M : Array
        Mass [h-1 Msun]
    cosmo : jc.Cosmology
        Underlying cosmology

    Returns
    -------
    Array
        Lagrangian radius [h-1 Mpc]
    """
    M = jnp.asarray(M)
    rho_crit_0 = cosmology.critical_density(0.0, cosmo)
    rho_m0 = cosmo.Omega_m * rho_crit_0

    return (3.0 * M / (4.0 * jnp.pi * rho_m0)) ** (1.0 / 3.0)


def overdensity_c_to_m(
    delta_c: float, z: ArrayLike, cosmo: jc.Cosmology
) -> Array:
    """Convert critical overdensity to mean overdensity.

    Parameters
    ----------
    delta_c : float
        Overdensity with respect to critical density
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology

    Returns
    -------
    Array
        Overdensity with respect to mean matter density
    """
    z = jnp.asarray(z)
    rho_m = (
        cosmo.Omega_m * cosmology.critical_density(0.0, cosmo) * (1 + z) ** 3
    )
    rho_c = cosmology.critical_density(z, cosmo)
    return delta_c * rho_c / rho_m


def sigma_R(
    R: ArrayLike,
    z: ArrayLike,
    cosmo: jc.Cosmology,
    k_min: float = 1e-5,
    k_max: float = 1e2,
) -> Array:
    """Compute RMS variance of density fluctuations in spheres
    of radius R at redshift z.

    Parameters
    ----------
    R : Array
        Radius [h-1 Mpc]
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology
    k_min : float
        Minimum k for integration [h Mpc-1], default 1e-5
    k_max : float
        Maximum k for integration [h Mpc-1], default 1e2

    Returns
    -------
    Array
        RMS variance :math:`\\sigma(R,z)`
    """
    R = jnp.asarray(R)
    z = jnp.asarray(z)
    # Create k array for integration (already in h/Mpc)
    k = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), 5000)

    # Power spectrum at redshift z
    a = 1.0 / (1.0 + z)
    pk = jc.power.linear_matter_power(cosmo, k, a=a)  # type: ignore
    pk *= _jax_cosmo_pk_corr  # consistency with colossus

    # Window function for spherical top-hat
    # Handle broadcasting for both scalar and array R
    kR = k * R[..., None]  # Broadcasting works for both scalar and array R
    W = jnp.where(
        kR < 1e-3,
        1.0 - kR**2 / 10.0,  # Small kR approximation
        3.0 * (jnp.sin(kR) - kR * jnp.cos(kR)) / kR**3,
    )

    # Integrate: sigma^2 = (1/2pi^2) * int k^2 P(k) W^2(kR) dk
    integrand = k**2 * pk * W**2
    sigma2 = jnp.trapezoid(integrand, k, axis=-1) / (2 * jnp.pi**2)

    return jnp.sqrt(sigma2)


def sigma_M(M: ArrayLike, z: ArrayLike, cosmo: jc.Cosmology) -> Array:
    """Compute RMS variance of density fluctuations within the
    Lagrangian radius of a halo with mass M at redshift z.

    Parameters
    ----------
    M : Array
        Mass [h-1 Msun]
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology

    Returns
    -------
    Array
        RMS variance :math:`\\sigma(M,z)`
    """
    M = jnp.asarray(M)
    z = jnp.asarray(z)
    R = mass_to_lagrangian_radius(M, cosmo)
    return sigma_R(R, z, cosmo)

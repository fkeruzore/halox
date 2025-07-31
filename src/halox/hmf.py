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
    delta_m = overdensity_c_to_m(delta_c, z, cosmo)

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
    sigma = sigma_M(M, z, cosmo)
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
    d_ln_sigma_inv = jax.grad(lambda M: jnp.log(1.0 / sigma_M(M, z, cosmo)))

    dn_dm = f_sigma * (rho_m / M) * jax.vmap(d_ln_sigma_inv)(M)

    return jnp.squeeze(M * dn_dm)

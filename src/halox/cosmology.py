from functools import partial
from jax import Array, grad
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.background as jcb
from typing import Callable


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


def stack_cosmologies(cosmos: list[jc.Cosmology]) -> jc.Cosmology:
    """Stacks a list of cosmologies into a single batched cosmology.

    Each cosmological parameter of the returned object is a 1-D array
    whose i-th element corresponds to that parameter in the i-th input
    cosmology. This is useful for vectorized evaluation of functions
    over a set of cosmologies via JAX broadcasting.

    Parameters
    ----------
    cosmos : list[jc.Cosmology]
        List of cosmologies to stack.

    Returns
    -------
    jc.Cosmology
        A single cosmology whose parameters are 1-D arrays of length
        ``len(cosmos)``.
    """
    return jc.Cosmology(
        Omega_b=jnp.array([c.Omega_b for c in cosmos]),
        Omega_c=jnp.array([c.Omega_c for c in cosmos]),
        sigma8=jnp.array([c.sigma8 for c in cosmos]),
        h=jnp.array([c.h for c in cosmos]),
        n_s=jnp.array([c.n_s for c in cosmos]),
        Omega_k=jnp.array([c.Omega_k for c in cosmos]),
        w0=jnp.array([c.w0 for c in cosmos]),
        wa=jnp.array([c.wa for c in cosmos]),
    )


def sensitivity(
    func: Callable[[jc.Cosmology], Array],
    cosmo: jc.Cosmology,
) -> list[str]:
    """Computes the sensitivity of a scalar function to each
    cosmological parameter using autodifferentiation. Gradients are
    evaluated at the given cosmology and at two perturbed cosmologies
    (all parameters scaled by 0.9 and 1.1) to avoid missing
    sensitivity at local extrema.

    Parameters
    ----------
    func : Callable[[jc.Cosmology], Array]
        A function that takes a :class:`jax_cosmo.Cosmology` and
        returns a scalar.
    cosmo : jc.Cosmology
        Underlying cosmology

    Returns
    -------
    list[str]
        Names of cosmological parameters to which ``func`` is
        sensitive.
    """
    param_names = [
        "Omega_c",
        "Omega_b",
        "h",
        "n_s",
        "sigma8",
        "Omega_k",
        "w0",
        "wa",
    ]
    cosmo_l = jc.Cosmology(**{p: 0.9 * getattr(cosmo, p) for p in param_names})
    cosmo_u = jc.Cosmology(**{p: 1.1 * getattr(cosmo, p) for p in param_names})
    grad_func = grad(func)
    grads_c = grad_func(cosmo)
    grads_l = grad_func(cosmo_l)
    grads_u = grad_func(cosmo_u)
    sensitive = [
        p
        for p in param_names
        if any(getattr(g, p) != 0.0 for g in (grads_c, grads_l, grads_u))
    ]
    return sensitive


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

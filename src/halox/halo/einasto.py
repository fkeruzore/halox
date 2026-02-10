from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
import jax.scipy as jsp
from halox import lss
from . import nfw

from ..cosmology import G
from .. import cosmology

# TODO: 
# Add velocity dispersion and surface density (allign capabilities with NFW)
# Add potential function to the jax friendlieness test (see how gamma funcs behave)

class EinastoHalo:
    """
    Properties of a dark matter halo following an Einasto profile.
    Units in Mpc and solar masses

    Parameters
    ----------
    m_delta: float
        Mass at overdensity `delta` [h-1 Msun]
    c_delta: float
        Concentration at overdensity `delta`
    z: float
        Redshift
    cosmo: jc.Cosmology
        Underlying cosmology
    delta: float
        Density contrast in units of critical density at redshift z,
        defaults to 200.
    """
    # Reference: https://ui.adsabs.harvard.edu/abs/2012A%26A...540A..70R/abstract

    def __init__(
        self,
        m_delta: ArrayLike,
        c_delta: ArrayLike,
        z: ArrayLike,
        alpha: ArrayLike,
        cosmo: jc.Cosmology,
        delta: float = 200,
    ):
        self.m_delta = jnp.asarray(m_delta)
        self.c_delta = jnp.asarray(c_delta)
        self.z = jnp.asarray(z)
        self.alpha = jnp.asarray(alpha)
        self.delta = delta
        self.cosmo = cosmo

        # Potential future choice? This is from a paper, fairly ubiquitous, need to cite
        # Use the formula :math:`\\Delta_{vir} = 18 * \\pi ^ 2 + 82x -39x ^ 2`
        # :math: 'x = \\Omega_m (z) -1'
        # delta_vir = 18*jnp.pi**2 + 82*(cosmo.Omega_m(self.z) - 1) - 39*(cosmo.Omega_m(self.z) - 1)**2

        mean_rho = delta * cosmology.critical_density(self.z, cosmo)
        self.r_delta = (3 * self.m_delta / (4 * jnp.pi * mean_rho)) ** (1 / 3)
        self.Rs = self.r_delta / self.c_delta  # final point of certainty
        rho0_denum = (
            4
            * jnp.pi
            * self.Rs**3
            * jnp.exp(2 / self.alpha)
            / self.alpha
            * (2 / self.alpha) ** (-3 / self.alpha)
            * jsp.special.gammainc(
                3 / self.alpha,
                2 / self.alpha * (self.r_delta / self.Rs) ** self.alpha,
            )
            * jsp.special.gamma(3 / self.alpha)
        )
        self.rho0 = self.m_delta / rho0_denum  # output is rho_-2, in units of

    def density(self, r: ArrayLike) -> Array:
        """Einasto density profile :math:`\\rho(r)`.

        Parameters
        ----------
        r : Array [h-1 Mpc]
            Radius

        Returns
        -------
        Array [h2 Msun Mpc-3]
            Density at radius `r`
        """
        r = jnp.asarray(r)
        return self.rho0 * jnp.exp(
            -2 / self.alpha * ((r / self.Rs) ** self.alpha - 1)
        )

    def enclosed_mass(self, r: ArrayLike) -> Array:
        """Enclosed mass profile :math:`M(<r)`.

        Parameters
        ----------
        r : Array [h-1 Mpc]
            Radius

        Returns
        -------
        Array [h-1 Msun]
            Enclosed mass at radius `r`
        """

        r = jnp.asarray(r)
        return (
            (4 * jnp.pi * self.Rs**3)
            * self.rho0
            * jnp.exp(2 / self.alpha)
            / self.alpha
            * (2 / self.alpha) ** (-3 / self.alpha)
            * jsp.special.gammainc(
                3 / self.alpha, 2 / self.alpha * (r / self.Rs) ** self.alpha
            )
            * jsp.special.gamma(3 / self.alpha)
        )

    def circular_velocity(self, r: ArrayLike) -> Array:
        """Circular velocity profile :math:`v_c(r)`.

        The circular velocity is related to the enclosed mass by:
        :math:`v_c^2(r) = GM(<r)/r`

        Parameters
        ----------
        r : Array [h-1 Mpc]
            Radius

        Returns
        -------
        Array [km s-1]
            Circular velocity at radius `r`
        """
        r = jnp.asarray(r)
        return jnp.sqrt(G * self.enclosed_mass(r) / r)

    def potential(self, r: ArrayLike) -> Array:
        # Working test: Possible addition to JAX friendlieness test
        """Potential profile :math:`\\phi(r)`.

        Parameters
        ----------
        r : Array [h-1 Mpc]
            Radius

        Returns
        -------
        Array [km2 s-2]
            Potential at radius `r`
        """
        r = jnp.asarray(r)
        prefact = -4 * jnp.pi * G * self.rho0 * jnp.exp(2 / self.alpha)
        a2 = 2.0 / self.alpha
        a3 = 3.0 / self.alpha

        s = (2.0 / self.alpha) ** (1.0 / self.alpha) * r / self.Rs
        x = s**self.alpha

        gamma2 = jsp.special.gamma(a2)
        gamma3 = jsp.special.gamma(a3)

        lower3 = (
            jsp.special.gammainc(a3, x) * gamma3 / (s / r) ** 2 / self.alpha
        )
        upper2 = (
            gamma2
            * (1.0 - jsp.special.gammainc(a2, x))
            / (s / r) ** 2
            / self.alpha
        )

        phi = prefact * (lower3 / s + upper2)
        return phi

    def to_delta(self, delta_new: float) -> tuple[Array, Array, Array]:
        """Convert halo properties to a different overdensity definition.

        Parameters
        ----------
        delta_new : float
            New density contrast in units of critical density at redshift z

        Returns
        -------
        Array [h-1 Msun]
            Mass at new overdensity
        Array [h-1 Mpc]
            Radius at new overdensity
        Array
            Concentration at new overdensity
        
        Notes
        -----
        This conversion assumes an NFW profile.
        """
        output = nfw.delta_delta(
            self.m_delta,
            self.c_delta,
            self.z,
            self.cosmo,
            self.delta,
            delta_new,
        )
        # self.m_delta = output[0] #removed, must instantiate new halo after using this function if you want all the properties of the new halo
        # self.r_delta = output[1]
        # self.c_delta = output[2]
        # self.delta = delta_new
        return output


# TODO
# Need to add velocity dispersion, surface density, to_delta, and lsq


# Passing alpha is optional since it can also be estimated from the
# Gao et al. 2008 relation between alpha and peak height. This relation was calibrated for
# nu_vir, so if the given mass definition is not 'vir' we convert the given mass to Mvir
# assuming an NFW profile with the given mass and concentration. This leads to a negligible
# inconsistency, but solving for the correct alpha iteratively would be much slower.
def a_from_nu(
    M: ArrayLike,  # this should be the virial mass, currently not, will need to change, see profile_einasto for more details
    z: ArrayLike,
    cosmo: jc.Cosmology,
    n_k_int: int = 5000,
    delta_sc: float = 1.686,
) -> Array:
    """
    Returns the alpha parameter from the peak height value of the halo.

    Parameters
    ----------
    M : Array
        Virial mass [h-1 Msun]
    z : Array
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology
    n_k_int : int
        Number of k-space integration points for :math:`\\sigma(R,z)`,
        default 5000
    delta_sc: float
        Required overdensity for spherical collapse, usually 1.686 (tophat)

    Returns
    -------
    Array
        Returns alpha for halos

    Notes
    -----
    This assumes that the input mass is the virial mass.
    """
    nu = lss.peak_height(M, z, cosmo, n_k_int, delta_sc)
    alpha = 0.155 + 0.0095 * nu**2
    return alpha

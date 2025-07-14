from jax import Array
import jax.numpy as jnp
import jax_cosmo as jc

from .cosmology import Planck18, G
from . import cosmology


class NFWHalo:
    """
    Properties of a dark matter halo following a Navarro-Frenk-White
    density profile.

    Parameters
    ----------
    m_delta: float
        Mass at overdensity `delta` [Msun]
    c_delta: float
        Concentration at overdensity `delta`
    z: float
        Redshift
    delta: float
        Density contrast in units of critical density at redshift z,
        defaults to 200.
    cosmo: jc.Cosmology
        Underlying cosmology, defaults to Planck 2018.
    """

    def __init__(
        self,
        m_delta: float,
        c_delta: float,
        z: float,
        delta: float = 200.0,
        cosmo: jc.Cosmology = Planck18,
    ):
        self.m_delta = m_delta
        self.c_delta = c_delta
        self.delta = delta
        self.z = z
        self.cosmo = cosmo

        mean_rho = delta * cosmology.critical_density(z, cosmo)
        self.Rdelta = (3 * m_delta / (4 * jnp.pi * mean_rho)) ** (1 / 3)
        self.Rs = self.Rdelta / c_delta
        rho0_denum = 4 * jnp.pi * self.Rs**3
        rho0_denum *= jnp.log(1 + c_delta) - c_delta / (1 + c_delta)
        self.rho0 = m_delta / rho0_denum

    def density(self, r: Array) -> Array:
        """NFW density profile :math:`\\rho(r)`.

        Parameters
        ----------
        r : Array [Mpc]
            Radius

        Returns
        -------
        Array [Msun Mpc-3]
            Density at radius `r`
        """
        return self.rho0 / (r / self.Rs * (1 + r / self.Rs) ** 2)

    def enclosed_mass(self, r: Array) -> Array:
        """Enclosed mass profile :math:`M(<r)`.

        Parameters
        ----------
        r : Array [Mpc]
            Radius

        Returns
        -------
        Array [Msun]
            Enclosed mass at radius `r`
        """
        prefact = 4 * jnp.pi * self.rho0 * self.Rs**3
        return prefact * (jnp.log(1 + r / self.Rs) - r / (r + self.Rs))

    def potential(self, r: Array) -> Array:
        """Potential profile :math:`\\phi(r)`.

        Parameters
        ----------
        r : Array [Mpc]
            Radius

        Returns
        -------
        Array [km2 s-2]
            Potential at radius `r`
        """
        # G = G.to("km2 Mpc Msun-1 s-2").value
        prefact = -4 * jnp.pi * G * self.rho0 * self.Rs**3
        return prefact * jnp.log(1 + r / self.Rs) / r

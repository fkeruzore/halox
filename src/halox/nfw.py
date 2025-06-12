from jax import Array
import jax.numpy as jnp
import jax_cosmo as jc

from .cosmology import Planck18, G


class NFW:
    def __init__(
        self,
        MDelta: float,
        cDelta: float,
        Delta: str,
        z: float,
        cosmo: jc.Cosmology = Planck18,
    ):
        self.MDelta = MDelta
        self.cDelta = cDelta
        if Delta == "200c":
            mean_rho = 200 * jc.background.rho_crit(cosmo, z)
        elif Delta == "500c":
            mean_rho = 500 * jc.background.rho_crit(cosmo, z)
        else:
            raise ValueError(
                f"{Delta=} not supported yet, must be either '200c' or '500c'"
            )
        self.RDelta = (3 * MDelta / (4 * jnp.pi * mean_rho)) ** (1 / 3)
        self.Rs = self.RDelta / cDelta
        rho0_denum = 4 * jnp.pi * self.Rs**3
        rho0_denum *= jnp.log(1 + cDelta) - cDelta / (1 + cDelta)
        self.rho0 = MDelta / rho0_denum

    def density(self, r: Array) -> Array:
        """NFW density profile
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
        """Enclosed mass profile
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
        """Potential profile
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

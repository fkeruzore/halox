from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
from jaxopt import LBFGSB

from ..cosmology import G
from .. import cosmology


class NFWHalo:
    """
    Properties of a dark matter halo following a Navarro-Frenk-White
    density profile.

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

    def __init__(
        self,
        m_delta: ArrayLike,
        c_delta: ArrayLike,
        z: ArrayLike,
        cosmo: jc.Cosmology,
        delta: float = 200.0,
    ):
        self.m_delta = jnp.asarray(m_delta)
        self.c_delta = jnp.asarray(c_delta)
        self.z = jnp.asarray(z)
        self.delta = delta
        self.cosmo = cosmo

        mean_rho = delta * cosmology.critical_density(self.z, cosmo)
        self.r_delta = (3 * self.m_delta / (4 * jnp.pi * mean_rho)) ** (1 / 3)
        self.Rs = self.r_delta / self.c_delta
        rho0_denum = 4 * jnp.pi * self.Rs**3
        rho0_denum *= jnp.log(1 + self.c_delta) - self.c_delta / (
            1 + self.c_delta
        )
        self.rho0 = self.m_delta / rho0_denum

    def density(self, r: ArrayLike) -> Array:
        """NFW density profile :math:`\\rho(r)`.

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
        return self.rho0 / (r / self.Rs * (1 + r / self.Rs) ** 2)

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
        prefact = 4 * jnp.pi * self.rho0 * self.Rs**3
        return prefact * (jnp.log(1 + r / self.Rs) - r / (r + self.Rs))

    def potential(self, r: ArrayLike) -> Array:
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
        # G = G.to("km2 Mpc Msun-1 s-2").value
        prefact = -4 * jnp.pi * G * self.rho0 * self.Rs**3
        return prefact * jnp.log(1 + r / self.Rs) / r

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
        m_enc = self.enclosed_mass(r)
        return jnp.sqrt(G * m_enc / r)

    def velocity_dispersion(self, r: ArrayLike) -> Array:
        """Radial velocity dispersion profile :math:`\\sigma_r(r)`.

        Uses the Jeans equation assuming isotropic orbits:
        :math:`\\sigma_r^2(r) = \\frac{1}{\\rho(r)} \\int_r^{\\infty}
        \\rho(s) \\frac{GM(<s)}{s^2} ds`

        For NFW halos, this has an analytical solution.

        Parameters
        ----------
        r : Array [h-1 Mpc]
            Radius

        Returns
        -------
        Array [km s-1]
            Radial velocity dispersion at radius `r`
        """
        r = jnp.asarray(r)
        x = r / self.Rs

        # Analytical solution for NFW velocity dispersion
        # From Lokas & Mamon 2001, Eq. 16
        g_x = (jnp.log(1 + x) - x / (1 + x)) / x**2

        # Factor involving concentration
        c = self.c_delta
        gc = jnp.log(1 + c) - c / (1 + c)

        # Velocity dispersion squared
        sigma_r2 = (
            G * self.m_delta * gc * g_x / (self.r_delta * x * (1 + x) ** 2)
        )

        return jnp.sqrt(sigma_r2)

    def surface_density(self, r: ArrayLike) -> Array:
        """Projected surface density profile :math:`\\Sigma(r)`.

        The projected surface density is obtained by integrating the 3D
        density profile along the line of sight:
        :math:`\\Sigma(r) = 2 \\int_r^{\\infty} \\frac{\\rho(s) s ds}
        {\\sqrt{s^2 - r^2}}`

        For NFW halos, this has an analytical solution.

        Parameters
        ----------
        r : Array [h-1 Mpc]
            Projected radius

        Returns
        -------
        Array [h Msun Mpc-2]
            Surface density at projected radius `r`
        """
        r = jnp.asarray(r)
        x = r / self.Rs

        # Analytical solution for NFW surface density
        # From Bartelmann 1996, Eq. 13
        prefact = 2 * self.rho0 * self.Rs

        # Handle different regimes for numerical stability
        def f(x):
            return jnp.where(
                x < 1.0,
                # x < 1 case
                (
                    1
                    - 2
                    * jnp.arctanh(jnp.sqrt((1 - x) / (1 + x)))
                    / jnp.sqrt(1 - x**2)
                )
                / (x**2 - 1),
                jnp.where(
                    x > 1.0,
                    # x > 1 case
                    (
                        1
                        - 2
                        * jnp.arctan(jnp.sqrt((x - 1) / (1 + x)))
                        / jnp.sqrt(x**2 - 1)
                    )
                    / (x**2 - 1),
                    # x = 1 case
                    1.0 / 3.0,
                ),
            )

        return prefact * f(x)

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
        """

        # Target density for the new overdensity definition
        rho_c = cosmology.critical_density(self.z, self.cosmo)
        target_density = delta_new * rho_c

        # Normalized objective function (critical for numerical stability)
        def lsq(r_new):
            m_enc = self.enclosed_mass(r_new[0])
            mean_density = m_enc / (4.0 * jnp.pi * r_new[0] ** 3 / 3.0)
            # Normalize by target_density to get dimensionless objective
            return ((mean_density - target_density) / target_density) ** 2

        # Initial guess based on scaling relation
        r0 = jnp.array([self.r_delta * (self.delta / delta_new) ** (1 / 3)])

        # Bounds for the optimization
        lower = jnp.array([0.01 * self.r_delta])
        upper = jnp.array([10.0 * self.r_delta])
        bounds = (lower, upper)

        # Use jaxopt LBFGSB optimizer
        optimizer = LBFGSB(fun=lsq, tol=1e-12)
        result = optimizer.run(r0, bounds=bounds)

        r_new = result.params[0]

        # Calculate new mass and concentration
        m_new = self.enclosed_mass(r_new)
        c_new = r_new / self.Rs

        return m_new, r_new, c_new

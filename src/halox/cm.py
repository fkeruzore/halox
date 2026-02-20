from dataclasses import dataclass
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from halox import lss, cosmology

jax.config.update("jax_enable_x64", True)

# currently implementing all of these for 200c


class duffy08:
    """
    Duffy et al. (2008) c-M relation using :math:`M_{200c}`.

    https://ui.adsabs.harvard.edu/abs/2008MNRAS.390L..64D/abstract

    Parameters
    ----------
    M : ArrayLike
        Halo mass :math:`M_{200c}` [Msun].
    z : ArrayLike
        Redshift.

    Returns
    -------
    c : ArrayLike
        Concentration :math:`c_{200c}`.

    Notes
    -----
    - The functional form is

      .. math::

        c(M, z) = A \\left(\\frac{M}{M_0}\\right)^B (1 + z)^C

      with :math:`M_0 = 2 \\times 10^{12}\\, M_\\odot`.
    - Calibrated on WMAP5.
    - Valid for :math:`10^{11} \\le M \\le 10^{15}\\, M_\\odot`,
      :math:`0 \\le z \\le 2`.
    """

    name = "duffy08"
    m_min = 1e11
    m_max = 1e15
    z_min = 0.0
    z_max = 2.0

    A = 5.71
    B = -0.084
    C = -0.47

    def __call__(
        self, M: ArrayLike, z: ArrayLike
    ) -> ArrayLike:  # could I need cosmo?
        # valid:bool = (
        # (M >= self.m_min) &
        # (M <= self.m_max) &
        # (z >= self.z_min) &
        # (z <= self.z_max)
        # )
        M0 = 2e12
        return self.A * (M / M0) ** self.B * (1 + z) ** self.C  # , valid


@dataclass
class prada12:
    """
    Prada et al. (2012) c-M relation using :math:`M_{200c}`.

    http://adsabs.harvard.edu/abs/2012MNRAS.423.3018P

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology used to compute :math:`\\sigma(M, z)`.
    M : ArrayLike
        Halo mass :math:`M_{200c}` [h-1 Msun].
    z : ArrayLike
        Redshift.

    Returns
    -------
    c : ArrayLike
        Concentration :math:`c_{200c}`.

    Notes
    -----
    - Predicts concentration as a function of peak height via
      math:`\\sigma(M, z)`, capturing an upturn in concentration
      at high masses.
    - Cosmology-dependent through :math:`\\sigma(M, z)`.
    - Valid for any cosmology, :math:`M > 0`, :math:`z \\ge 0`.
    """

    name: str = "prada12"
    m_min: float = 0
    m_max: float = jnp.inf
    z_min: float = 0
    z_max: float = jnp.inf

    def __init__(self, cosmo: jc.Cosmology):
        self.cosmo = cosmo

    def __call__(
        self,
        M: ArrayLike,
        z: ArrayLike,
    ) -> ArrayLike:
        # valid:bool = (
        # (M >= self.m_min) &
        # (M <= self.m_max) &
        # (z >= self.z_min) &
        # (z <= self.z_max)
        # )
        def cmin(x):
            return 3.681 + (5.033 - 3.681) * (
                1.0 / jnp.pi * jnp.arctan(6.948 * (x - 0.424)) + 0.5
            )

        def smin(x):
            return 1.047 + (1.646 - 1.047) * (
                1.0 / jnp.pi * jnp.arctan(7.386 * (x - 0.526)) + 0.5
            )

        a = (1 + z) ** -1
        x = (self.cosmo.Omega_de / self.cosmo.Omega_m) ** (1.0 / 3.0) * a

        B0 = cmin(x) / cmin(1.393)
        B1 = smin(x) / smin(1.393)

        temp_sig = lss.sigma_M(M, z, self.cosmo, k_max=1e3, n_k_int=20000)
        temp_sigp = temp_sig * B1
        temp_C = (
            2.881
            * ((temp_sigp / 1.257) ** 1.022 + 1)
            * jnp.exp(0.060 / temp_sigp**2)
        )
        c = B0 * temp_C

        return c  # , valid


class klypin11:
    """
    Klypin et al. (2011) c-M relation using
    :math:`M_{\\rm vir}` at :math:`z = 0`.

    http://adsabs.harvard.edu/abs/2011ApJ...740..102K

    Parameters
    ----------
    M : ArrayLike
        Halo mass :math:`M_{\\rm vir}` [h-1 Msun].

    Returns
    -------
    c : ArrayLike
        Concentration :math:`c_{\\rm vir}`.

    Notes
    -----
    - Only implements the :math:`z = 0` case from the original
      paper, which also provides redshift evolution.
    - Calibrated on WMAP7.
    - Valid for :math:`3 \\times 10^{10}` to
      :math:`5 \\times 10^{14}\\, h^{-1} M_\\odot`,
      :math:`z = 0` only.
    """

    name = "klypin11"
    m_min = 3e10
    m_max = 5e14
    z_min = 0.0
    z_max = 0.0

    def __call__(
        self,
        M: ArrayLike,
    ) -> ArrayLike:  # could I need cosmo, no
        # valid:bool = (
        # (M >= self.m_min) &
        # (M <= self.m_max) &
        # (z >= self.z_min) &
        # (z <= self.z_max)
        # )
        return 9.6 * (M / 1e12) ** -0.075


@dataclass
class child18all:
    """
    Child et al. (2018) c-M relation for all halos using
    :math:`M_{200c}`.

    https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology for :math:`\\sigma(R)` and the growth factor.
    M : ArrayLike
        Halo mass :math:`M_{200c}` [h-1 Msun].
    z : ArrayLike
        Redshift.

    Returns
    -------
    c : ArrayLike
        Concentration :math:`c_{200c}`.

    Notes
    -----
    - Defines a characteristic mass :math:`M_*` through

      .. math::

          \\sigma(M_*, z) = \\frac{\\delta_{\\rm sc}}{D(z)}

      Concentration depends on the ratio :math:`M / M_*`.
    - Calibrated on WMAP7.
    - Valid for
      :math:`M > 2.1 \\times 10^{11}\\, h^{-1} M_\\odot`,
      :math:`0 < z < 4`.
    """

    same: str = "child18all"
    m_min: float = 2.1e11
    m_max: float = jnp.inf
    z_min: float = 0
    z_max: float = 4

    def __init__(self, cosmo: jc.Cosmology):
        self.cosmo = cosmo
        logR = jnp.linspace(-3, 5, 2048)  # 1e-2 to 1e2 Mpc/h
        self.R_grid = 10**logR
        self.sigma_grid = lss.sigma_R(self.R_grid, z=0, cosmo=cosmo)

    def __call__(
        self,
        M: ArrayLike,
        z: ArrayLike,
    ) -> ArrayLike:
        # valid: bool etc.
        deltath = 1.68647

        def R_of_sigma(sigma_val, sigma_grid, R_grid):
            return jnp.interp(
                jnp.log10(sigma_val),
                jnp.log10(sigma_grid[::-1]),
                jnp.log10(R_grid[::-1]),
            )

        a = jnp.atleast_1d(1 / (1 + z))
        Dz = jc.background.growth_factor(self.cosmo, a)
        sigma_target = deltath / Dz
        Rstr = 10 ** R_of_sigma(sigma_target, self.sigma_grid, self.R_grid)
        rho_m0 = cosmology.critical_density(
            0.0, self.cosmo
        ) * jc.background.Omega_m_a(self.cosmo, 1.0)
        Mstr: ArrayLike = 4 * jnp.pi / 3 * rho_m0 * Rstr**3

        mex: float = -0.10
        A: float = 3.44
        b: float = 430.49
        c0: float = 3.19
        x = M / Mstr / b
        return c0 + A * (x**mex * (1 + x) ** -mex - 1)


@dataclass
class child18relaxed:
    """
    Child et al. (2018) c-M relation for relaxed halos using
    :math:`M_{200c}`.

    https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology for :math:`\\sigma(R)` and the growth factor.
    M : ArrayLike
        Halo mass :math:`M_{200c}` [h-1 Msun].
    z : ArrayLike
        Redshift.

    Returns
    -------
    c : ArrayLike
        Concentration :math:`c_{200c}`.

    Notes
    -----
    - Same functional form as :class:`child18all` but calibrated
      for relaxed halo populations.
    - Calibrated on WMAP7.
    - Valid for
      :math:`M > 2.1 \\times 10^{11}\\, h^{-1} M_\\odot`,
      :math:`0 < z < 4`.
    """

    same: str = "child18relaxed"
    m_min: float = 2.1e11
    m_max: float = jnp.inf
    z_min: float = 0
    z_max: float = 4

    def __init__(self, cosmo: jc.Cosmology):
        self.cosmo = cosmo
        logR = jnp.linspace(-3, 5, 2048)  # 1e-2 to 1e2 Mpc/h
        self.R_grid = 10**logR
        self.sigma_grid = lss.sigma_R(self.R_grid, z=0, cosmo=cosmo)

    def __call__(
        self,
        M: ArrayLike,
        z: ArrayLike,
    ) -> ArrayLike:
        # valid: bool etc.
        deltath = 1.68647

        def R_of_sigma(sigma_val, sigma_grid, R_grid):
            return jnp.interp(
                jnp.log10(sigma_val),
                jnp.log10(sigma_grid[::-1]),
                jnp.log10(R_grid[::-1]),
            )

        a = jnp.atleast_1d(1 / (1 + z))
        Dz = jc.background.growth_factor(self.cosmo, a)
        sigma_target = deltath / Dz
        Rstr = 10 ** R_of_sigma(sigma_target, self.sigma_grid, self.R_grid)
        rho_m0 = cosmology.critical_density(
            0.0, self.cosmo
        ) * jc.background.Omega_m_a(self.cosmo, 1.0)
        Mstr: ArrayLike = 4 * jnp.pi / 3 * rho_m0 * Rstr**3

        mex: float = -0.09  # need to adjust
        A: float = 2.88
        b: float = 1644.53
        c0: float = 3.54
        return c0 + A * (
            (M / Mstr / b) ** mex * (1 + (M / Mstr / b)) ** -mex - 1
        )

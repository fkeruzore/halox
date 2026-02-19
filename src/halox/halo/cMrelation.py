from dataclasses import dataclass
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from halox import lss, cosmology

jax.config.update("jax_enable_x64", True)

# currently implementing all of these for 200c


@dataclass(frozen=True)
class duffy08:
    """
    duffy08 mass concentration relation using M200c.

    (https://ui.adsabs.harvard.edu/abs/2008MNRAS.390L..64D/abstract)

    Calibrated cosmologies
    ----------------------
    WMAP5

    Parameters
    ----------

    z, ArrayLike: Redshift
    M, ArrayLike: M200c (Solar Masses)

    Returns
    -------
    c, ArrayLike: Concentration
    """

    name: str = "duffy08"  # only the 200c parameters
    m_min: float = 1e11
    m_max: float = 1e15
    z_min: float = 0.0
    z_max: float = 2.0

    A: float = 5.71
    B: float = -0.084
    C: float = -0.47

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
    prada12 mass-concentration relation using M200c.

    (http://adsabs.harvard.edu/abs/2012MNRAS.423.3018P)

    Calibrated cosmologies
    ----------------------
    Any (cosmology-dependent through sigma(M, z))

    Valid range
    -----------
    M : Any
    z : Any

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology used to compute sigma(M, z)

    M : ArrayLike
        M200c (h^-1 Solar Masses)

    z : ArrayLike
        Redshift

    Returns
    -------
    c : ArrayLike
        Concentration (c200c)

    Notes
    -----
    This model predicts concentration as a function of peak height
    through sigma(M, z), and captures an upturn in concentration at
    high masses.
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


@dataclass(frozen=True)
class klypin11:
    """
    klypin11 mass-concentration relation using Mvir at z = 0.

    (http://adsabs.harvard.edu/abs/2011ApJ...740..102K)

    Calibrated cosmologies
    ----------------------
    WMAP7

    Valid range
    -----------
    3e10 < M < 5e14  (h^-1 Solar Masses)
    z = 0

    Parameters
    ----------
    M : ArrayLike
        Mvir (h^-1 Solar Masses)

    Returns
    -------
    c : ArrayLike
        Concentration (cvir)

    Notes
    -----
    The original paper provides redshift evolution, but this
    implementation corresponds only to the z = 0 case.
    """

    name: str = "klypin11"
    m_min: float = 3e10
    m_max: float = 5e14
    z_min: float = 0.0
    z_max: float = 0.0

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
    child18 mass–concentration relation for all halos using M200c.

    (https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract)

    Compatible cosmologies
    ----------------------
    WMAP7 (calibrated)

    Valid range
    -----------
    M > 2.1e11  (h^-1 Solar Masses)
    0 < z < 4

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology used to compute σ(R) and growth factor

    M : ArrayLike
        M200c (h^-1 Solar Masses)

    z : ArrayLike
        Redshift

    Returns
    -------
    c : ArrayLike
        Concentration (c200c)

    Notes
    -----
    This model introduces a characteristic mass scale M* defined
    via sigma(M*, z) = delta_sc / D(z). The concentration depends on
    the ratio M / M*.
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
    child18 mass–concentration relation for relaxed halos using M200c.

    (https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract)

    Compatible cosmologies
    ----------------------
    WMAP7 (calibrated)

    Valid range
    -----------
    M > 2.1e11  (h^-1 Solar Masses)
    0 < z < 4

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology used to compute σ(R) and growth factor

    M : ArrayLike
        M200c (h^-1 Solar Masses)

    z : ArrayLike
        Redshift

    Returns
    -------
    c : ArrayLike
        Concentration (c200c)

    Notes
    -----
    Same functional form as child18all but calibrated
    specifically for relaxed halo populations.
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

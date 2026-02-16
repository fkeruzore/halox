from dataclasses import dataclass
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from halox import lss, cosmology

jax.config.update("jax_enable_x64", True)

# TODO
# Add in validity flag, need to do this array wise, no prints, just as a data product

# currently implementing all of these for 200c
# not sure how we would implement multiple mass defs besidse brute force (so no if statements, need explicit calls)
# CONCENTRATION_MODELS = {
#     "duffy08": duffy08(),
#     "prada12": prada12(),
#     "klypin11": klypin11(),
#     "child18all": child18all(),
#     "child18relaxed": child18relaxed()
# }

@dataclass(frozen=True)
class duffy08: #need to verify this is not a halucination
    """
    duffy08 mass concentration relation. 
    (https://ui.adsabs.harvard.edu/abs/2008MNRAS.390L..64D/abstract)
    
    z, ArrayLike: Redshift
    M, ArrayLike: M200c
    """
    name:str = "duffy08" # the following are the 200c parameters, there are others
    m_min: float = 1e11
    m_max: float = 1e15
    z_min: float = 0.0
    z_max: float = 2.0

    A: float = 5.71
    B: float = -0.084
    C: float = -0.47

    def __call__(self, 
                 M: ArrayLike,
                 z: ArrayLike): #could I need cosmo?
        # valid:bool = (
        # (M >= self.m_min) &
        # (M <= self.m_max) &
        # (z >= self.z_min) &
        # (z <= self.z_max)
        # )
        M0 = 2e12 
        return self.A * (M / M0) ** self.B * (1 + z) ** self.C #, valid

@dataclass(frozen=True)
class prada12: 
    name:str = "prada12"
    m_min: float = 0
    m_max: float = jnp.inf
    z_min: float = 0
    z_max: float = jnp.inf

    def __call__(self, 
                 M: ArrayLike,
                 z: ArrayLike, 
                 cosmo: jc.Cosmology, ):
        # valid:bool = (
        # (M >= self.m_min) &
        # (M <= self.m_max) &
        # (z >= self.z_min) &
        # (z <= self.z_max)
        # )
        def cmin(x):
            return 3.681 + (5.033 - 3.681) * (1.0 / jnp.pi * jnp.arctan(6.948 * (x - 0.424)) + 0.5)
        def smin(x):
            return 1.047 + (1.646 - 1.047) * (1.0 / jnp.pi * jnp.arctan(7.386 * (x - 0.526)) + 0.5)

        a = (1+z)**-1
        x = (cosmo.Omega_de / cosmo.Omega_m) ** (1.0 / 3.0) * a

        B0 = cmin(x) / cmin(1.393)
        B1 = smin(x) / smin(1.393)

        temp_sig = lss.sigma_M(M,z,cosmo, k_max = 1e3, n_k_int=20000) #slight differ from colossus here
        temp_sigp = temp_sig * B1
        temp_C = 2.881 * ((temp_sigp / 1.257) ** 1.022 + 1) * jnp.exp(0.060 / temp_sigp ** 2)
        c = B0 * temp_C

        return c #, valid
    
@dataclass(frozen=True)
class klypin11: #need to verify this is not a halucination
    """
    Docstring for klypin11
    
    Klypin et al. 2011 gives values for other redshifts, but no clear redshift dependence is stated.
    Here, we only provide the function at z = 0.

    :var M: Description
    :vartype M: Array
    """
    name:str = "klypin11" # the following are not the only possible parameters, maybe need othe ones too?
    m_min: float = 3e10
    m_max: float = 5e14
    z_min: float = 0.0
    z_max: float = 0.0

    def __call__(self, 
                 M: ArrayLike,): #could I need cosmo, no
        # valid:bool = (
        # (M >= self.m_min) &
        # (M <= self.m_max) &
        # (z >= self.z_min) &
        # (z <= self.z_max)
        # )
        return 9.6 * (M / 1E12)**-0.075

@dataclass(frozen=True) 
class child18all: #maybe we need 4 cases of this???
    """
    child18all
    M = M200c
    For all halos
    M_T is the transition mass
    """
    same:str = "child18all"
    m_min: float = 2.1E11
    m_max: float = jnp.inf
    z_min: float = 0
    z_max: float = 4
    
    def __call__(self,
                 M: ArrayLike, 
                 z: ArrayLike,
                 cosmo:jc.Cosmology, ):
        #valid: bool etc.
        deltath = 1.68647
        logR = jnp.linspace(-3, 5, 2048)   # 1e-2 to 1e2 Mpc/h
        R_grid = 10**logR
        sigma_grid = lss.sigma_R(R_grid, z=0, cosmo = cosmo)
        def R_of_sigma(sigma_val, sigma_grid, R_grid):
            return jnp.interp(
                jnp.log10(sigma_val),
                jnp.log10(sigma_grid[::-1]),
                jnp.log10(R_grid[::-1])
            )
        a = jnp.atleast_1d(1/(1+z))
        Dz = jc.background.growth_factor(cosmo, a)
        sigma_target = deltath / Dz
        Rstr = 10**R_of_sigma(sigma_target, sigma_grid, R_grid)
        rho_m0 = (
            cosmology.critical_density(0.0, cosmo)
            * jc.background.Omega_m_a(cosmo, 1.0)
            )
        Mstr:ArrayLike = 4*jnp.pi/3 * rho_m0 * Rstr**3 

        mex:float = -0.10
        A:float = 3.44
        b:float = 430.49
        c0:float = 3.19
        x = M/Mstr / b
        return c0 + A*(x**mex * (1 + x)**-mex - 1)

@dataclass(frozen=True)
class child18relaxed: #maybe we need 4 cases of this???
    """
    child18all
    M = M200c
    Only apply to relaxed halo populations
    """
    same:str = "child18relaxed"
    m_min: float = 2.1E11
    m_max: float = jnp.inf
    z_min: float = 0
    z_max: float = 4

    def __call__(self,
                 M: ArrayLike, 
                 z: ArrayLike,
                 cosmo:jc.Cosmology, ):
        #valid: bool etc.
        deltath = 1.68647
        logR = jnp.linspace(-3, 5, 2048)   # 1e-2 to 1e2 Mpc/h
        R_grid = 10**logR
        sigma_grid = lss.sigma_R(R_grid, z=0, cosmo = cosmo)
        def R_of_sigma(sigma_val, sigma_grid, R_grid):
            return jnp.interp(
                jnp.log10(sigma_val),
                jnp.log10(sigma_grid[::-1]),
                jnp.log10(R_grid[::-1])
            )
        a = jnp.atleast_1d(1/(1+z))
        Dz = jc.background.growth_factor(cosmo, a)
        sigma_target = deltath / Dz
        Rstr = 10**R_of_sigma(sigma_target, sigma_grid, R_grid)
        rho_m0 = (
            cosmology.critical_density(0.0, cosmo)
            * jc.background.Omega_m_a(cosmo, 1.0)
            )
        Mstr:ArrayLike = 4*jnp.pi/3 * rho_m0 * Rstr**3 

        mex:float = -0.09#need to adjust
        A:float = 2.88
        b:float = 1644.53
        c0:float = 3.54
        return c0 + A*((M/Mstr / b)**mex * (1 + (M/Mstr / b))**-mex - 1)
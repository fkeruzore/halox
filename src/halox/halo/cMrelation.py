from dataclasses import dataclass
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt as jop
import jax_cosmo as jc
from halox import lss, cosmology

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
        
        
        nu = lss.peakheight(M, z, cosmo)

        a = 1 / (1+z)
        x = (cosmo.Omega_de / cosmo.Omega_m) ** (1.0 / 3.0) * a
        B0 = cmin(x) / cmin(1.393)
        B1 = smin(x) / smin(1.393)
        temp_sig = 1.686 / nu # Replace with lss.sigma_M. This bakes in the 1.686 tophat density requirement, but intentionally?
        temp_sigp = temp_sig * B1
        temp_C = 2.881 * ((temp_sigp / 1.257) ** 1.022 + 1) * jnp.exp(0.06 / temp_sigp ** 2)
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
    
# For array like z??? need to look into this more
# Rstr = jax.vmap(lambda zi: jnp.exp(
#     jop.Bisection(
#         lambda lr: lss.sigma_R(jnp.exp(lr), zi, cosmo) - deltath,
#         lower=logRmin,
#         upper=logRmax,
#     ).run().params
# ))(z)

@dataclass(frozen=True) #something is wrong with my implementation
#solver slows this down, can I get rid of it?
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
        deltath = 1.686
        def F(logR): #maybe make this external if implementing all four child18 c-Ms
            """
            Zero function to optimize to to find the right R
            """
            return lss.sigma_R(jnp.exp(logR), z, cosmo) - deltath
        
        logRmin = float(jnp.log(1E-2))
        logRmax = float(jnp.log(50))

        solver = jop.Bisection(F, lower = logRmin, upper = logRmax, maxiter=50)
        Rstr = jnp.exp(solver.run().params)
        Mstr:ArrayLike = 4*jnp.pi/3 * cosmology.critical_density(z, cosmo) * cosmo.Omega_m(z) * Rstr**3 

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
        deltath = 1.686
        def F(logR): #maybe make this external if implementing all four child18 c-Ms
            """
            Zero function to optimize to to find the right R
            """
            return lss.sigma_R(jnp.exp(logR), z, cosmo) - deltath
        
        logRmin = float(jnp.log(1E-2))
        logRmax = float(jnp.log(50))

        solver = jop.Bisection(F, lower = logRmin, upper = logRmax, maxiter=50)
        Rstr = jnp.exp(solver.run().params)
        Mstr:ArrayLike = 4*jnp.pi/3 * cosmology.critical_density(z, cosmo) * cosmo.Omega_m(z) * Rstr**3

        mex:float = -0.09#need to adjust
        A:float = 2.88
        b:float = 1644.53
        c0:float = 3.54
        return c0 + A*((M/Mstr / b)**mex * (1 + (M/Mstr / b))**-mex - 1)
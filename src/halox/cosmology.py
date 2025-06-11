from functools import partial
import jax_cosmo as jc

# Planck 2018 cosmology parameters
Planck18 = partial(
    jc.Cosmology,
    Omega_c=0.2607,
    Omega_b=0.04897,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,
)

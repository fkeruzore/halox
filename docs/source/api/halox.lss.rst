halox.lss: Large-scale structure calculations
===============================================

``halox`` provides a JAX implementation of large-scale structure calculations including RMS variance computations, mass-to-radius conversions, and overdensity transformations.
Cosmology calculations (e.g. power spectra) rely on `jax-cosmo <https://github.com/DifferentiableUniverseInitiative/jax_cosmo>`_.

.. currentmodule:: halox.lss

.. autosummary::
    sigma_R
    sigma_M
    overdensity_c_to_m
    mass_to_lagrangian_radius

.. autofunction:: sigma_R
.. autofunction:: sigma_M
.. autofunction:: overdensity_c_to_m
.. autofunction:: mass_to_lagrangian_radius
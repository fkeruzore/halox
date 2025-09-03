halox.hmf: Halo mass functions calculations
===============================================

``halox`` provides a JAX implementation of the `Tinker08 <https://ui.adsabs.harvard.edu/abs/2008ApJ...688..709T/abstract>`_ halo mass function.
Cosmology calculations (e.g. power spectra) rely on `jax-cosmo <https://github.com/DifferentiableUniverseInitiative/jax_cosmo>`_.

.. currentmodule:: halox.hmf

.. autosummary::
    tinker08_mass_function
    tinker08_f_sigma

.. autofunction:: tinker08_mass_function
.. autofunction:: tinker08_f_sigma
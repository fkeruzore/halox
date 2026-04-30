halox.bias: Halo bias
=====================

``halox`` provides a JAX implementation of the `Tinker10 <https://ui.adsabs.harvard.edu/abs/2010ApJ...724..878T/abstract>`_ halo mass function.
Cosmology calculations (e.g. power spectra) rely on `jax-cosmo <https://github.com/DifferentiableUniverseInitiative/jax_cosmo>`_.
For examples, see :doc:`../notebooks/bias`.

.. currentmodule:: halox.bias

.. autosummary::
    tinker10_bias

.. autofunction:: tinker10_bias
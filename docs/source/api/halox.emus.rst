halox.emus: Emulators
=====================

``halox`` provides JAX-powered neural network emulators to speed up expensive theory computations.
In particular, it packs an emulator of the RMS of density fluctuations $\\sigma$ as a function of mass, redshift, and cosmology, which can be used as a backend to compute halo bias and halo mass function.

.. currentmodule:: halox.emus

.. autosummary::
    SigmaMEmulator

.. autoclass:: SigmaMEmulator
    :member-order: bysource
    :members: build_input, __call__

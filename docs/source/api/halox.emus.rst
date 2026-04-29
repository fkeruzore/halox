halox.emus: Emulators
=====================

``halox`` provides JAX-powered neural network emulators to speed up expensive theory computations.
In particular, it packs an emulator of the RMS of density fluctuations $\\sigma$ as a function of mass, redshift, and cosmology, which can be used as a backend to compute halo bias and halo mass function.

.. list-table:: Available emulators
   :widths: 18 12 12 12 10 10 12 12 10 12 12
   :header-rows: 1

   * - File Name
     - Network
     - Epochs
     - :math:`M`
     - :math:`z`
     - :math:`\Omega_b`
     - :math:`\Omega_m`
     - :math:`h`
     - :math:`\sigma_8`
     - :math:`n_s`

   * - sigma_mp4.npz (default)
     - MLP; 3 hidden layers (size 64); SiLU inner activations
     - 25,000
     - [11, 16]
     - [-0.04, 5]
     - [0.01, 0.08]
     - [0.085, 0.5]
     - [0.6, 1.0]
     - [0.4, 1.0]
     - [0.8, 1.1]



.. currentmodule:: halox.emus

.. autosummary::
    SigmaMEmulator

.. autoclass:: SigmaMEmulator
    :member-order: bysource
    :members: build_input, __call__

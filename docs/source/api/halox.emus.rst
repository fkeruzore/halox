halox.emus: Emulators
=====================

``halox`` provides JAX-powered neural network emulators to speed up expensive theory computations.
In particular, it packs an emulator of the RMS of density fluctuations $\\sigma$ as a function of mass, redshift, and cosmology, which can be used as a backend to compute halo bias and halo mass function.

.. list-table:: Neural Network Details
   :widths: 25 25 25 25
   :header-rows: 1

   * - File Name
     - Network (width x depth)
     - Activation Function
     - Training Epochs
   * - sigma_mp4.npz (default)
     - 64 x 5
     - SiLU
     - 25,000

.. list-table:: sigma_mp4 Training Bounds
   :widths: 20 10 10 15 15 10 15 15
   :header-rows: 1

   * - 
     - Mass (log Solar)
     - Redshift
     - Baryon density
     - Matter density
     - Expansion rate
     - Density fluctuation amplitude
     - Scalar spectral index

   * - Symbol
     - :math:`M`
     - :math:`z`
     - :math:`\Omega_b`
     - :math:`\Omega_m`
     - :math:`h`
     - :math:`\sigma_8`
     - :math:`n_s`

   * - Bounds
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

.. halox documentation master file, created by
   sphinx-quickstart on Fri Jun 13 14:43:28 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*****
halox
*****

JAX-powered Python library for differentiable dark matter halo property and mass function calculations.

``halox`` provides differentiable and GPU-accelerated implementations of widely used dark matter halo and large-scale structure calculations.
All functions support ``jax.jit``, ``jax.vmap``, and ``jax.grad``; halo masses, redshifts, and cosmological parameters are all valid differentiation targets, enabling efficient gradient-based workflows such as Hamiltonian Monte Carlo sampling or machine learning pipelines.

Features
^^^^^^^^

* :doc:`notebooks/nfw` and :doc:`notebooks/einasto`: Radial profiles of dark matter halos (density, enclosed mass, gravitational potential, circular velocity, velocity dispersion, projected surface density)
* :doc:`notebooks/cMrelations`: Concentration–mass relations
* :doc:`notebooks/lss`: Large-scale structure (σ(R), σ(M))
* :doc:`notebooks/hmf`: Halo mass function (Tinker et al. 2008)
* :doc:`notebooks/bias`: Halo bias (Tinker et al. 2010)
* :doc:`notebooks/using_the_emulator`: Neural-network emulator for σ(M), enabling up to 95× speedup over the analytic calculation

All quantities support cosmology dependence via `jax-cosmo <https://github.com/DifferentiableUniverseInitiative/jax_cosmo>`_.

Installation
^^^^^^^^^^^^

``halox`` can be installed via ``pip``:

.. code-block:: bash

   pip install halox

See :doc:`install` for source installation instructions.

Quick start
^^^^^^^^^^^

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from halox import cosmology, hmf, halo

   jax.config.update("jax_enable_x64", True)

   cosmo = cosmology.Planck18()

   # Halo mass function at z = 0.5
   M = jnp.logspace(13, 15, 100)  # h⁻¹ M☉
   dn_dlnM = hmf.tinker08_mass_function(M, 0.5, cosmo)

   # NFW profile
   h = halo.nfw.NFWHalo(1e14, 5.0, 0.0, cosmo)
   r = jnp.logspace(-1, 1, 50)  # h⁻¹ Mpc
   rho = h.density(r)

   # Derivatives w.r.t. halo mass
   dhmf_dM = jax.grad(
       lambda M: hmf.tinker08_mass_function(M, 0.5, cosmo)
   )(1e14)

Units
^^^^^

All input/output units for ``halox`` functions are reported in docstrings.
Units are assumed to be in proper coordinates (not comoving) and include factors of `little h <https://arxiv.org/abs/1308.4150)>`_.


Learn more
^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   install
   Automatic differentiation and JAX features <notebooks/gradients.ipynb>
   The σ(M) emulator <notebooks/using_the_emulator.ipynb>
   Comparison with colossus <notebooks/halox_vs_colossus.ipynb>
   Other utilities <notebooks/others.ipynb>


.. toctree::
   :maxdepth: 1
   :caption: Physics modules gallery

   notebooks/nfw.ipynb
   notebooks/einasto.ipynb
   notebooks/cMrelations.ipynb
   notebooks/lss.ipynb
   notebooks/hmf.ipynb
   notebooks/bias.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/halox.cosmology
   api/halox.nfw
   api/halox.einasto
   api/halox.cm
   api/halox.lss
   api/halox.emus
   api/halox.hmf
   api/halox.bias

Citation
^^^^^^^^

If you use ``halox`` for your research, please cite the `original paper <https://arxiv.org/abs/2509.22478>`_:

.. code-block:: bibtex

   @ARTICLE{2025arXiv250922478K,
          author = {{K{\'e}ruzor{\'e}}, Florian},
           title = "{halox: Dark matter halo properties and large-scale structure calculations using JAX}",
         journal = {arXiv e-prints},
        keywords = {Instrumentation and Methods for Astrophysics, Cosmology and Nongalactic Astrophysics},
            year = 2025,
           month = sep,
             eid = {arXiv:2509.22478},
           pages = {arXiv:2509.22478},
             doi = {10.48550/arXiv.2509.22478},
   archivePrefix = {arXiv},
          eprint = {2509.22478},
    primaryClass = {astro-ph.IM},
   }

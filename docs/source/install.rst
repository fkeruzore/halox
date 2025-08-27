Installation
============

Standard installation
^^^^^^^^^^^^^^^^^^^^^

``halox`` can be install via ``pip``:

.. code-block:: bash

   pip install halox

From source
^^^^^^^^^^^

Alternatively, ``halox`` can be installed from its `source repository <https://github.com/fkeruzore/halox>`_:

.. code-block:: bash

   git clone git@github.com:fkeruzore/halox.git
   cd halox
   pip install .

Dependencies
^^^^^^^^^^^^

``halox`` requires `JAX <https://docs.jax.dev/en/latest/>`_ for all computations and `jax-cosmo <https://github.com/DifferentiableUniverseInitiative/jax_cosmo>`_ for cosmology-dependent computations.
Dependencies are managed using `uv <https://docs.astral.sh/uv/>`_.

Running tests
^^^^^^^^^^^^^

When installing ``halox`` from source using ``uv``, you can install test dependencies:

.. code-block:: bash

   git clone git@github.com:fkeruzore/halox.git
   cd halox
   uv sync --extra tests
   uv pip install .

Then, you can run the full suite of unit tests:

.. code-block:: bash

   uv run pytest

The test suite involves validation of physics modules against astropy/colossus, and a coverage check.
Installation
============

Standard installation
^^^^^^^^^^^^^^^^^^^^^

``halox`` can be installed via ``pip``:

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

When installing ``halox`` from source using ``uv``, you can install optional dependency groups:

.. code-block:: bash

   git clone git@github.com:fkeruzore/halox.git
   cd halox
   uv sync --extra tests   # test dependencies (pytest, astropy, colossus, gala)
   uv sync --extra docs    # documentation dependencies (sphinx, myst-nb, ...)
   uv sync --all-extras    # all optional dependencies
   uv pip install .

Running tests
^^^^^^^^^^^^^

After installing with ``--extra tests``, run the full suite of unit tests:

.. code-block:: bash

   uv run pytest

The test suite validates physics modules against astropy/colossus/gala, and enforces a 100% coverage.

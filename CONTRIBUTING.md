# Contributing to halox

Thanks for your interest in contributing! This guide covers what you need to develop and add features to halox.

## Development Setup

halox uses [uv](https://docs.astral.sh/uv/) for dependency management. If you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone, install all extras, and enable the pre-commit hooks:
```bash
git clone https://github.com/fkeruzore/halox.git
cd halox
uv sync --all-extras
uv run pre-commit install
```

Verify your environment:
```bash
uv run pytest
```

## Tests and Linting

### Tests
- `uv run pytest` runs the full suite and enforces 100% coverage.
- Target a module with `uv run pytest tests/test_hmfs.py`, or filter with `-k <keyword>`.
- Place new tests under `tests/` with descriptive module names.
- Prefer comparisons against trusted references (astropy, colossus) when validating scientific results.
- Avoid modifying or deleting existing tests.
- Please don't have AI agents implement unit tests for new features, they are a human-maintained safety net.

### Linting & Formatting
[ruff](https://docs.astral.sh/ruff/) handles linting and formatting; config lives in `pyproject.toml` (79-character lines, `E`/`F`/`B` rule sets):
- `uv run ruff check` (lint)
- `uv run ruff format` (format)

### Pre-commit Hooks
The repo ships a [pre-commit](https://pre-commit.com) config that runs ruff plus basic file-hygiene checks (trailing whitespace, end-of-file, YAML/TOML syntax, large-file guard) on staged files. Install once with `uv run pre-commit install` (see Setup above). If a hook rewrites a file, the commit aborts so you can review and re-stage. To run all hooks against the whole repo before opening a PR:
```bash
uv run pre-commit run --all-files
```

## Implementing New Features

### Module Layout
- **`cosmology`** — utilities extending jax-cosmo (Hubble parameter, critical density, …)
- **`halo`** — halo profiles (`halo.nfw`, `halo.einasto`)
- **`lss`** — large-scale structure (RMS variance, …)
- **`hmf`** — halo mass functions (Tinker08)
- **`bias`** — halo bias (Tinker10)
- **`cm`** — concentration–mass relations
- **`emus`** — neural-network emulators for expensive quantities

Add features to the appropriate module, or create a new submodule if none fits.

### JAX Conventions
- Use `jax.numpy` with `import jax.numpy as jnp`, not `numpy`.
- Support vectorization via `jax.vmap` where applicable.
- Enable 64-bit precision in examples: `jax.config.update("jax_enable_x64", True)`.

### Cosmology Dependencies
Always thread a `jax_cosmo.Cosmology` object through cosmology-dependent calculations:
```python
import jax_cosmo as jc
import halox.cosmology as hc

def compute_something(z: ArrayLike, cosmo: jc.Cosmology) -> Array:
    Om = cosmo.Omega_m
    rho_c = hc.critical_density(z, cosmo)
    ...
```
Extend `halox.cosmology` when jax-cosmo lacks something you need.

### Type Hints and Docstrings
Use NumPy-style docstrings with units, and annotate inputs with `ArrayLike`, returns with `Array`:
```python
from jax import Array
from jax.typing import ArrayLike
import jax_cosmo as jc

def my_function(M: ArrayLike, z: ArrayLike, cosmo: jc.Cosmology) -> Array:
    """Short description.

    Parameters
    ----------
    M : ArrayLike
        Halo mass [h-1 Msun]
    z : ArrayLike
        Redshift
    cosmo : jc.Cosmology
        Underlying cosmology

    Returns
    -------
    Array
        Result [h2 Msun Mpc-3]
    """
```
Conventions:
- `cosmo` is documented as "Underlying cosmology".
- Units in square brackets, with `h-1`, `h2`, etc. for little-h factors.
- Reference original papers when relevant.

## Documentation

### Example Notebooks
New functionality should come with an example notebook in `notebooks/` showing typical usage and JAX benefits. If a colossus equivalent exists, add a comparison to `notebooks/halox_vs_colossus.ipynb`. Reuse the import/style boilerplate from existing notebooks rather than redefining it.

### Building the Docs
```bash
uv run sphinx-build -b html docs/source docs/build
```
Output lands in `docs/build/html/`.

## Pull Requests

Before opening a PR, confirm:

- [ ] `uv run ruff format` and `uv run ruff check` are clean (pre-commit handles this if installed).
- [ ] `uv run pytest` passes with coverage at 100%.
- [ ] Public functions have complete docstrings.
- [ ] Documentation and notebooks are updated where relevant.

Work on a feature branch, then open a PR describing the change, motivation, testing performed, and any breaking changes. CI runs the test suite (multiple Python versions), ruff, and coverage; all must pass before merge.

## Getting Help
- **Issues** — bug reports and feature requests
- **Discussions** — questions and design conversations

Thanks for contributing!

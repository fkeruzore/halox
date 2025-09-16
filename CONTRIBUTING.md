# Contributing to halox

Thank you for your interest in contributing to halox! This guide will help you get started with developing and adding new features to the library.

## Development Setup

### Step 0: Install uv
This project uses [uv](https://docs.astral.sh/uv/) for dependency management. If you don't have uv installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 1: Clone and Install Dependencies
```bash
git clone https://github.com/fkeruzore/halox.git
cd halox
uv sync --all-extras  # Install all dependencies including tests and docs
uv pip install .
```

### Step 2: Verify Installation
Run tests to confirm your environment is ready before you start coding:
```bash
uv run pytest
```
(For more information, see [Tests and linting / Running tests](#running-tests))

## Tests and linting

### Running Tests
- `uv run pytest` executes the full suite and enforces 100% coverage.
- Target modules with `uv run pytest tests/test_hmfs.py` or add `-k <keyword>` for finer selection.
- All tests must pass before you open a pull request.

### Unit Test Conventions
- Place tests under `tests/` with descriptive module names.
- Prefer comparisons against trusted references (e.g., astropy, colossus) when validating scientific results.
- Never modify or delete existing tests.

### Linting & Formatting
We recommend using [ruff](https://docs.astral.sh/uv/) for linting and formatting.
The configuration in `pyproject.toml` will allow you to run:
- `uv run ruff check` to run the linter with the `E` (pycodestyles), `F` (Pyflakes), and `B` (flake8-bugbear) rulesets;
- `uv run ruff format` to apply code formatting (PEP 8 with 79 character lines).

## Implementing New Features

### Code Structure
The halox library is organized into five main modules:

- **`cosmology`**: Cosmology utility functions (Hubble parameter, critical density) extending jax-cosmo
- **`halo`**: Halo profile implementations (currently only NFW profiles in `halo.nfw`)
- **`lss`**: Large-scale structure properties beyond jax-cosmo (e.g. RMS variance)
- **`hmf`**: Halo mass function calculations (currently only Tinker et al. 2008)
- **`bias`**: Halo bias calculations (currently only Tinker et al. 2010)

Make sure your features are added to the right module, or to a new submodule if none of them fit.

### JAX Best Practices
- Support vectorization using `jax.vmap` where applicable
- Leverage JAX's autodifferentiation capabilities
- Use `jax.numpy` instead of `numpy`
- Enable 64-bit precision in examples: `jax.config.update("jax_enable_x64", True)`

### Cosmology Dependencies
Always use `jax_cosmo.Cosmology` objects for cosmological calculations:
```python
import jax_cosmo as jc
import halox.cosmology as hc

def compute_something(z: ArrayLike, cosmo: jc.Cosmology) -> Array:
    """Compute something cosmology-dependent."""
    # Use cosmo.h, cosmo.Omega_m, etc.
    Om = cosmo.Omega_m
    rho_c = hc.critical_density(z, cosmo)
    rho_m = Om * rho_c
    # Your implementation
```
Feel free to extend functionality through the `halox.cosmology` module when jax-cosmo doesn't provide needed functions.

### Docstring Format
Use NumPy-style docstrings with clear units:
```python
def some_function(M: ArrayLike, z: ArrayLike, cosmo: jc.Cosmology) -> Array:
    """[description]
    
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
        Some result [h2 Msun Mpc-3]
    """
```
Key conventions:
- Document the `cosmo` parameter as "Underlying cosmology"
- Specify units in square brackets
- Use `h-1` notation for little h factors (e.g., `[h-1 Msun]`, `[h2 Msun Mpc-3]`)
- Include mathematical context and references to original papers

### Type Hints
Use proper type hints throughout your code:
```python
from jax import Array
from jax.typing import ArrayLike
import jax_cosmo as jc

def my_function(M: ArrayLike, z: ArrayLike, cosmo: jc.Cosmology) -> Array:
    """Function description."""
    # Implementation
```
- Use `ArrayLike` for input parameters that accept arrays or scalars
- Use `Array` for return values and internal JAX arrays
- Use `jax_cosmo.Cosmology` for cosmology objects

## Writing Documentation

### API Documentation
All public functions require comprehensive docstrings as shown above. Include:

- Clear description of functionality
- Parameter descriptions with units
- Return value descriptions
- References to original papers when relevant

### Example Notebooks
When adding new functionality, provide an example notebook in `notebooks/` that covers typical usage and highlights JAX-based benefits.
If relevant, also add a comparison against a trusted reference like colossus to the `notebooks/halox_vs_colossus.ipynb` notebook.
New notebooks should reuse the import/style pattern from existing notebooks instead of redefining it.

### Building Documentation
Build the Sphinx documentation locally:

```bash
uv run sphinx-build -b html docs/source docs/build
```

Documentation will be built in `docs/build/html/`.

## Pull Requests and CI/CD

### Before Submitting
Ensure your contribution meets these requirements:

- [ ] Code is properly formatted (`uv run ruff format`)
- [ ] No linting errors (`uv run ruff check`)
- [ ] New features include comprehensive tests (test coverage should always remain at 100%)
- [ ] All tests pass (`uv run pytest`)
- [ ] Public functions have complete docstrings
- [ ] Documentation is updated if needed

### Pull Request Process
1. **Create a feature branch**:
   ```bash
   git checkout -b feature/descriptive-name
   ```
2. **Make your changes** following the guidelines above
3. **Commit with clear messages**:
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```
4. **Push and create PR**:
   ```bash
   git push origin feature/descriptive-name
   ```
5. **Submit pull request** on GitHub with:
   - Clear description of changes
   - Motivation for the feature/fix
   - Testing performed
   - Any breaking changes

### CI/CD Pipeline
Our CI system automatically:

- Runs the full test suite on multiple Python versions
- Checks code formatting and linting
- Measures test coverage

All checks must pass before a PR can be merged.

## Getting Help
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Refer to the built documentation and example notebooks

## Development Best Practices
- Write clear code
- Comment a lot
- Test thoroughly
- Follow existing code patterns and conventions
- Please avoid having AI agents implement unit tests for new features; they are meant to be a human-maintained safety net ensuring accuracy.

Thank you for your help!

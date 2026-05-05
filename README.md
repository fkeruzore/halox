<div align="center">
<img src="https://raw.githubusercontent.com/fkeruzore/halox/main/imgs/logo_text.png" alt="logo" width="500"></img>

# halox
JAX-powered Python library for differentiable dark matter halo property and mass function calculations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![tests](https://github.com/fkeruzore/halox/actions/workflows/tests.yml/badge.svg)
![coverage](https://raw.githubusercontent.com/fkeruzore/halox/main/imgs/coverage.svg)
[![PyPi version](https://img.shields.io/pypi/v/halox)](https://pypi.org/project/halox)
[![Documentation Status](https://readthedocs.org/projects/halox/badge/?version=latest)](https://halox.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2509.22478---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/2509.22478)

</div>

## Installation

`halox` can be installed via `pip`:

```bash
pip install halox
```

For a manual installation, see the [documentation pages](https://halox.readthedocs.io/en/latest/install.html).

## Features

`halox` offers a JAX-powered differentiable and GPU-accelerated implementation of some widely used properties of dark matter halos and large-scale structure, including:

* [`halox.halo`](https://halox.readthedocs.io/en/latest/notebooks/nfw.html): Radial profiles of dark matter halos following Navarro-Frenk-White (NFW) and Einasto distributions;
* [`halox.cm`](https://halox.readthedocs.io/en/latest/notebooks/cMrelations.html): Mass-concentration relations of dark matter halos;
* [`halox.lss`](https://halox.readthedocs.io/en/latest/notebooks/lss.html): Large-scale structure ($\sigma(R)$, $\sigma(M)$);
* [`halox.hmf`](https://halox.readthedocs.io/en/latest/notebooks/hmf.html): The halo mass function, quantifying the abundance of dark matter halos in mass and redshift and its dependence on cosmological parameters;
* [`halox.bias`](https://halox.readthedocs.io/en/latest/notebooks/bias.html): The halo bias;
* [`halox.emus`](https://halox.readthedocs.io/en/latest/notebooks/using_the_emulator.html): A neural-network emulator for $\sigma(M)$, providing up to 95× speedup over the analytic calculation while remaining differentiable and GPU-accelerated.

All functions support `jax.jit`, `jax.vmap`, and `jax.grad`.
Halo masses, redshifts, and cosmological parameters are all valid differentiation targets.
All properties support cosmology dependence using [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo).
More information on the modules available can be found in the [documentation pages](https://halox.readthedocs.io/en/latest/).

## Quick start

```python
import jax
import jax.numpy as jnp
from halox import cosmology, hmf, halo

jax.config.update("jax_enable_x64", True)  # optional

cosmo = cosmology.Planck18()

# Halo mass function at z = 0.5
M = jnp.logspace(13, 15, 100)
dn_dlnM = hmf.tinker08_mass_function(M, 0.5, cosmo)

# NFW profile
h = halo.nfw.NFWHalo(1e14, 5.0, 0.0, cosmo)
r = jnp.logspace(-1, 1, 50)
rho = h.density(r)

# Derivative w.r.t. halo mass
dhmf_dM = jax.grad(lambda M: hmf.tinker08_mass_function(M, 0.5, cosmo))(1e14)
```

## Units

All input/output units for `halox` functions are reported in docstrings.
Units are assumed to be in proper coordinates (not comoving) and include factors of [little h](https://arxiv.org/abs/1308.4150).

## Testing

All functions available in halox are validated against existing, non-JAX-based software, and a 100% coverage threshold is enforced.
Cosmology calculations are validated against [Astropy](https://www.astropy.org) for varying cosmological parameters and redshifts.
Other quantities are validated against [Colossus](https://bdiemer.bitbucket.io/colossus/index.html#) or [gala](https://gala.adrian.pw/en/latest/) for varying halo masses, redshifts, critical overdensities, and cosmological parameters.
These tests are included in the automatic CI/CD pipeline; a visual comparison is also included in the [documentation](https://halox.readthedocs.io/en/latest/notebooks/halox_vs_colossus.html).

## Documentation

For more detail on the code and features, please visit our [documentation pages](https://halox.readthedocs.io/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code conventions, and the pull-request process.

## Citation

If you use `halox` for your research, please cite the [original paper](https://arxiv.org/abs/2509.22478):

```bib
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
```

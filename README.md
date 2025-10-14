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

* [`halox.nfw`](https://halox.readthedocs.io/en/latest/notebooks/nfw.html): Radial profiles of dark matter halos following a Navarro-Frenk-White (NFW) distribution;
* [`halox.hmf`](https://halox.readthedocs.io/en/latest/notebooks/hmf.html): The halo mass function, quantifying the abundance of dark matter halos in mass and redshift, including its dependence on cosmological parameters;
* [`halox.bias`](https://halox.readthedocs.io/en/latest/notebooks/bias.html): The halo bias.

More information on the modules available can be found in the [documentation pages](https://halox.readthedocs.io/en/latest/physics_modules.html).

## Testing

All functions available in halox are validated against existing, non-JAX-based software.
Cosmology calculations are validated against [Astropy](https://www.astropy.org) for varying cosmological parameters and redshifts.
Other quantities are validated against [Colossus](https://bdiemer.bitbucket.io/colossus/index.html#) for varying halo masses, redshifts, critical overdensities, and cosmological parameters.
These tests are included in the automatic CI/CD pipeline; a visual comparison is also included in the [documentation](https://halox.readthedocs.io/en/latest/notebooks/halox_vs_colossus.html).

## Documentation

For more detail on the code and features, please visit our [documentation pages](https://halox.readthedocs.io/).

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


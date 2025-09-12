---
title: 'halox: Dark matter halo properties and large-scale structure calculations using JAX'
tags:
  - Python
  - astronomy
  - cosmology
authors:
  - name: Florian Kéruzoré
    orcid: 0000-0002-9605-5588
    affiliation: 1
affiliations:
 - name: High Energy Physics Division, Argonne National Laboratory, Lemont, IL 60439, USA
   index: 1
date: 8 September 2025
bibliography: paper.bib

---

# Summary

Dark matter halos are fundamental structures in cosmology, forming the gravitational potential wells hosting galaxies and clusters of galaxies.
Their properties and statistical distribution (the halo mass function) are invaluable tools to infer the fundamental properties of the Universe.
The `halox` package is a JAX-powered Python library enabling differentiable and accelerated computations of key properties of dark matter halos, and of the halo mass function.
The automatic differentiation capabilities of `halox` enable its usage in gradient-based workflows, *e.g.* in efficient Hamiltonian Monte Carlo sampling or machine learning applications.

# Statement of need

In cosmology and astrophysics, modeling dark matter halos is central to understanding the large-scale structure of the Universe and its formation.
This has motivated the development of many toolkits focused on halo modeling, such as, *e.g.*, halofit [@Smith:2003], halotools [@Hearin:2017], colossus [@Diemer:2018], or pyCCL [@Chisari:2019].
Recently, the AI-driven advent of novel computational frameworks such as JAX [@Bradbury:2018], have led to the development of differentiable and hardware-accelerated software to simulate and model physical processes, with *e.g.* Brax [@Brax:2021] and JAX, MD [@Jaxmd:2020].
The increasing complexity of cosmological data and astrophysical models has motivated the wide adoption of this framework in cosmology, where JAX-powered software has been published to address a wide variety of scientific goals, including
modeling fundamental cosmological quantities, with, *e.g.*, JAX-cosmo [@Campagne:2023] and LINX [@Giovanetti:2024];
simulating density fields and observables, with, *e.g.*, SHAMNet [@Hearin:2022], DISCO-DJ [@Hahn:2024], JAXpm [@Jaxpm:2025], and JAX-GalSim [@JaxGalSim:2025];
emulating likelihoods for accelerated inference, with, *e.g.*, CosmoPower-JAX [@Piras:2023] and candl [@Balkenhol:2024];
or modeling various physical properties of dark matter halos, such as mass acretion history [Diffmah, @Hearin:2021], galaxy star formation history [Diffstar, @Alarcon:2023], halo concentration [Diffprof, @Stevanovich:2023], gas-halo connection [picasso, @Keruzore:2024], and halo mass function [@Buisman:2025]^[Note that halox also provides an implementation of the halo mass function, but choses a lighter, halo model-based approach; see **Features**.].

The `halox` library offers a JAX implementation of some widely used properties which, while existing in other libraries focused on halo modeling, do not currently have a publicly available, differentiable and GPU-accelerated implementation, namely:

* Radial profiles of dark matter halos following a Navarro-Frenk-White [NFW, @Navarro:1997] distribution;
* The halo mass function, quantifying the abundance of dark matter halos in mass and redshift, including its dependence on cosmological parameters;
* The halo bias.

The use of JAX as a backend allows these functions to be compiled and GPU-accelerated, enabling high-performance computations; and automatically differentiable, enabling their efficient use in gradient-based workflows, such as sensitivity analyses, Hamiltonian Monte-Carlo sampling for Bayesian inference, or machine learning-based methods.

# Features

## Available physical quantities

The `halox` library seeks to provide JAX-based implementations of common models of dark matter halo properties and of large-scale structure.
At the time of writing (software version 1.1.0), this includes the following properties:

* Cosmological quantities: `halox` relies on JAX-cosmo [@Campagne:2023] for cosmology-dependent calculations, and includes wrapper functions to compute some additional properties, such as critical density $\rho_{\rm c}$ and differential comoving volume element ${\rm d}V_{c} / {\rm d}\Omega {\rm d}z$.
* Radially-dependent physical properties of NFW dark matter halos. Our implementations are based on the analytical derivations of @Lokas:2001, and include the following quantities:
  * Matter density $\rho(r)$;
  * Enclosed mass $M(\leq r)$;
  * Gravitational potential $\phi(r)$;
  * Circular velocity $v_{\rm circ}(r)$;
  * Velocity dispersion $\sigma_{v}(r)$;
  * Projected surface density $\Sigma(r)$.
* Large-scale structure: Building upon the power spectra computations implemented in JAX-cosmo, `halox` provides implementations of the RMS variance of the matter distribution in spheres of radius $R$, $\sigma(R)$. It also includes a wrapper function to perform the computation within the Lagrangian radius of a halo of mass $M$, $\sigma(M)$.
* Halo mass function (HMF): The HMF model of @Tinker:2008, predicting ${\rm d}N / {\rm d} \ln M$ as a function of halo mass $M$, redshift $z$, and cosmology.
* Halo bias: The linear bias model of @Tinker:2010 as a function of halo mass $M$, redshift $z$, and cosmology.
* Overdensities: All properties in `halox` can be computed for spherical overdensity (SO) halo masses defined for any critical overdensity value. Convenience functions are provided to convert halo properties from one critical overdensity to another, or to convert critical overdensities to mean matter overdensities.

## Automatic differentiation and hardware acceleration

All calculations available in `halox` are written using JAX and JAX-cosmo.
As a result, all functions can be compiled just-in-time using `jax.jit`, hardware-accelerated, and are automatically differentiable with respect to their input parameters, including halo mass, redshift, and cosmological parameters.

# Validation

All functions available in `halox` are validated against existing, non-JAX-based software.
Cosmology calculations are validated against Astropy [@Astropy:2022] for varying cosmological parameters and redshifts.
Other quantities are validated against `colossus` [@Diemer:2018] for varying halo masses, redshifts, critical overdensities, and cosmological parameters.
These tests are included in an automatic CI/CD pipeline on the GitHub repository, and presented graphically in the online documentation.

# Acknowledgments

I would like to thank Andrew Hearin and Lindsey Bleem for useful discussions and feedback on this manuscript.
I acknowledge the use of Anthropic's Claude Code in the development of `halox`.
Argonne National Laboratory’s work was supported by the U.S. Department of Energy, Office of Science, Office of High Energy Physics, under contract DE-AC02-06CH11357.

# References


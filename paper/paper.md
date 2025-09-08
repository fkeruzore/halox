---
title: 'halox: Dark matter halo properties and large scale structure calculations using JAX'
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
date: 4 September 2025
bibliography: paper.bib

---

# Summary

Dark matter halos are fundamental structures in cosmology, forming the gravitational potential wells hosting galaxies and clusters of galaxies.
Their properties and statistical distribution (the halo mass function) are invaluable tools to infer the fundamental properties of the Universe.
The `halox` package is a JAX-powered Python library enabling differentiable and accelerated computations of key iproperties of dark matter halos, and of the halo mass function.

# Statement of need

In cosmology and astrophysics, modeling dark matter halos is central to understanding the large-scale structure of the Universe and its formation.
This has motivated the development of many toolkits focused on halo modeling, such as, *e.g.*, halofit [@Smith:2003], halotools [@Hearin:2017], colossus [@Diemer:2018], or pyCCL [@Chisari:2019].
Recently, the increasing complexity of cosmological data and astrophysical models, along with the AI-driven advent of novel computational frameworks such as JAX [@Bradbury:2018], have led to the development of differentiable and hardware-accelerated software.
Such software has been published to model fundamental cosmological quantities--*e.g.*, @Campagne:2023, @Piras:2023; as well as various models of dark matter halos physical properties--*e.g.*, @Hearin:2021, @Hearin:2022, @Alarcon:2023, @Stevanovich:2023, @Keruzore:2024.

The `halox` package offers a JAX implementation of some widely used properties which, while existing in other libraries focused on halo modeling, do not currently have a publicly available, differentiable and GPU-accelerated implementation, namely:

* Radial profiles of dark matter halos following a Navarro-Frenk-White [NFW, @Navarro:1997] distribution;
* The halo mass function of @Tinker:2008, quantifying the abundance of dark matter halos in mass and redshift, including its dependence on cosmological parameters;
* The halo bias of @Tinker:2010.

The use of JAX as a backend allows these functions to be compilable and GPU-accelerated, enabling high-performance computations; and automatically differentiable, enabling their efficient use in gradient-based workflows, such as sensitivity analyses, Hamiltonian Monte-Carlo sampling for Bayesian inference, or machine learning-based methods.

# Features

## Available physical quantities

`halox` seels to provide JAX-based implementations of common models of dark matter halo properties and of large-scale structure.
At the time of writing (software version 1.1.0), this includes the following properties:

* Radially-dependent physical properties of NFW dark matter halos. Our implementations are based on the analytical derivations of @Lokas:2001, and include the following quantities:
  * Matter density $\rho(r)$;
  * Enclosed mass $M(\leq r)$;
  * Gravitational potential $\phi(r)$;
  * Circular velocity $v_{\rm circ}(r)$;
  * Velocity dispersion $\sigma_{v}(r)$;
  * Projected surface density $\Sigma(r)$.


## Automatic differentiation and hardware acceleration

JAX

# Acknowledgments

FK thanks Andrew Hearin and Lindsey Bleem for useful discussions, and acknowledges the use of Anthropic's Claude Code in the development of `halox`.
Argonne National Laboratory’s work was supported by the U.S. Department of Energy, Office of Science, Office of High Energy Physics, under contract DE-AC02-06CH11357.

# References


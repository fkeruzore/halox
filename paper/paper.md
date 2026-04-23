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
Their properties and statistical distribution (including the halo mass function) are invaluable tools to infer the fundamental properties of the Universe.
The `halox` package is a JAX-powered Python library enabling differentiable and accelerated computations of key properties of dark matter halos, and of the halo mass function.
The automatic differentiation capabilities of `halox` enable its usage in gradient-based workflows, *e.g.* in efficient Hamiltonian Monte Carlo sampling or machine learning applications.
The acceleration capabilities of `halox` enable faster calculation of implemented quantities than current packages such as `colossus`, though only when running using GPU architectures.

# Statement of need

In cosmology and astrophysics, modeling dark matter halos is central to understanding the large-scale structure of the Universe and its formation.
This has motivated the development of many toolkits focused on halo modeling, such as, *e.g.*, halofit [@Smith:2003], halotools [@Hearin:2017], colossus [@Diemer:2018], or pyCCL [@Chisari:2019].
Recently, the AI-driven advent of novel computational frameworks such as JAX [@Bradbury:2018], have led to the development of differentiable and hardware-accelerated software to simulate and model physical processes, with *e.g.* Brax [@Brax:2021] and JAX, MD [@Jaxmd:2020].
The increasing complexity of cosmological data and astrophysical models has motivated the wide adoption of this framework in cosmology, where JAX-powered software has been published to address a wide variety of scientific goals, including
modeling fundamental cosmological quantities, with, *e.g.*, JAX-cosmo [@Campagne:2023] and LINX [@Giovanetti:2024];
simulating density fields and observables, with, *e.g.*, SHAMNet [@Hearin:2022], DISCO-DJ [@Hahn:2024], JAXpm [@Jaxpm:2025], and JAX-GalSim [@Mendoza:2025; @JaxGalSim:2025];
emulating likelihoods for accelerated inference, with, *e.g.*, CosmoPower-JAX [@Piras:2023] and candl [@Balkenhol:2024];
or modeling various physical properties of dark matter halos, such as mass accretion history [Diffmah, @Hearin:2021], galaxy star formation history [Diffstar, @Alarcon:2023], halo concentration [Diffprof, @Stevanovich:2023], gas-halo connection [picasso, @Keruzore:2024], and halo mass function [@Buisman:2025]^[Note that halox also provides an implementation of the halo mass function, but chooses a lighter, halo model-based approach; see **Features**.].

The `halox` library offers a JAX implementation of some widely used properties which, while existing in other libraries focused on halo modeling, do not currently have a publicly available, differentiable and GPU-accelerated implementation, namely:

* Radial profiles of dark matter halos following Navarro-Frenk-White [@Navarro:1997] and Einasto [@Einasto:1965] distributions;
* Concentration-mass relations
* The halo mass function, quantifying the abundance of dark matter halos in mass and redshift, including its dependence on cosmological parameters;
* The halo bias.

The use of JAX as a backend allows these functions to be compiled and GPU-accelerated, enabling high-performance computations; and automatically differentiable, enabling their efficient use in gradient-based workflows, such as sensitivity analyses, Hamiltonian Monte-Carlo sampling for Bayesian inference, or machine learning-based methods. In addition, expensive computations of large-scale structure properties are further accelerated using neural network emulators, preserving hardware acceleration and differentiability while enabling faster calculations thanks to approximate calculations.

# Features

## Available physical quantities

The `halox` library seeks to provide JAX-based implementations of common models of dark matter halo properties and of large-scale structure.
At the time of writing (software version 1.2.0), this includes the following properties:

* Cosmological quantities: `halox` relies on JAX-cosmo [@Campagne:2023] for cosmology-dependent calculations, and includes wrapper functions to compute some additional properties, such as critical density $\rho_{\rm c}$ and differential comoving volume element ${\rm d}V_{c} / {\rm d}\Omega {\rm d}z$.
* Radially-dependent physical properties of NFW and Einasto dark matter halos. Our NFW and Einasto implementations are based on the analytical derivations of @Lokas:2001 and @Retana-Montenegro:2012 respectively, and include the following quantities:
  * Matter density $\rho(r)$;
  * Enclosed mass $M(\leq r)$;
  * Gravitational potential $\phi(r)$;
  * Circular velocity $v_{\rm circ}(r)$;
  * Velocity dispersion $\sigma_{v}(r)$ (NFW only);
  * Projected surface density $\Sigma(r)$ (NFW only).
* Concentration-mass relations: There are implementations of several relations including:
  * @Duffy:2008
  * @Klypin:2011
  * @Prada:2012
  * @Child:2018 (for all halo and relaxed halo populations)
* Large-scale structure: Building upon the power spectra computations implemented in JAX-cosmo, `halox` provides implementations of the RMS variance of the matter distribution in spheres of radius $R$, $\sigma(R)$. It also includes a wrapper function to perform the computation within the Lagrangian radius of a halo of mass $M$, $\sigma(M)$.
* Halo mass function (HMF): The HMF model of @Tinker:2008, predicting ${\rm d}N / {\rm d} \ln M$ as a function of halo mass $M$, redshift $z$, and cosmology.
* Halo bias: The linear bias model of @Tinker:2010 as a function of halo mass $M$, redshift $z$, and cosmology.
* Overdensities: All properties in `halox` can be computed for spherical overdensity (SO) halo masses defined for any critical overdensity value. Convenience functions are provided to convert halo properties from one critical overdensity to another, or to convert critical overdensities to mean matter overdensities.


## Automatic differentiation and hardware acceleration

All calculations available in `halox` are written using JAX and JAX-cosmo.
As a result, all functions can be compiled just-in-time using `jax.jit`, hardware-accelerated, and are automatically differentiable with respect to their input parameters, including halo mass, redshift, and cosmological parameters.

## Emulation

$\sigma(M)$ is the root-mean-square fluctuations of the density field for the region of space within radius R that would collapse to a total mass of M given the average density of the universe. This is found using the equation: 

$$\sigma^2(M,z) = D^2(z)\,\sigma^2(M, 0) = D^2(z) \, \frac{1}{2 \pi} \int _0 ^\infty k^2 P(k,R) dk $$

where $R = \left(\frac{3M}{4 \pi \bar{\rho}_0}\right)$. Computing this integral requires significant computational resources, making $\sigma(M)$ the primary bottleneck when computing the HMF and halo bias. Therefore, `halox` also includes an emulated calculation of this quantity using a multi-layer perceptron. The emulator is trained on the halox $\sigma(M)$ implementation. The training set is taken from a Sobol sample over log(M), log(1+z), and the cosmological parameters $\Omega_b$, $\Omega_c$, $h$, $n_s$, and $\sigma_8$. The emulator accepts input vectors in M, z, and those same cosmological parameters, and this input mirrors the inputs for the original function. The emulator is accurate to within a percent for both $\sigma(M)$ and the halo bias, and within six percent for the HMF. To compute $\sigma(M)$, HMF, or halo bias using the emulator, simply instantiate the emulator, then pass it in as the “emu” argument to the original $\sigma(M)$ function in halox as seen below.
```
# analytical calculation
sigma_analytical = lss.sigmaM(M, z, cosmo)

# emulated calculation (using the default network weights)
emu = emus.sigmaM.SigmaMEmulation()
sigma_emulated = lss.sigmaM(M, z, cosmo, emu = emu)
```
![Comparing emulated and non-emulated $\sigma(M)$ calculations, plotted against mass, varying both redshift and cosmology. Residuals stay below the percent level. \label{fig:figure1}](sigmaM_emulator_validation.png)

![Comparing HMF calculation using emulated and non-emulated evaluations of $\sigma(M)$. \label{fig:figure2}](hmf_emulator_validation.png)

# Speedup

To benchmark the speed up provided by calculating with `halox`, the tool was tested on different architectures, both with and without JIT compilation. JIT compilation alone provides a significant acceleration, but leveraging GPU architecture and JIT compilation provides an even greater speedup. Emulation provides even more acceleration; the JIT compiled, emulated function running on GPUs is even faster than the non-emulated counterpart. `halox` is still slower than `colossus` [@Diemer:2018] when JIT compiled on CPU architecture, but can still provide considerable speedup over `halox` and `colossus` using GPU architecture.

![The performance of HMF computation for the halox package on different architectures and against `colossus`. All CPU runs are still slower than `colossus` irrespective of emulation. GPU architecture enables further speedup, allowing for faster computations than `colossus` both with and without emulation, with significant speedup when using the emulated function over the standard calculation. \label{fig:figure3}](benchmark_hmf_results.png)

# Validation

All functions available in `halox` are validated against existing, non-JAX-based software.
Cosmology calculations are validated against Astropy [@Astropy:2022] for varying cosmological parameters and redshifts.
Other quantities are validated against `colossus` [@Diemer:2018] for varying halo masses, redshifts, critical overdensities, and cosmological parameters.
These tests are included in an automatic CI/CD pipeline on the GitHub repository, and presented graphically in the online documentation.

# Acknowledgments

We would like to thank Andrew Hearin, Matt Becker, Georgios Zacharegkas, and Lindsey Bleem for useful discussions and feedback on `halox`.
We acknowledge the use of Anthropic's Claude Code in the development of `halox`.
Argonne National Laboratory’s work was supported by the U.S. Department of Energy, Office of Science, Office of High Energy Physics, under contract DE-AC02-06CH11357.

# References


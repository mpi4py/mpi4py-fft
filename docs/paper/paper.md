---
title: 'mpi4py-fft: Parallel Fast Fourier Transforms with MPI for Python'
tags:
 - Fast Fourier transforms
 - Fast Chebyshev transforms
 - MPI
 - Python
authors:
 - name: Mikael Mortensen
   orcid: 0000-0002-3293-7573
   affiliation: "1"
 - name: Lisandro Dalcin
   orcid: 0000-0001-8086-0155
   affiliation: "2"
 - name: David Elliot Keyes
   orcid: 0000-0002-4052-7224
   affiliation: "2"
affiliations:
 - name: University of Oslo, Department of Mathematics
   index: 1
 - name: King Abdullah University of Science and Technology, Extreme Computing Research Center
   index: 2
date: 4 March 2019
bibliography: paper.bib
---

# Summary

The fast Fourier transform (FFT) is an algorithm that efficiently
computes the discrete Fourier transform. Since the dawn
of our digital society, the FFT has permeated to the heart of everyday
life applications involving audio, image, and video processing. The
importance of the FFT extends beyond signal processing into scientific
computing because it diagonalizes the Poisson operator, which is
ubiquitous in descriptions of electromagnetism,  gravitation,
acoustic and elastic waves, potential flow in fluids, stress in
solids, Hamiltonians of the  Schroedinger’s equation for probability
distribution of electrons in quantum mechanics, and diffusion of
internal energy, chemical species, and momentum.  The
FFT has been named *the most important numerical algorithm of our
time* by Professor Gilbert Strang [@strang94].

``mpi4py-fft`` (https://github.com/mpi4py/mpi4py-fft) is an
open-source Python package for computing (in parallel) FFTs of
possibly very large and distributed multidimensional arrays. A
multidimensional FFT is computed in sequence, over all axes, one axis
at the time. In order to fit in the memory of multiple processors,
multidimensional arrays have to be distributed along some, but not
all, of its axes.  Consequently, parallel FFTs are computed with
successive sequential (serial) transforms over undivided axes,
combined with global array redistributions (using interprocess
communication) that realign the arrays for further serial transforms.

For global redistributions, ``mpi4py-fft`` makes use of a new and
completely generic algorithm [@dalcin18] based on advanced MPI
features that allows for any index sets of a multidimensional array to
be distributed. It can distribute a single index set (slab
decomposition), two index sets (pencil decomposition), or even more
for higher-dimensional arrays. The required MPI communications are
always handled under the hood by MPI for Python [@mpi4py08]. For
serial FFT transforms, ``mpi4py-fft`` uses Cython [@cython11] to wrap
most of the FFTW library [@fftw05] and provide support for
complex-to-complex, real-to-complex, complex-to-real and real-to-real
transforms.

``mpi4py-fft`` is highly configurable in how it distributes and
redistributes arrays. Large arrays may be globally redistributed for
alignment along any given axis, whenever needed by the user. This
flexibility has enabled the development of ``shenfun``
[@mortensen_joss, @mortensen17], which is a Python framework for
solving partial differential equations (PDEs) by the spectral Galerkin
method. ``shenfun`` is able to solve PDEs of any given dimensionality
by creating tensor product bases as outer products of one-dimensional
bases. This leads to large multidimensional arrays that are
distributed effortlessly through ``mpi4py-fft``.
Throughout the ``spectralDNS`` (https://github.com/spectralDNS/spectralDNS)
project ``shenfun`` is being used extensively for
Direct Numerical Simulations (DNS) of turbulent flows
[@mortensen16, @mortensen16b, @ketcheson], using arrays with billions of
unknowns.

``mpi4py-fft`` provides a flexible distributed NumPy array interface,
which allows for further reuse in applications beyond the FFT. The
distribution requires at least one undivided axis, but apart from that
there are no restrictions nor limitations. The interface can
simply be used to boost performance of global array operations through MPI.

# Acknowledgements

M Mortensen acknowledges support from the 4DSpace Strategic Research
Initiative at the University of Oslo.

L Dalcin and D Keyes acknowledge support from the Extreme Computing Research
Center and the KAUST Supercomputing Laboratory at King Abdullah
University of Science and Technology.

# References

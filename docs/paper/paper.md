---
title: 'Mpi4py-fft'
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
affiliations:
 - name: University of Oslo, Department of Mathematics
   index: 1
 - name: King Abdullah University of Science and Technology, Extreme Computing Research Center
   index: 2
date: 7 November 2018
bibliography: paper.bib
---

# Summary

The fast Fourier transform (FFT) is an algorithm that efficiently computes the
discrete Fourier transform. The FFT is one of the most important algorithms
utilized throughout science and society and it has been named *the most
important numerical algorith of our time* by Prof Gilbert Strang [@strang].

``mpi4py-fft`` (https://bitbucket.org/mpi4py/mpi4py-fft) is an open-source
Python package for computing (in parallel) FFTs of possibly very large and
distributed multidimensional arrays.
A multidimensional FFT is computed sequentially, over all axes, one axis at the time.
A problem with parallel FFTs is that, to fit in the memory of multiple processors,
multidimensional arrays will be distributed along some, but not all, of its axes.
Consequently, parallel FFTs are computed as sequential (serial) transforms over
non-distributed axes, combined with global redistributions (using MPI) that
realign the arrays for further serial transforms. A parallel FFT is, in other
words, computed as a combination of serial FFTs and global redistributions.

For global redistribution ``mpi4py-fft`` makes use of a new and completely
generic algorithm [@dalcin18] that allows for any index sets of a
multidimensional array to be distributed. We can distribute just one index
(a slab decomposition), two index sets (pencil decomposition) or even more for
higher-dimensional arrays. The required MPI communications are always handled
under the hood by MPI for Python. For serial transforms
``mpi4py-fft`` wraps most of the FFTW library using Cython, making it callable
from Python. We include wrappers for complex-to-complex, real-to-complex,
complex-to-real and real-to-real transforms.

``mpi4py-fft`` is highly configurable in how it distributes and redistributes
arrays. Large arrays may be globally redistributed for alignement
along any given axis, whenever needed by the user. This
flexibility has enabled the development of ``shenfun``
[@mortensen_joss,@mortensen17], which is a computing platform
for solving partial differential equations (PDEs) by the spectral Galerkin method.
In ``shenfun`` it is possible to solve PDEs of any given dimensionality, by creating
tensor product bases as outer products of one-dimensional bases. This leads to
large multidimensional arrays that are distributed effortlessly through ``mpi4py-fft``.

``mpi4py-fft`` can be utilized by anyone that needs to perform FFTs on large
multidimensional arrays. Through its distributed array interface it can also be
utilized by any application relying on algorithms (not just FFTs) with varying
degrees of locality on multidimensional arrays, where MPI can be used to boost
performance.

``mpi4py-fft`` is installable from ``pypi`` and ``conda-forge``, and
released under a permissive 2-clause BSD-license, in the hope that it will be
useful.

# Acknowledgements

M Mortensen acknowledges support from the 4DSpace Strategic Research Initiative at the
University of Oslo

# References

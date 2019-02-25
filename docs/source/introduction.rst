Introduction
============

The Python package mpi4py-fft is a tool primarily for working with Fast
Fourier Transforms (FFTs) of (large) multidimensional arrays. There is really
no limit as to how large the arrays can be, just as long as there is sufficient
computing powers available. Also, there are no limits as to how transforms can
be configured. Just about any combination of transforms from the FFTW library
is supported. Furthermore, mpi4py-fft can also be used simply to perform global
redistributions (distribute and communicate) of large arrays with MPI, without
any transforms at all.

The main contribution of mpi4py-fft can be found in just a few classes in
the main modules:

    * :mod:`.mpifft`
    * :mod:`.pencil`
    * :mod:`.distributedarray`
    * :mod:`.libfft`
    * :mod:`.fftw`

The :class:`.mpifft.PFFT` class is the major entry point for most users. It is a
highly configurable class, which under the hood distributes large dataarrays and
performs any type of transform, along any axes of a multidimensional array.

The :mod:`.pencil` module is responsible for global redistributions through MPI.
However, this module is rarely used on its own, unless one simply needs to do
global redistributions without any transforms at all. The :mod:`.pencil` module
is used heavily by the :class:`.PFFT` class.

The :mod:`.distributedarray` module contains classes for simply distributing
multidimensional arrays, with no regard to transforms. The distributed arrays
created from the classes here can very well be used in any MPI application that
requires a large multidimensional distributed array.

The :mod:`.libfft` module provides a common interface to any of the serial
transforms in the `FFTW <http://www.fftw.org>`_  library.

The :mod:`.fftw` module contains wrappers to the transforms provided by the
`FFTW <http://www.fftw.org>`_ library. We provide our own wrappers mainly
because `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ does not include support
for real-to-real transforms. Through the interface in :mod:`.fftw` we can do
here, in Python, pretty much everything that you can do in the original
FFTW library.

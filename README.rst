mpi4py-fft
----------


.. image:: https://circleci.com/bb/mpi4py/mpi4py-fft.svg?style=svg
    :target: https://circleci.com/bb/mpi4py/mpi4py-fft

.. image:: https://api.codacy.com/project/badge/Grade/edf0f9ed1e114ab090ac4f9863c05fa3
    :target: https://www.codacy.com/app/mikaem/mpi4py-fft?utm_source=mpi4py@bitbucket.org&amp;utm_medium=referral&amp;utm_content=mpi4py/mpi4py-fft&amp;utm_campaign=Badge_Grade

.. image:: https://codecov.io/bb/mpi4py/mpi4py-fft/branch/master/graph/badge.svg
  :target: https://codecov.io/bb/mpi4py/mpi4py-fft

.. image:: https://readthedocs.org/projects/mpi4py-fft/badge/?version=latest
   :target: https://mpi4py-fft.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


mpi4py-fft is a Python package for computing Fast Fourier Transforms (FFTs).
Large arrays are distributed and communications are handled under the hood by
MPI for Python (mpi4py). To distribute large arrays we are using a
`new and completely generic algorithm <https://arxiv.org/abs/1804.09536>`_
that allows for any index set of a multidimensional array to be distributed. We
can distribute just one index (a slab decomposition), two index sets (pencil
decomposition) or even more for higher-dimensional arrays.

In mpi4py-fft there is also included a Python interface to the
`FFTW <http://www.fftw.org>`_ library. This interface can be used without MPI,
much like `pyfftw <https://hgomersall.github.io/pyFFTW/>`_, and even for
real-to-real transforms, like discrete cosine or sine transforms.

Further documentation is found at `readthedocs <https://mpi4py-fft.readthedocs.io/en/latest/>`_.

Installation
------------

The mpi4py-fft package can be installed using::

    pip install mpi4py-fft

or, to get the latest version from bitbucket::

    pip install git+https://bitbucket.org/mpi4py/mpi4py-fft@master

Install with conda from the coda-forge channel::

    conda install -c conda-forge mpi4py-fft

or build it with conda build from the main source directory::

    conda build -c conda-forge conf/
    conda create --name mpi4py_fft mpi4py_fft --use-local

which will pull in the required dependencies from the conda-forge channel.

Note that mpi4py-fft depends on Python packages

    * mpi4py
    * numpy
    * cython

and the C-library

    * `FFTW <http://www.fftw.org>`_

Note in particular that *mpi4py* requires that you have a working
MPI installation, with the compiler wrapper *mpicc*, on your search path.
The FFTW header and libraries must also be available on the search path, and
we will build wrappers for any precision found of the FFTW libraries.

All of the above dependencies are available and will be downloaded through
the conda-forge channel if conda is used for installation. However, pip
will not help you with MPI or FFTW.

For IO you need to install either `h5py <https://www.h5py.org>`_ or
`netCDF4 <http://unidata.github.io/netcdf4-python/>`_ with support for
MPI. These libraries are, unfortunately, not yet compiled with MPI on
conda-forge. The two libraries are available, though, for both OSX and
linux from the `spectralDNS <https://anaconda.org/spectralDNS>`_ channel
on anaconda cloud::

    conda install -c spectralDNS h5py-parallel netcdf4-parallel

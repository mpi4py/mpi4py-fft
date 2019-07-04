"""
This is the **mpi4py-fft** package

What is **mpi4py-fft**?
=======================

The Python package **mpi4py-fft** is a tool primarily for working with Fast
Fourier Transforms (FFTs) of (large) multidimensional arrays. There is really
no limit as to how large the arrays can be, just as long as there is sufficient
computing powers available. Also, there are no limits as to how transforms can
be configured. Just about any combination of transforms from the FFTW library
is supported. Furthermore, **mpi4py-fft** can also be used simply to perform
global redistributions (distribute and communicate) of large arrays with MPI,
without any transforms at all.

For more information, see `documentation <https://mpi4py-fft.readthedocs.io>`_.

"""
__version__ = '2.0.3'
__author__ = 'Lisandro Dalcin and Mikael Mortensen'

from .distarray import DistArray, newDistArray, Function
from .mpifft import PFFT
from . import fftw
from .io import HDF5File, NCFile, generate_xdmf

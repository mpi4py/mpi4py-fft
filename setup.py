#!/usr/bin/env python

import os, sys
from distutils.core import setup, Extension
from numpy import get_include
from Cython.Build import cythonize

cwd = os.path.abspath(os.path.dirname(__file__))
fftwdir = os.path.join(cwd, "mpi4py_fft", "fftw")

ext = cythonize([Extension("mpi4py_fft.fftw.xfftn",
                                sources=[os.path.join(fftwdir, "xfftn.pyx"),
                                         os.path.join(fftwdir, "planxfftn.c")],
                                libraries=['m', 'fftw3', 'fftw3_threads',
                                           'fftw3f', 'fftw3f_threads',
                                           'fftw3l', 'fftw3l_threads'],
                                include_dirs=[get_include(),
                                              os.path.join(sys.prefix, 'include')],
                                library_dirs=[os.path.join(sys.prefix, 'lib')])])[0]

setup(name = "mpi4py-fft",
      version = "1.0-beta",
      description = "mpi4py-fft -- FFT with MPI",
      long_description = "",
      author = "Lisandro Dalcin and Mikael Mortensen",
      url = 'https://bitbucket.org/mpi4py/mpi4py-fft',
      packages = ["mpi4py_fft",
                  "mpi4py_fft.fftw"],
      package_dir = {"mpi4py_fft": "mpi4py_fft"},
      ext_modules = [ext]
    )

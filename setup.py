#!/usr/bin/env python

from distutils.core import setup

setup(name = "mpi4py-fft",
      version = "1.0-beta",
      description = "mpi4py-fft -- FFT with MPI",
      long_description = "",
      author = "Lisandro Dalsin and Mikael Mortensen",
      url = 'https://bitbucket.org/mpi4py/mpi4py-fft',
      packages = ["mpi4py_fft"],
      package_dir = {"mpi4py_fft": "mpi4py_fft"}
    )

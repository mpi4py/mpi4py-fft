#!/usr/bin/env python

import os, sys
import shutil
from distutils.core import setup, Extension
from numpy import get_include
from Cython.Build import cythonize

cwd = os.path.abspath(os.path.dirname(__file__))
fftwdir = os.path.join(cwd, 'mpi4py_fft', 'fftw')

# For now assuming that all precisions are available

prec = {'fftwf_': 'float', 'fftw': 'double', 'fftwl_': 'long double'}
libs = {
    'fftwf_': ['m', 'fftw3f', 'fftw3f_threads'],
    'fftw_': ['m', 'fftw3', 'fftw3_threads'],
    'fftwl_': ['m', 'fftw3l', 'fftw3l_threads']}

for fl in ('fftw_planxfftn.h', 'fftw_planxfftn.c', 'fftw_xfftn.pyx', 'fftw_xfftn.pxd'):
    for p in ('fftwf_', 'fftwl_'):
        fp = fl.replace('fftw_', p)
        shutil.copy(os.path.join(fftwdir, fl), os.path.join(fftwdir, fp))
        sedcmd = "sed -i ''" if sys.platform == 'darwin' else "sed -i''"
        os.system(sedcmd + " 's/fftw_/{0}/g' {1}".format(p, os.path.join(fftwdir, fp)))
        os.system(sedcmd + " 's/double/{0}/g' {1}".format(prec[p], os.path.join(fftwdir, fp)))

ext = cythonize([Extension("mpi4py_fft.fftw.utilities".format(p),
                           sources=[os.path.join(fftwdir, "utilities.pyx".format(p))],
                                    libraries=libs[p],
                           include_dirs=[get_include(),
                                         os.path.join(sys.prefix, 'include')],
                           library_dirs=[os.path.join(sys.prefix, 'lib')])])

for p in ('fftw_', 'fftwf_', 'fftwl_'):
    ext.append(cythonize([Extension("mpi4py_fft.fftw.{}xfftn".format(p),
                                    sources=[os.path.join(fftwdir, "{}xfftn.pyx".format(p)),
                                             os.path.join(fftwdir, "{}planxfftn.c".format(p))],
                                    #define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                                    libraries=libs[p],
                                    include_dirs=[get_include(),
                                                  os.path.join(sys.prefix, 'include')],
                                    library_dirs=[os.path.join(sys.prefix, 'lib')])])[0])

setup(name="mpi4py-fft",
      version="1.0-beta",
      description="mpi4py-fft -- FFT with MPI",
      long_description="",
      author="Lisandro Dalcin and Mikael Mortensen",
      url='https://bitbucket.org/mpi4py/mpi4py-fft',
      packages=["mpi4py_fft",
                "mpi4py_fft.fftw"],
      package_dir={"mpi4py_fft":"mpi4py_fft"},
      ext_modules=ext
)

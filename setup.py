#!/usr/bin/env python

import os, sys
import shutil
from setuptools import setup, find_packages
from setuptools.extension import Extension
from numpy import get_include

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

ext = [Extension("mpi4py_fft.fftw.utilities",
                 sources=[os.path.join(fftwdir, "utilities.pyx")],
                 libraries=libs[p],
                 include_dirs=[get_include(),
                               os.path.join(sys.prefix, 'include')],
                 library_dirs=[os.path.join(sys.prefix, 'lib')])]

for p in ('fftw_', 'fftwf_', 'fftwl_'):
    ext.append(Extension("mpi4py_fft.fftw.{}xfftn".format(p),
                         sources=[os.path.join(fftwdir, "{}xfftn.pyx".format(p)),
                                  os.path.join(fftwdir, "{}planxfftn.c".format(p))],
                         #define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                         libraries=libs[p],
                         include_dirs=[get_include(),
                                       os.path.join(sys.prefix, 'include')],
                         library_dirs=[os.path.join(sys.prefix, 'lib')]))

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(name="mpi4py-fft",
      version="1.0.1",
      description="mpi4py-fft -- FFT with MPI",
      long_description=long_description,
      author="Lisandro Dalcin and Mikael Mortensen",
      url='https://bitbucket.org/mpi4py/mpi4py-fft',
      packages=["mpi4py_fft",
                "mpi4py_fft.fftw",
                "mpi4py_fft.utilities"],
      package_dir={"mpi4py_fft": "mpi4py_fft"},
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: BSD License',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      ext_modules=ext,
      install_requires=["mpi4py", "numpy", "six"],
      setup_requires=["setuptools>=18.0", "cython>=0.25"]
      )

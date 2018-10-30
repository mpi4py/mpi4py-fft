#!/usr/bin/env python

import os, sys
import shutil
from distutils import ccompiler
from setuptools import setup
from setuptools.extension import Extension
from numpy import get_include

cwd = os.path.abspath(os.path.dirname(__file__))
fftwdir = os.path.join(cwd, 'mpi4py_fft', 'fftw')

include_dirs = [get_include(), os.path.join(sys.prefix, 'include')]
library_dirs = [os.path.join(sys.prefix, 'lib')]
for f in ('FFTW_ROOT', 'FFTW_DIR'):
    if f in os.environ['PATH']:
        library_dirs.append(os.path.join(os.environ[f], 'lib'))
        include_dirs.append(os.path.join(os.environ[f], 'include'))

compiler = ccompiler.new_compiler()
assert compiler.find_library_file(library_dirs, 'fftw3'), 'Cannot find FFTW library!'
has_threads = compiler.find_library_file(library_dirs, 'fftw3_threads')

prec_map = {'float': 'fftwf_', 'double': 'fftw_', 'long double': 'fftwl_'}

libs = {
    'float': ['fftw3f'],
    'double': ['fftw3'],
    'long double': ['fftw3l']
    }

has_prec = ['double']
for d in ('float', 'long double'):
    if compiler.find_library_file(library_dirs, libs[d][0]):
        has_prec.append(d)

if has_threads:
    for d in has_prec:
        libs[d].append('_'.join((libs[d][0], 'threads')))
        if sys.platform in ('unix', 'darwin'):
            libs[d].append('m')

# Generate files with float and long double if needed
for d in has_prec[1:]:
    p = prec_map[d]
    for fl in ('fftw_planxfftn.h', 'fftw_planxfftn.c', 'fftw_xfftn.pyx', 'fftw_xfftn.pxd'):
        fp = fl.replace('fftw_', p)
        shutil.copy(os.path.join(fftwdir, fl), os.path.join(fftwdir, fp))
        sedcmd = "sed -i ''" if sys.platform == 'darwin' else "sed -i''"
        os.system(sedcmd + " 's/fftw_/{0}/g' {1}".format(p, os.path.join(fftwdir, fp)))
        os.system(sedcmd + " 's/double/{0}/g' {1}".format(d, os.path.join(fftwdir, fp)))

ext = [Extension("mpi4py_fft.fftw.utilities",
                 sources=[os.path.join(fftwdir, "utilities.pyx")],
                 include_dirs=[get_include(),
                               os.path.join(sys.prefix, 'include')])]

for d in has_prec:
    p = prec_map[d]
    ext.append(Extension("mpi4py_fft.fftw.{}xfftn".format(p),
                         sources=[os.path.join(fftwdir, "{}xfftn.pyx".format(p)),
                                  os.path.join(fftwdir, "{}planxfftn.c".format(p))],
                         #define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                         libraries=libs[d],
                         include_dirs=include_dirs,
                         library_dirs=library_dirs))

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

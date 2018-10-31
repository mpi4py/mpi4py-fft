#!/usr/bin/env python

import os
import sys
import re
import shutil
from distutils import ccompiler
from setuptools import setup
from setuptools.extension import Extension
from numpy import get_include

cwd = os.path.abspath(os.path.dirname(__file__))
fftwdir = os.path.join(cwd, 'mpi4py_fft', 'fftw')
prec_map = {'float': 'f', 'double': '', 'long double': 'l'}

def get_library_dirs():
    lib_dirs = [os.path.join(sys.prefix, 'lib')]
    for f in ('FFTW_ROOT', 'FFTW_DIR'):
        if f in os.environ:
            lib_dirs.append(os.path.join(os.environ[f], 'lib'))
    return lib_dirs

def get_include_dirs():
    inc_dirs = [get_include(), os.path.join(sys.prefix, 'include')]
    for f in ('FFTW_ROOT', 'FFTW_DIR'):
        if f in os.environ:
            inc_dirs.append(os.path.join(os.environ[f], 'include'))
    return inc_dirs

def get_fftw_libs():
    """Return FFTW libraries"""
    compiler = ccompiler.new_compiler()
    library_dirs = get_library_dirs()
    libs = {}
    for d in ('float', 'double', 'long double'):
        lib = 'fftw3'+prec_map[d]
        if compiler.find_library_file(library_dirs, lib):
            libs[d] = [lib]
            tlib = '_'.join((lib, 'threads'))
            if compiler.find_library_file(library_dirs, tlib):
                libs[d].append(tlib)
            if sys.platform in ('unix', 'darwin'):
                libs[d].append('m')
    assert len(libs) > 0, "No FFTW libraries found in {} {}".format(library_dirs, sys.prefix)
    return libs

def generate_extensions(fftwlibs):
    """Generate files with float and long double"""
    for d in fftwlibs:
        if d == 'double':
            continue
        p = 'fftw'+prec_map[d]+'_'
        for fl in ('fftw_planxfftn.h', 'fftw_planxfftn.c', 'fftw_xfftn.pyx', 'fftw_xfftn.pxd'):
            fp = fl.replace('fftw_', p)
            shutil.copy(os.path.join(fftwdir, fl), os.path.join(fftwdir, fp))
            sedcmd = "sed -i ''" if sys.platform == 'darwin' else "sed -i''"
            os.system(sedcmd + " 's/fftw_/{0}/g' {1}".format(p, os.path.join(fftwdir, fp)))
            os.system(sedcmd + " 's/double/{0}/g' {1}".format(d, os.path.join(fftwdir, fp)))

def get_extensions(fftwlibs):
    """Return list of extension modules"""
    include_dirs = get_include_dirs()
    library_dirs = get_library_dirs()
    ext = [Extension("mpi4py_fft.fftw.utilities",
                     sources=[os.path.join(fftwdir, "utilities.pyx")],
                     include_dirs=include_dirs)]

    for d, libs in fftwlibs.items():
        p = 'fftw'+prec_map[d]+'_'
        ext.append(Extension("mpi4py_fft.fftw.{}xfftn".format(p),
                             sources=[os.path.join(fftwdir, "{}xfftn.pyx".format(p)),
                                      os.path.join(fftwdir, "{}planxfftn.c".format(p))],
                             #define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                             libraries=libs,
                             include_dirs=include_dirs,
                             library_dirs=library_dirs))
    return ext

def version():
    srcdir = os.path.join(cwd, 'mpi4py_fft')
    with open(os.path.join(srcdir, '__init__.py')) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
        return m.groups()[0]

with open("README.rst", "r") as fh:
    long_description = fh.read()

if __name__ == '__main__':
    fftw_libs = get_fftw_libs()
    generate_extensions(fftw_libs)
    setup(name="mpi4py-fft",
          version=version(),
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
          ext_modules=get_extensions(fftw_libs),
          install_requires=["mpi4py", "numpy"],
          setup_requires=["setuptools>=18.0", "cython>=0.25"]
          )

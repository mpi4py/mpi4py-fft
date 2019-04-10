#!/usr/bin/env python

import os
import sys
import re
import platform
import sysconfig
from distutils import ccompiler
from setuptools import setup
from setuptools.extension import Extension
import numpy

cwd = os.path.abspath(os.path.dirname(__file__))
fftwdir = os.path.join(cwd, 'mpi4py_fft', 'fftw')
prec_map = {'float': 'f', 'double': '', 'long double': 'l'}
triplet = sysconfig.get_config_var('MULTIARCH') or ''
bits = platform.architecture()[0][:-3]

def append(dirlist, *args):
    entry = os.path.join(*args)
    entry = os.path.normpath(entry)
    if os.path.isdir(entry):
        if entry not in dirlist:
            dirlist.append(entry)

def get_prefix_dirs():
    dirs = []
    for envvar in ('FFTW_ROOT', 'FFTW_DIR'):
        if envvar in os.environ:
            prefix = os.environ[envvar]
            append(dirs, prefix)
    append(dirs, sys.prefix)
    if 'CONDA_BUILD' not in os.environ:
        append(dirs, '/usr')
    return dirs

def get_include_dirs():
    dirs = []
    if 'FFTW_INCLUDE_DIR' in os.environ:
        entry = os.environ['FFTW_INCLUDE_DIR']
        append(dirs, entry)
    for prefix in get_prefix_dirs():
        append(dirs, prefix, 'include', triplet)
        append(dirs, prefix, 'include')
    dirs.append(numpy.get_include())
    return dirs

def get_library_dirs():
    dirs = []
    if 'FFTW_LIBRARY_DIR' in os.environ:
        entry = os.environ['FFTW_LIBRARY_DIR']
        append(dirs, entry)
    for prefix in get_prefix_dirs():
        append(dirs, prefix, 'lib' + bits)
        append(dirs, prefix, 'lib', triplet)
        append(dirs, prefix, 'lib')
    return dirs

def get_fftw_libs():
    """Return FFTW libraries"""
    compiler = ccompiler.new_compiler()
    library_dirs = get_library_dirs()
    libs = {}
    for d in ('float', 'double', 'long double'):
        lib = 'fftw3'+prec_map[d]
        tlib = lib+'_threads'
        if compiler.find_library_file(library_dirs, lib):
            libs[d] = [lib]
            if compiler.find_library_file(library_dirs, tlib):
                libs[d].append(tlib)
            if os.name == 'posix':
                libs[d].append('m')
    assert len(libs) > 0, "No FFTW libraries found in {}".format(library_dirs)
    return libs

def generate_extensions(fftwlibs):
    """Generate files with float and long double"""
    for d in fftwlibs:
        if d == 'double':
            continue
        p = 'fftw'+prec_map[d]+'_'
        for fname in (
                'fftw_planxfftn.h',
                'fftw_planxfftn.c',
                'fftw_xfftn.pyx',
                'fftw_xfftn.pxd',
        ):
            src = os.path.join(fftwdir, fname)
            dst = os.path.join(fftwdir, fname.replace('fftw_', p))
            with open(src, 'r') as fin:
                code = fin.read()
                code = re.sub('fftw_', p, code)
                code = re.sub('double', d, code)
                with open(dst, 'w') as fout:
                    fout.write(code)

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
          description="mpi4py-fft -- Parallel Fast Fourier Transforms (FFTs) using MPI for Python",
          long_description=long_description,
          author="Lisandro Dalcin and Mikael Mortensen",
          url='https://bitbucket.org/mpi4py/mpi4py-fft',
          packages=["mpi4py_fft",
                    "mpi4py_fft.fftw",
                    "mpi4py_fft.io"],
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
          setup_requires=["setuptools>=18.0", "cython>=0.25"],
          keywords=['Python', 'FFTW', 'FFT', 'DCT', 'DST', 'MPI']
          )

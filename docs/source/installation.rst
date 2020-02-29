Installation
============

Mpi4py-fft has a few dependencies

    * `mpi4py`_
    * `FFTW`_ (serial)
    * `numpy`_
    * `cython`_ (build dependency)
    * `h5py`_ (runtime dependency, optional)
    * `netCDF4`_ (runtime dependency, optional)

that are mostly straight-forward to install, or already installed in
most Python environments. The first two are usually most troublesome.
Basically, for `mpi4py`_ you need to have a working MPI installation,
whereas `FFTW`_ is available on most high performance computer systems.
If you are using `conda`_, then all you need to install a fully functional
mpi4py-fft, with all the above dependencies, is

::

    conda install -c conda-forge mpifpy-fft h5py=*=mpi*

You probably want to install into a fresh environment, though, which
can be achieved with

::

    conda create --name mpi4py-fft -c conda-forge mpi4py-fft
    conda activate mpi4py-fft

Note that this gives you mpi4py-fft with default settings. This means that
you will probably get the openmpi backend. To make a specific choice of 
backend just specify which, like this

::

    conda create --name mpi4py-fft -c conda-forge mpi4py-fft mpich

If you do not use `conda`_, then you need to make sure that MPI
and FFTW are installed by some other means. You can then install
any version of mpi4py-fft hosted on `pypi`_ using `pip`_

::

    pip install mpi4py-fft

whereas either one of the following will install the latest version
from github

::

    pip install git+https://bitbucket.org/mpi4py/mpi4py-fft@master
    pip install https://bitbucket.org/mpi4py/mpi4py-fft/get/master.zip

You can also build mpi4py-fft yourselves from the top directory,
after cloning or forking

::

    pip install .

or using `conda-build`_ with the recipes in folder ``conf/``

::

    conda build -c conda-forge conf/
    conda create --name mpi4py-fft -c conda-forge mpi4py-fft --use-local
    conda activate mpi4py-fft


Additional dependencies
-----------------------

For storing and retrieving data you need either `HDF5`_ or `netCDF4`_, compiled
with support for MPI. Both are available
with parallel support on `conda-forge`_ and can be installed into the 
current conda environment as

::

    conda install -c conda-forge h5py=*=mpi* netcdf4=*=mpi*

Note that parallel HDF5 and NetCDF4 often are available as optimized modules on
supercomputers. Otherwise, see the respective packages for how to install
with support for MPI.

Test installation
-----------------

After installing (from source) it may be a good idea to run all the tests
located in the ``tests`` folder. A range of tests may be run using the
``runtests.sh`` script

::

    conda install scipy, coverage
    cd tests/
    ./runtests.sh

This test-suit is run automatically on every commit to github, see, e.g.,

.. image:: https://circleci.com/bb/mpi4py/mpi4py-fft.svg?style=svg
    :target: https://circleci.com/bb/mpi4py/mpi4py-fft


.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _cython: http://cython.org
.. _spectralDNS channel: https://anaconda.org/spectralDNS
.. _conda: https://conda.io/docs/
.. _conda-forge: https://conda-forge.org
.. _FFTW: http://www.fftw.org
.. _pip: https://pypi.org/project/pip/
.. _HDF5: https://www.hdfgroup.org
.. _netCDF4: http://unidata.github.io/netcdf4-python/
.. _h5py: https://www.h5py.org
.. _mpich: https://www.mpich.org
.. _openmpi: https://www.open-mpi.org
.. _numpy: https://www.numpy.org
.. _numba: https://www.numba.org
.. _conda-build: https://conda.io/docs/commands/build/conda-build.html
.. _pypi: https://pypi.org/project/shenfun/

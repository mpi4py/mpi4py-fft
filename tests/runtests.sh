#!/bin/sh
set -e

PY=$(python -c 'import sys; print(sys.version_info.major)')
export OMPI_MCA_plm=isolated
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_rmaps_base_oversubscribe=yes

if [ $PY -eq 3 ]; then
    # coverage only for python version 3

    python -m coverage erase

    python -m coverage run -m test_fftw
    python -m coverage run -m test_libfft
    python -m coverage run -m test_io
    python -m coverage run -m test_darray
    mpiexec -n  2 python -m coverage run -m test_pencil

    #mpiexec -n  4 python -m coverage test_pencil.py
    #mpiexec -n  8 python -m coverage test_pencil.py
    mpiexec -n  2 python -m coverage run -m test_mpifft
    #mpiexec -n  4 python -m coverage test_mpifft.py
    # mpiexec -n  8 python -m coverage test_mpifft.py
    # mpiexec -n 12 python -m coverage test_mpifft.py
    mpiexec -n 2 python -m coverage run spectral_dns_solver.py
    mpiexec -n 2 python -m coverage run -m test_io
    mpiexec -n 4 python -m coverage run -m test_io
    mpiexec -n 2 python -m coverage run -m test_darray
    mpiexec -n 4 python -m coverage run -m test_darray

    python -m coverage combine

else
    python test_fftw.py
    python test_libfft.py
    mpiexec -n  2 python test_pencil.py
    #mpiexec -n  4 python test_pencil.py
    #mpiexec -n  8 python test_pencil.py
    mpiexec -n  2 python test_mpifft.py
    #mpiexec -n  4 python test_mpifft.py
    # mpiexec -n  8 python test_mpifft.py
    # mpiexec -n 12 python test_mpifft.py
    mpiexec -n 2 python test_io.py
    mpiexec -n 2 python test_darray.py
    mpiexec -n 2 python spectral_dns_solver.py
fi

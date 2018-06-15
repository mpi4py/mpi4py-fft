#!/bin/sh
set -e

python -m coverage erase

python -m coverage run -m test_fftw
python -m coverage run -m test_libfft

mpiexec -n  2 python -m coverage run -m test_pencil
#mpiexec -n  4 python -m coverage test_pencil.py
#mpiexec -n  8 python -m coverage test_pencil.py

mpiexec -n  2 python -m coverage run -m test_mpifft
#mpiexec -n  4 python -m coverage test_mpifft.py
# mpiexec -n  8 python -m coverage test_mpifft.py
# mpiexec -n 12 python -m coverage test_mpifft.py

mpiexec -n 2 python -m coverage run spectral_dns_solver.py

python -m coverage combine

#/bin/sh
set -e

python test_libfft.py

mpiexec -n  2 python test_pencil.py
mpiexec -n  4 python test_pencil.py
# mpiexec -n  8 python test_pencil.py

mpiexec -n  2 python test_mpifft.py
mpiexec -n  4 python test_mpifft.py
# mpiexec -n  8 python test_mpifft.py
# mpiexec -n 12 python test_mpifft.py

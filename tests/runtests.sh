#!/bin/sh
set -e

export OMPI_MCA_plm_ssh_agent=false
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_mpi_yield_when_idle=true
export OMPI_MCA_btl_base_warn_component_unused=false
export OMPI_MCA_rmaps_base_oversubscribe=true
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe

set -x

python -m coverage erase

python -m coverage run -m test_fftw
python -m coverage run -m test_libfft
python -m coverage run -m test_io
python -m coverage run -m test_darray

mpiexec -n  2 python -m coverage run -m test_pencil
mpiexec -n  4 python -m coverage run -m test_pencil
#mpiexec -n  8 python -m coverage test_pencil.py

mpiexec -n  2 python -m coverage run -m test_mpifft
mpiexec -n  4 python -m coverage run -m test_mpifft
#mpiexec -n  8 python -m coverage test_mpifft.py
#mpiexec -n 12 python -m coverage test_mpifft.py

mpiexec -n 2 python -m coverage run -m test_io
mpiexec -n 4 python -m coverage run -m test_io

mpiexec -n 2 python -m coverage run -m test_darray
mpiexec -n 4 python -m coverage run -m test_darray

mpiexec -n 2 python -m coverage run spectral_dns_solver.py

python -m coverage combine
python -m coverage xml

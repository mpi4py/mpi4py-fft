__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-03-09"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT, Function
from time import time


# Set global size of the computational box
M = 4
N = array([2**M, 2**(M+1), 2**(M+2)], dtype=int)

fft = PFFT(MPI.COMM_WORLD, N, collapse=False)
pfft = PFFT(MPI.COMM_WORLD, N, padding=[1., 1.5, 1.])

u = Function(fft)
u[:] = random.random(u.shape).astype(u.dtype)

u_hat = Function(fft, False)
u_hat = fft.forward(u, u_hat)
uj = zeros_like(u)
uj = fft.backward(u_hat, uj)
assert allclose(uj, u)

u_padded = zeros(pfft.forward.input_array.shape)
u_padded[:] = random.random(u_padded.shape)
uc = u_padded.copy()
u_hat.fill(0)
u_hat = pfft.forward(u_padded, u_hat)
u_padded.fill(0)
u_padded = pfft.backward(u_hat, u_padded)
assert allclose(u_padded, uc)

cfft = PFFT(MPI.COMM_WORLD, N, dtype=complex, padding=[1., 1., 1.5])

uc = random.random(cfft.backward.input_array.shape).astype(complex)
u2 = cfft.backward(uc)
u3 = uc.copy()
u3 = cfft.forward(u2, u3)

assert allclose(uc, u3)

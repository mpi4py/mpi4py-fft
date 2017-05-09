from numpy import *
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT, Function
from time import time


# Set global size of the computational box
M = 4
N = array([25, 28, 23], dtype=int)

fft = PFFT(MPI.COMM_WORLD, N, axes=(0,1,2), collapse=False, slab=True)
pfft = PFFT(MPI.COMM_WORLD, N, axes=(0,1,2), padding=[1.5, 1.5, 1.5], slab=True)

u = Function(fft, False)
u[:] = random.random(u.shape).astype(u.dtype)

u_hat = Function(fft)
u_hat = fft.forward(u, u_hat)
uj = zeros_like(u)
uj = fft.backward(u_hat, uj)
assert allclose(uj, u)

u_padded = zeros(pfft.forward.input_array.shape)
uc = u_hat.copy()
u_padded = pfft.backward(u_hat, u_padded)
u_hat = pfft.forward(u_padded, u_hat)
assert allclose(u_hat, uc)

cfft = PFFT(MPI.COMM_WORLD, N, dtype=complex, padding=[1.5, 1.5, 1.5])

uc = random.random(cfft.backward.input_array.shape).astype(complex)
u2 = cfft.backward(uc)
u3 = uc.copy()
u3 = cfft.forward(u2, u3)

assert allclose(uc, u3)



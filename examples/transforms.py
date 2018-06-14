from time import time
import numpy as np
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT, Function


# Set global size of the computational box
N = np.array([4, 4, 4], dtype=int)

fft = PFFT(MPI.COMM_WORLD, N, axes=(0,1,2), collapse=False, slab=True)
pfft = PFFT(MPI.COMM_WORLD, N, axes=(0,1,2), padding=[1.5, 1.5, 1.5], slab=True)

u = Function(fft, False)
u[:] = np.random.random(u.shape).astype(u.dtype)

u_hat = Function(fft)
u_hat = fft.forward(u, u_hat)
uj = np.zeros_like(u)
uj = fft.backward(u_hat, uj)
assert np.allclose(uj, u)

u_padded = np.zeros(pfft.forward.input_array.shape)
uc = u_hat.copy()
u_padded = pfft.backward(u_hat, u_padded)
u_hat = pfft.forward(u_padded, u_hat)
assert np.allclose(u_hat, uc)

#cfft = PFFT(MPI.COMM_WORLD, N, dtype=complex, padding=[1.5, 1.5, 1.5])
cfft = PFFT(MPI.COMM_WORLD, N, dtype=complex)

uc = np.random.random(cfft.backward.input_array.shape).astype(complex)
u2 = cfft.backward(uc)
u3 = uc.copy()
u3 = cfft.forward(u2, u3)

assert np.allclose(uc, u3)



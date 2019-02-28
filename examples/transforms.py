import functools
import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, DistArray
from mpi4py_fft.fftw import dctn, idctn

# Set global size of the computational box
N = np.array([18, 18, 18], dtype=int)

dct = functools.partial(dctn, type=3)
idct = functools.partial(idctn, type=3)

transforms = {(1, 2): (dct, idct)}

fft = PFFT(MPI.COMM_WORLD, N, axes=None, collapse=True, slab=True, transforms=transforms)
pfft = PFFT(MPI.COMM_WORLD, N, axes=((0,), (1, 2)), slab=True, padding=[1.5, 1.0, 1.0], transforms=transforms)

assert fft.axes == pfft.axes

u = DistArray(pfft=fft, forward_output=False)
u[:] = np.random.random(u.shape).astype(u.dtype)

u_hat = DistArray(pfft=fft, forward_output=True)
u_hat = fft.forward(u, u_hat)
uj = np.zeros_like(u)
uj = fft.backward(u_hat, uj)
assert np.allclose(uj, u)

u_padded = DistArray(pfft=pfft, forward_output=False)
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

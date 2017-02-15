__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-03-09"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT
from time import time


# Set global size of the computational box
M = 4
N = array([2**M, 2**(M+1)+2, 2**(M+2)], dtype=int)

fft = PFFT(MPI.COMM_WORLD, N)
pfft = PFFT(MPI.COMM_WORLD, N, padding=True)

u = random.random(fft.forward.input_array.shape)

u_hat = fft.forward(u)
u_padded = pfft.backward(u_hat)
u_hatc = u_hat.copy()
u_hatc = pfft.forward(u_padded)

assert allclose(u_hat, u_hatc)

# Just complex to complex
ffft = PFFT(MPI.COMM_WORLD, N, dtype=complex, padding=False)
cfft = PFFT(MPI.COMM_WORLD, N, dtype=complex, padding=True)
uc = random.random(cfft.backward.input_array.shape) + random.random(cfft.backward.input_array.shape)*1j

u2 = cfft.backward(uc)
f2 = ffft.backward(uc)
u2 = cfft.backward(uc)
f2 = ffft.backward(uc)

u3 = cfft.forward(u2)
f3 = ffft.forward(f2)

assert allclose(f3, u3)

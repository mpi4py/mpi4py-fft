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
N = array([2**M, 2**(M+1), 2**(M+2)], dtype=int)

fft = PFFT(MPI.COMM_WORLD, N)
pfft = PFFT(MPI.COMM_WORLD, N, padding=True)
#print(fft.axes, fft.forward._xfftn[1])

u = random.random(fft.forward.input_array.shape)
#MPI.COMM_WORLD.barrier()

#t0 = time()
u_hat = fft.forward(u)
#u_hat[N[0]//2] = 0
#u_hat[:, N[1]//2] = 0
u_hat[:, :, 0] = 0
u_padded = pfft.backward(u_hat)
u_hatc = u_hat.copy()
u_hatc = pfft.forward(u_padded)

assert allclose(u_hat, u_hatc)

cfft = PFFT(MPI.COMM_WORLD, N, dtype=complex, padding=True)

uc = random.random(cfft.backward.input_array.shape).astype(complex)
u2 = cfft.backward(uc)
u3 = cfft.forward(u2)

assert allclose(uc, u3)

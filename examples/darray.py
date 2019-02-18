import numpy as np
from mpi4py import MPI
from mpi4py_fft.pencil import Subcomm
from mpi4py_fft.distributedarray import DistributedArray
from mpi4py_fft.mpifft import PFFT, Function

# Test DistributedArray. Start with alignment in axis 0, then tranfer to 1 and
# finally to 2
N = (16, 14, 12)
z0 = DistributedArray(N, dtype=np.float, alignment=0)
z0[:] = np.random.randint(0, 10, z0.shape)
s0 = MPI.COMM_WORLD.allreduce(np.sum(z0))
print(MPI.COMM_WORLD.Get_rank(), z0.shape)
z1 = z0.redistribute(1)
s1 = MPI.COMM_WORLD.allreduce(np.sum(z1))
print(MPI.COMM_WORLD.Get_rank(), z1.shape)
z2 = z1.redistribute(2)
print(MPI.COMM_WORLD.Get_rank(), z2.shape)

s2 = MPI.COMM_WORLD.allreduce(np.sum(z2))
assert s0 == s1 == s2

fft = PFFT(Subcomm(MPI.COMM_WORLD, [s.Get_size() for s in z2.p0.subcomm]), N, dtype=z2.dtype)
z3 = Function(fft, True)
fft.forward(z2, z3)
import numpy as np
from mpi4py import MPI
from mpi4py_fft.pencil import Subcomm
from mpi4py_fft.distributedarray import DistributedArray, getDarray
from mpi4py_fft.mpifft import PFFT

# Test DistributedArray. Start with alignment in axis 0, then tranfer to 1 and
# finally to 2
N = (16, 14, 12)
z0 = DistributedArray(N, dtype=np.float, alignment=0)
z0[:] = np.random.randint(0, 10, z0.shape)
s0 = MPI.COMM_WORLD.allreduce(np.sum(z0))
z1 = z0.redistribute(2)
s1 = MPI.COMM_WORLD.allreduce(np.sum(z1))
z2 = z1.redistribute(1)
s2 = MPI.COMM_WORLD.allreduce(np.sum(z2))
assert s0 == s1 == s2

fft = PFFT(MPI.COMM_WORLD, darray=z2, axes=(0, 2, 1))
z3 = getDarray(fft, forward_output=True)
z2c = z2.copy()
fft.forward(z2, z3)
fft.backward(z3, z2)
s0, s1 = np.linalg.norm(z2), np.linalg.norm(z2c)
assert abs(s0-s1) < 1e-12, s0-s1

print(z3.local_slice(), z3.substart, z3.commsizes)

v0 = getDarray(fft, forward_output=False, rank=1)
v0[:] = np.random.random(v0.shape)
v0c = v0.copy()
v1 = getDarray(fft, forward_output=True, rank=1)

for i in range(3):
    v1[i] = fft.forward(v0[i], v1[i])
for i in range(3):
    v0[i] = fft.backward(v1[i], v0[i])
s0, s1 = np.linalg.norm(v0c), np.linalg.norm(v0)
assert abs(s0-s1) < 1e-12

print(v0.substart, v0.commsizes)

nfft = PFFT(MPI.COMM_WORLD, darray=v0[0], axes=(0, 2, 1))
for i in range(3):
    v1[i] = nfft.forward(v0[i], v1[i])
for i in range(3):
    v0[i] = nfft.backward(v1[i], v0[i])
s0, s1 = np.linalg.norm(v0c), np.linalg.norm(v0)
assert abs(s0-s1) < 1e-12

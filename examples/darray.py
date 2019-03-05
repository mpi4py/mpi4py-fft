import numpy as np
from mpi4py import MPI
from mpi4py_fft.pencil import Subcomm
from mpi4py_fft.distarray import DistArray, newDistArray, Function
from mpi4py_fft.mpifft import PFFT

# Test DistArray. Start with alignment in axis 0, then tranfer to 2 and
# finally to 1
N = (16, 14, 12)
z0 = DistArray(N, dtype=np.float, alignment=0)
z0[:] = np.random.randint(0, 10, z0.shape)
s0 = MPI.COMM_WORLD.allreduce(np.sum(z0))
z1 = z0.redistribute(2)
s1 = MPI.COMM_WORLD.allreduce(np.sum(z1))
z2 = z1.redistribute(1)
s2 = MPI.COMM_WORLD.allreduce(np.sum(z2))
assert s0 == s1 == s2

fft = PFFT(MPI.COMM_WORLD, darray=z2, axes=(0, 2, 1))
z3 = newDistArray(fft, forward_output=True)
z2c = z2.copy()
fft.forward(z2, z3)
fft.backward(z3, z2)
s0, s1 = np.linalg.norm(z2), np.linalg.norm(z2c)
assert abs(s0-s1) < 1e-12, s0-s1

v0 = newDistArray(fft, forward_output=False, rank=1)
#v0 = Function(fft, forward_output=False, rank=1)
v0[:] = np.random.random(v0.shape)
v0c = v0.copy()
v1 = newDistArray(fft, forward_output=True, rank=1)

for i in range(3):
    v1[i] = fft.forward(v0[i], v1[i])
for i in range(3):
    v0[i] = fft.backward(v1[i], v0[i])
s0, s1 = np.linalg.norm(v0c), np.linalg.norm(v0)
assert abs(s0-s1) < 1e-12

nfft = PFFT(MPI.COMM_WORLD, darray=v0[0], axes=(0, 2, 1))
for i in range(3):
    v1[i] = nfft.forward(v0[i], v1[i])
for i in range(3):
    v0[i] = nfft.backward(v1[i], v0[i])
s0, s1 = np.linalg.norm(v0c), np.linalg.norm(v0)
assert abs(s0-s1) < 1e-12

N = (6, 6, 6)
z = DistArray(N, dtype=float, alignment=0)
z[:] = MPI.COMM_WORLD.Get_rank()
g0 = z.get((0, slice(None), 0))
z2 = z.redistribute(2)
z = z2.redistribute(darray=z)
g1 = z.get((0, slice(None), 0))
assert np.all(g0 == g1)
s0 = MPI.COMM_WORLD.reduce(np.linalg.norm(z)**2)
s1 = MPI.COMM_WORLD.reduce(np.linalg.norm(z2)**2)
if MPI.COMM_WORLD.Get_rank() == 0:
    assert abs(s0-s1) < 1e-12

N = (3, 3, 6, 6, 6)
z2 = DistArray(N, dtype=float, val=1, alignment=2, rank=2)
z2[:] = MPI.COMM_WORLD.Get_rank()
z1 = z2.redistribute(1)
z0 = z1.redistribute(0)

s0 = MPI.COMM_WORLD.reduce(np.linalg.norm(z2)**2)
s1 = MPI.COMM_WORLD.reduce(np.linalg.norm(z0)**2)
if MPI.COMM_WORLD.Get_rank() == 0:
    assert abs(s0-s1) < 1e-12

z1 = z0.redistribute(darray=z1)
z0 = z1.redistribute(darray=z0)

N = (6, 6, 6, 6, 6)
m0 = DistArray(N, dtype=float, alignment=2)
m0[:] = MPI.COMM_WORLD.Get_rank()
m1 = m0.redistribute(4)
m0 = m1.redistribute(darray=m0)
s0 = MPI.COMM_WORLD.reduce(np.linalg.norm(m0)**2)
s1 = MPI.COMM_WORLD.reduce(np.linalg.norm(m1)**2)
if MPI.COMM_WORLD.Get_rank() == 0:
    assert abs(s0-s1) < 1e-12

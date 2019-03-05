import numpy as np
from mpi4py import MPI
from mpi4py_fft import DistArray, newDistArray, PFFT
from mpi4py_fft.pencil import Subcomm

comm = MPI.COMM_WORLD

def test_2Darray():
    N = (8, 8)
    for subcomm in ((0, 1), (1, 0), None, Subcomm(comm, (0, 1))):
        for rank in (0, 1, 2):
            M = (2,)*rank + N
            alignment = None
            if subcomm is None and rank == 1:
                alignment = 1
            a = DistArray(M, subcomm=subcomm, val=1, rank=rank, alignment=alignment)
            assert a.rank == rank
            assert a.global_shape == M
            s = a.substart
            c = a.subcomm
            z = a.commsizes
            p = a.pencil
            assert np.prod(np.array(z)) == comm.Get_size()
            if rank > 0:
                a0 = a[0]
                assert isinstance(a0, DistArray)
                assert a0.rank == rank-1
            aa = a.v
            assert isinstance(aa, np.ndarray)
            k = a.get((0,)*rank+(0, slice(None)))
            if comm.Get_rank() == 0:
                assert len(k) == N[1]
                assert np.sum(k) == N[1]
            k = a.get((0,)*rank+(slice(None), 0))
            if comm.Get_rank() == 0:
                assert len(k) == N[0]
                assert np.sum(k) == N[0]
            ls = a.local_slice()
            newaxis = (a.alignment+1)%2
            p0, t = a.get_pencil_and_transfer(newaxis)
            a[:] = MPI.COMM_WORLD.Get_rank()
            b = a.redistribute(newaxis)
            a = b.redistribute(darray=a)
            s0 = MPI.COMM_WORLD.reduce(np.linalg.norm(a)**2)
            s1 = MPI.COMM_WORLD.reduce(np.linalg.norm(b)**2)
            if MPI.COMM_WORLD.Get_rank() == 0:
                assert abs(s0-s1) < 1e-1
            c = a.redistribute(a.alignment)
            assert c is a

def test_3Darray():
    N = (8, 8, 8)
    for subcomm in ((0, 0, 1), (0, 1, 0),  (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), None, Subcomm(comm, (0, 0, 1))):
        for rank in (0, 1, 2):
            M = (3,)*rank + N
            alignment = None
            if subcomm is None and rank == 1:
                alignment = 2
            a = DistArray(M, subcomm=subcomm, val=1, rank=rank, alignment=alignment)
            assert a.rank == rank
            assert a.global_shape == M
            s = a.substart
            c = a.subcomm
            z = a.commsizes
            p = a.pencil
            assert np.prod(np.array(z)) == comm.Get_size()
            if rank > 0:
                a0 = a[0]
                assert isinstance(a0, DistArray)
                assert a0.rank == rank-1
            if rank == 2:
                a0 = a[0, 1]
                assert isinstance(a0, DistArray)
                assert a0.rank == 0
            aa = a.v
            assert isinstance(aa, np.ndarray)
            k = a.get((0,)*rank+(0, 0, slice(None)))
            if comm.Get_rank() == 0:
                assert len(k) == N[2]
                assert np.sum(k) == N[2]
            k = a.get((0,)*rank+(slice(None), 0, 0))
            if comm.Get_rank() == 0:
                assert len(k) == N[0]
                assert np.sum(k) == N[0]
            ls = a.local_slice()
            newaxis = (a.alignment+1)%3
            p0, t = a.get_pencil_and_transfer(newaxis)
            a[:] = MPI.COMM_WORLD.Get_rank()
            b = a.redistribute(newaxis)
            a = b.redistribute(darray=a)
            s0 = MPI.COMM_WORLD.reduce(np.linalg.norm(a)**2)
            s1 = MPI.COMM_WORLD.reduce(np.linalg.norm(b)**2)
            if MPI.COMM_WORLD.Get_rank() == 0:
                assert abs(s0-s1) < 1e-1

def test_newDistArray():
    N = (8, 8, 8)
    pfft = PFFT(MPI.COMM_WORLD, N)
    for forward_output in (True, False):
        for view in (True, False):
            for rank in (0, 1, 2):
                a = newDistArray(pfft, forward_output=forward_output,
                              rank=rank, view=view)
                if view is False:
                    assert isinstance(a, DistArray)
                    assert a.rank == rank
                    if rank == 0:
                        qfft = PFFT(MPI.COMM_WORLD, darray=a)
                    elif rank == 1:
                        qfft = PFFT(MPI.COMM_WORLD, darray=a[0])
                else:
                    assert isinstance(a, np.ndarray)
                    assert a.base.rank == rank

if __name__ == '__main__':
    test_2Darray()
    test_3Darray()
    test_newDistArray()

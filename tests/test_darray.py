import numpy as np
from mpi4py import MPI
from mpi4py_fft import DistArray, newDistArray, PFFT
from mpi4py_fft.pencil import Subcomm

comm = MPI.COMM_WORLD

def test_1Darray():
    N = (8,)
    z = DistArray(N, val=2)
    assert z[0] == 2
    assert z.shape == N

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
            _ = a.substart
            c = a.subcomm
            z = a.commsizes
            _ = a.pencil
            assert np.prod(np.array(z)) == comm.Get_size()
            if rank > 0:
                a0 = a[0]
                assert isinstance(a0, DistArray)
                assert a0.rank == rank-1
            aa = a.v
            assert isinstance(aa, np.ndarray)
            try:
                k = a.get((0,)*rank+(0, slice(None)))
                if comm.Get_rank() == 0:
                    assert len(k) == N[1]
                    assert np.sum(k) == N[1]
                k = a.get((0,)*rank+(slice(None), 0))
                if comm.Get_rank() == 0:
                    assert len(k) == N[0]
                    assert np.sum(k) == N[0]
            except ModuleNotFoundError:
                pass
            _ = a.local_slice()
            newaxis = (a.alignment+1)%2
            p, t = a.get_pencil_and_transfer(newaxis)
            a[:] = MPI.COMM_WORLD.Get_rank()
            b = a.redistribute(newaxis)
            a = b.redistribute(out=a)
            a = b.redistribute(a.alignment, out=a)
            s0 = MPI.COMM_WORLD.reduce(np.linalg.norm(a)**2)
            s1 = MPI.COMM_WORLD.reduce(np.linalg.norm(b)**2)
            if MPI.COMM_WORLD.Get_rank() == 0:
                assert abs(s0-s1) < 1e-1
            c = a.redistribute(a.alignment)
            assert c is a
            t.destroy()

def test_3Darray():
    N = (8, 8, 8)
    for subcomm in ((0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), None, Subcomm(comm, (0, 0, 1))):
        for rank in (0, 1, 2):
            M = (3,)*rank + N
            alignment = None
            if subcomm is None and rank == 1:
                alignment = 2
            a = DistArray(M, subcomm=subcomm, val=1, rank=rank, alignment=alignment)
            assert a.rank == rank
            assert a.global_shape == M
            _ = a.substart
            _ = a.subcomm
            z = a.commsizes
            _ = a.pencil
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
            try:
                k = a.get((0,)*rank+(0, 0, slice(None)))
                if comm.Get_rank() == 0:
                    assert len(k) == N[2]
                    assert np.sum(k) == N[2]
                k = a.get((0,)*rank+(slice(None), 0, 0))
                if comm.Get_rank() == 0:
                    assert len(k) == N[0]
                    assert np.sum(k) == N[0]
            except ModuleNotFoundError:
                pass
            _ = a.local_slice()
            newaxis = (a.alignment+1)%3
            p, t = a.get_pencil_and_transfer(newaxis)
            a[:] = MPI.COMM_WORLD.Get_rank()
            b = a.redistribute(newaxis)
            a = b.redistribute(out=a)
            s0 = MPI.COMM_WORLD.reduce(np.linalg.norm(a)**2)
            s1 = MPI.COMM_WORLD.reduce(np.linalg.norm(b)**2)
            if MPI.COMM_WORLD.Get_rank() == 0:
                assert abs(s0-s1) < 1e-1
            t.destroy()

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
                        qfft = PFFT(MPI.COMM_WORLD, darray=a[0, 0])
                    qfft.destroy()

                else:
                    assert isinstance(a, np.ndarray)
                    assert a.base.rank == rank
    pfft.destroy()

def test_redistribute_out():
    shape = (max(8, 2*comm.Get_size()), max(7, comm.Get_size()), 5)
    same_layout = Subcomm(comm, (0, 1, 1))

    def coordinate_values(array):
        starts = [part.start for part in array.local_slice()]
        i, j, k = np.indices(array.shape, dtype=np.int64)
        return ((i + starts[0])*shape[1]*shape[2] +
                (j + starts[1])*shape[2] + k + starts[2])

    source = DistArray(shape, subcomm=same_layout, alignment=1)
    source.v[...] = coordinate_values(source)
    source_values = coordinate_values(source)

    # Supplying out must still copy when the requested alignment already
    # matches the source alignment.
    same_alignment_out = DistArray(shape, subcomm=same_layout, alignment=1)
    same_alignment_out.v.fill(-1)
    returned = source.redistribute(axis=1, out=same_alignment_out)
    assert returned is same_alignment_out
    np.testing.assert_array_equal(same_alignment_out.v, source_values)
    np.testing.assert_array_equal(source.v, source_values)

    # Both axes are undivided, so this path needs no communication, but it
    # must not change the source pencil or ignore the requested output.
    undivided_out = DistArray(shape, subcomm=same_layout, alignment=2)
    undivided_out.v.fill(-1)
    source_alignment = source.alignment
    returned = source.redistribute(axis=2, out=undivided_out)
    assert returned is undivided_out
    assert source.alignment == source_alignment
    np.testing.assert_array_equal(undivided_out.v, source_values)
    np.testing.assert_array_equal(source.v, source_values)

    # axis=None takes the target alignment and performs a real redistribution.
    for axis in (None, 0):
        target_pencil = source.pencil.pencil(0)
        redistributed_out = DistArray(shape, subcomm=target_pencil.subcomm,
                                      alignment=0)
        redistributed_out.v.fill(-1)
        returned = source.redistribute(axis=axis, out=redistributed_out)
        assert returned is redistributed_out
        np.testing.assert_array_equal(redistributed_out.v,
                                      coordinate_values(redistributed_out))
        np.testing.assert_array_equal(source.v, source_values)

    # out=self remains valid and preserves the source data.
    returned = source.redistribute(axis=source.alignment, out=source)
    assert returned is source
    np.testing.assert_array_equal(source.v, source_values)

    same_layout.destroy()

if __name__ == '__main__':
    test_1Darray()
    test_2Darray()
    test_3Darray()
    test_newDistArray()
    test_redistribute_out()

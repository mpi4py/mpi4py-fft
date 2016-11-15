import numpy as np
from mpi4py import MPI


def subsize(N, size, rank):
    return N // size + (N % size > rank)


def distribution(N, size):
    q = N // size
    r = N % size
    n = s = i = 0
    while i < size:
        n = q
        s = q * i
        if i < r:
            n += 1
            s += i
        else:
            s += r
        yield (n, s)
        i += 1


def subarrays(comm, subshape, shape, axis, dtype):
    N = shape[axis]
    p = comm.Get_size()
    datatype = MPI._typedict[dtype.char]
    sizes = list(subshape)
    subsizes = sizes[:]
    substarts = [0] * len(sizes)
    subarrays = []
    for n, s in distribution(N, p):
        subsizes[axis] = n
        substarts[axis] = s
        newtype = datatype.Create_subarray(sizes, subsizes, substarts)
        newtype.Commit()
        subarrays.append(newtype)
    return tuple(subarrays)


class Subcomm(tuple):

    def __new__(cls, comm, dims=None, reorder=True):
        assert not comm.Is_inter()
        if comm.Get_topology() == MPI.CART:
            assert comm.Get_dim() > 0
            assert dims is None
            cartcomm = comm
        else:
            if dims is None:
                dims = [0, 0]
            elif np.ndim(dims) > 0:
                assert len(dims) > 0
                dims = [max(0, d) for d in dims]
            else:
                assert dims > 0
                dims = [0] * dims
            dims = MPI.Compute_dims(comm.Get_size(), dims)
            cartcomm = comm.Create_cart(dims, reorder=reorder)

        dim = cartcomm.Get_dim()
        subcomm = [None] * dim
        for i in range(dim):
            remdims = [False] * dim
            remdims[i] = True
            subcomm[i] = cartcomm.Sub(remdims)

        if cartcomm != comm:
            cartcomm.Free()

        return super(Subcomm, cls).__new__(cls, subcomm)

    def destroy(self):
        for comm in self:
            if comm:
                comm.Free()


class Transfer(object):

    def __init__(self,
                 comm, shape, dtype,
                 subshapeA, axisA,
                 subshapeB, axisB):
        self.comm = comm
        self.shape = tuple(shape)
        self.dtype = dtype = np.dtype(dtype)
        self.subshapeA, self.axisA = tuple(subshapeA), axisA
        self.subshapeB, self.axisB = tuple(subshapeB), axisB
        self._subarraysA = subarrays(comm, subshapeA, shape, axisB, dtype)
        self._subarraysB = subarrays(comm, subshapeB, shape, axisA, dtype)
        size = comm.Get_size()
        self._counts_displs = ([1] * size, [0] * size) # XXX (None, None)

    def forward(self, A, B):
        assert self.subshapeA == A.shape
        assert self.subshapeB == B.shape
        assert self.dtype == A.dtype
        assert self.dtype == B.dtype
        self.comm.Alltoallw([A, self._counts_displs, self._subarraysA],
                            [B, self._counts_displs, self._subarraysB])

    def backward(self, B, A):
        assert self.subshapeA == A.shape
        assert self.subshapeB == B.shape
        assert self.dtype == A.dtype
        assert self.dtype == B.dtype
        self.comm.Alltoallw([B, self._counts_displs, self._subarraysB],
                            [A, self._counts_displs, self._subarraysA])

    def destroy(self):
        for datatype in self._subarraysA:
            if datatype:
                datatype.Free()
        for datatype in self._subarraysB:
            if datatype:
                datatype.Free()


class Pencil(object):

    def __init__(self, subcomm, shape, axis=-1):
        assert len(shape) >= 2
        assert min(shape) >= 1
        assert -len(shape) <= axis < len(shape)
        assert 1 <= len(subcomm) <= len(shape)

        if axis < 0:
            axis += len(shape)
        if len(subcomm) < len(shape):
            subcomm = list(subcomm)
            while len(subcomm) < len(shape) - 1:
                subcomm.append(MPI.COMM_SELF)
            subcomm.insert(axis, MPI.COMM_SELF)
        else:
            assert subcomm[axis].Get_size() == 1

        subshape = [None] * len(shape)
        for i in range(len(shape)):
            comm = subcomm[i]
            size = comm.Get_size()
            rank = comm.Get_rank()
            assert shape[i] >= size
            subshape[i] = subsize(shape[i], size, rank)

        self.subcomm = tuple(subcomm)
        self.shape = tuple(shape)
        self.subshape = tuple(subshape)
        self.axis = axis

    def pencil(self, axis):
        assert -len(self.shape) <= axis < len(self.shape)
        if axis < 0:
            axis += len(self.shape)
        i, j = self.axis, axis
        subcomm = list(self.subcomm)
        subcomm[j], subcomm[i] = subcomm[i], subcomm[j]
        return Pencil(subcomm, self.shape, axis)

    def transfer(self, pencil, dtype):
        penA, penB = self, pencil
        assert penA.shape == penB.shape
        assert penA.axis != penB.axis
        for i in range(len(penA.shape)):
            if i != penA.axis and i != penB.axis:
                assert penA.subcomm[i] == penB.subcomm[i]
                assert penA.subshape[i] == penB.subshape[i]
        assert penA.subcomm[penB.axis] == penB.subcomm[penA.axis]

        axis = penB.axis
        comm = penA.subcomm[axis]
        shape = list(penA.subshape)
        shape[axis] = penA.shape[axis]
        return Transfer(comm, shape, dtype,
                        penA.subshape, penB.axis,
                        penB.subshape, penA.axis)

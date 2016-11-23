import numpy as np
from mpi4py import MPI


def _blockdist(N, size, rank):
    q, r = divmod(N, size)
    n = q + (1 if r > rank else 0)
    s = rank * q + min(rank, r)
    return (n, s)


def _subarraytypes(comm, shape, axis, subshape, dtype):
    # pylint: disable=too-many-locals
    # pylint: disable=protected-access
    N = shape[axis]
    p = comm.Get_size()
    datatype = MPI._typedict[dtype.char]
    sizes = list(subshape)
    subsizes = sizes[:]
    substarts = [0] * len(sizes)
    datatypes = []
    for i in range(p):
        n, s = _blockdist(N, p, i)
        subsizes[axis] = n
        substarts[axis] = s
        newtype = datatype.Create_subarray(
            sizes, subsizes, substarts).Commit()
        datatypes.append(newtype)
    return tuple(datatypes)


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
        remdims = [False] * dim
        for i in range(dim):
            remdims[i] = True
            subcomm[i] = cartcomm.Sub(remdims)
            remdims[i] = False

        if cartcomm != comm:
            cartcomm.Free()

        return super(Subcomm, cls).__new__(cls, subcomm)

    def destroy(self):
        for comm in self:
            if comm:
                comm.Free()


class Transfer(object):

    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 comm, shape, dtype,
                 subshapeA, axisA,
                 subshapeB, axisB):
        # pylint: disable=too-many-arguments
        self.comm = comm
        self.shape = tuple(shape)
        self.dtype = dtype = np.dtype(dtype)
        self.subshapeA, self.axisA = tuple(subshapeA), axisA
        self.subshapeB, self.axisB = tuple(subshapeB), axisB
        self._subtypesA = _subarraytypes(comm, shape, axisA, subshapeA, dtype)
        self._subtypesB = _subarraytypes(comm, shape, axisB, subshapeB, dtype)
        size = comm.Get_size()
        self._counts_displs = ([1] * size, [0] * size)  # XXX (None, None)

    def forward(self, arrayA, arrayB):
        assert self.subshapeA == arrayA.shape
        assert self.subshapeB == arrayB.shape
        assert self.dtype == arrayA.dtype
        assert self.dtype == arrayB.dtype
        self.comm.Alltoallw([arrayA, self._counts_displs, self._subtypesA],
                            [arrayB, self._counts_displs, self._subtypesB])

    def backward(self, arrayB, arrayA):
        assert self.subshapeA == arrayA.shape
        assert self.subshapeB == arrayB.shape
        assert self.dtype == arrayA.dtype
        assert self.dtype == arrayB.dtype
        self.comm.Alltoallw([arrayB, self._counts_displs, self._subtypesB],
                            [arrayA, self._counts_displs, self._subtypesA])

    def destroy(self):
        for datatype in self._subtypesA:
            if datatype:
                datatype.Free()
        for datatype in self._subtypesB:
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
        substart = [None] * len(shape)
        for i, comm in enumerate(subcomm):
            size = comm.Get_size()
            rank = comm.Get_rank()
            assert shape[i] >= size
            n, s = _blockdist(shape[i], size, rank)
            subshape[i] = n
            substart[i] = s

        self.shape = tuple(shape)
        self.axis = axis
        self.subcomm = tuple(subcomm)
        self.subshape = tuple(subshape)
        self.substart = tuple(substart)

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
                        penA.subshape, penA.axis,
                        penB.subshape, penB.axis)

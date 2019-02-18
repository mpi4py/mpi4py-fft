import numpy as np
from mpi4py import MPI
from .pencil import Pencil, Subcomm

comm = MPI.COMM_WORLD

class DistributedArray(np.ndarray):
    """Distributed Numpy array

    Parameters
    ----------
    global_shape : sequence of ints
        Shape of non-distributed global array
    subcomm : None, Subcomm instance or sequence of ints
        Describes how to distribute the array
    val : int
        Initialize array with this value if buffer is not given
    dtype : np.dtype
        Type of array
    buffer : np.ndarray
        Array of correct shape
    alignment : None or int
        Make sure array is aligned in this direction

    """
    def __new__(cls, global_shape, subcomm=None, val=0, dtype=np.float,
                buffer=None, alignment=None):
        if isinstance(subcomm, Subcomm):
            pass
        else:
            if isinstance(subcomm, (tuple, list)):
                assert len(subcomm) == len(global_shape)
                # Do nothing if already containing communicators. A tuple of subcommunicators is not necessarily a Subcomm
                if not np.all([isinstance(s, MPI.Cartcomm) for s in subcomm]):
                    subcomm = Subcomm(comm, subcomm)
            else:
                assert subcomm is None
                if alignment is not None:
                    subcomm = [0] * len(global_shape)
                    subcomm[alignment] = 1
                subcomm = Subcomm(comm, subcomm)

        sizes = [s.Get_size() for s in subcomm]
        if alignment is not None:
            assert isinstance(alignment, int)
            assert sizes[alignment] == 1
        else:
            # Decide that alignment is the last axis with size 1
            alignment = np.flatnonzero(np.array(sizes) == 1)[-1]
        p0 = Pencil(subcomm, global_shape, axis=alignment)
        obj = np.ndarray.__new__(cls, p0.subshape, dtype=dtype, buffer=buffer)
        if buffer is None:
            obj.fill(val)
        obj.p0 = p0
        obj.global_shape = global_shape
        return obj

    def alignment(self):
        return self.p0.axis

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.p0 = getattr(obj, 'p0', None)
        self.global_shape = getattr(obj, 'global_shape', None)

    def redistribute(self, axis):
        """Global redistribution of array into alignment in ``axis``

        Parameters
        ----------
        axis : int
            Align array along this axis

        Returns
        -------
        DistributedArray
            self array globally redistributed along new axis

        """
        p1 = self.p0.pencil(axis)
        transfer = self.p0.transfer(p1, self.dtype)
        z0 = np.zeros(p1.subshape, dtype=self.dtype)
        transfer.forward(self, z0)
        return DistributedArray(self.global_shape,
                                subcomm=p1.subcomm,
                                dtype=self.dtype,
                                alignment=axis,
                                buffer=z0)

from mpi4py_fft.distarray import DistArrayBase
import cupy as cp
from mpi4py import MPI
from numbers import Number

comm = MPI.COMM_WORLD


class DistArrayCuPy(DistArrayBase, cp.ndarray):
    """Distributed CuPy array

    This CuPy array is part of a larger global array. Information about the
    distribution is contained in the attributes.

    Parameters
    ----------
    global_shape : sequence of ints
        Shape of non-distributed global array
    subcomm : None, :class:`.Subcomm` object or sequence of ints, optional
        Describes how to distribute the array
    val : Number or None, optional
        Initialize array with this number if buffer is not given
    dtype : cp.dtype, optional
        Type of array
    memptr : cupy.cuda.MemoryPointer, optional
        Pointer to the array content head.
    alignment : None or int, optional
        Make sure array is aligned in this direction. Note that alignment does
        not take rank into consideration.
    rank : int, optional
        Rank of tensor (number of free indices, a scalar is zero, vector one,
        matrix two)


    For more information, see `cupy.ndarray <https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray>`_

    """

    xp = cp

    def __new__(
        cls,
        global_shape,
        subcomm=None,
        val=None,
        dtype=float,
        memptr=None,
        strides=None,
        alignment=None,
        rank=0,
    ):
        if len(global_shape[rank:]) < 2:  # 1D case
            obj = cls.xp.ndarray.__new__(
                cls, global_shape, dtype=dtype, memptr=memptr, strides=strides
            )
            if memptr is None and isinstance(val, Number):
                obj.fill(val)
            obj._rank = rank
            obj._p0 = None
            return obj

        subcomm = cls.get_subcomm(subcomm, global_shape, rank, alignment)
        p0, subshape = cls.setup_pencil(subcomm, rank, global_shape, alignment)

        obj = cls.xp.ndarray.__new__(cls, subshape, dtype=dtype, memptr=memptr, strides=strides)
        if memptr is None and isinstance(val, Number):
            obj.fill(val)
        obj._p0 = p0
        obj._rank = rank
        return obj

    def get(self, *args, **kwargs):
        """Untangle inheritance conflicts"""
        if any(isinstance(me, tuple) for me in args):
            return DistArrayBase.get(self, *args, **kwargs)
        else:
            return cp.ndarray.get(self, *args, **kwargs)

    def asnumpy(self):
        """Copy the array to CPU"""
        return self.get()

    @property
    def v(self):
        """Return local ``self`` array as an ``ndarray`` object"""
        return cp.ndarray.__getitem__(self, slice(None, None, None))

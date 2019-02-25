import numpy as np
from mpi4py import MPI
from .pencil import Pencil, Subcomm

comm = MPI.COMM_WORLD

class DistributedArray(np.ndarray):
    """Distributed Numpy array

    This Numpy array is part of a larger global array. Information about the
    distribution is contained in the attributes

    Parameters
    ----------
    global_shape : sequence of ints
        Shape of non-distributed global array
    subcomm : None, Subcomm instance or sequence of ints, optional
        Describes how to distribute the array
    val : int or None, optional
        Initialize array with this int if buffer is not given
    dtype : np.dtype, optional
        Type of array
    buffer : np.ndarray, optional
        Array of correct shape
    alignment : None or int, optional
        Make sure array is aligned in this direction. Note that alignment does
        not take rank into consideration.
    rank : int
        Rank of tensor (scalar is zero, vector one, matrix two)

    Note
    ----
    Tensors of rank higher than 0 are not distributed in the first ``rank``
    indices. For example, when creating a vector of distributed arrays of global
    shape (12, 14, 16) the returned DistributedArray will have global shape
    (3, 12, 14, 16), and it can only be distributed in the last three axes.
    Also note that the ``alignment`` keyword does not take rank into
    consideration. Setting alignment=2 for the array above means that the last
    axis (of length 16) will be aligned, also when rank>0.

    """
    def __new__(cls, global_shape, subcomm=None, val=None, dtype=np.float,
                buffer=None, alignment=None, rank=0):

        if isinstance(subcomm, Subcomm):
            pass
        else:
            if isinstance(subcomm, (tuple, list)):
                assert len(subcomm) == len(global_shape[rank:])
                # Do nothing if already containing communicators. A tuple of subcommunicators is not necessarily a Subcomm
                if not np.all([isinstance(s, MPI.Comm) for s in subcomm]):
                    subcomm = Subcomm(comm, subcomm)
            else:
                assert subcomm is None
                if alignment is not None:
                    subcomm = [0] * len(global_shape[rank:])
                    subcomm[alignment] = 1
                subcomm = Subcomm(comm, subcomm)
        sizes = [s.Get_size() for s in subcomm]
        if alignment is not None:
            assert isinstance(alignment, int)
            assert sizes[alignment] == 1
        else:
            # Decide that alignment is the last axis with size 1
            alignment = np.flatnonzero(np.array(sizes) == 1)[-1]
        p0 = Pencil(subcomm, global_shape[rank:], axis=alignment)
        subshape = p0.subshape
        if rank > 0:
            subshape = global_shape[:rank] + subshape
        obj = np.ndarray.__new__(cls, subshape, dtype=dtype, buffer=buffer)
        if buffer is None and isinstance(val, int):
            obj.fill(val)
        obj._p0 = p0
        obj._rank = rank
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._p0 = getattr(obj, '_p0', None)
        self._rank = getattr(obj, '_rank', None)

    @property
    def alignment(self):
        """Return alignment of DistributedArray"""
        return self._p0.axis

    @property
    def global_shape(self):
        """Return global shape of DistributedArray"""
        return self.shape[:self.rank] + self._p0.shape

    @property
    def substart(self):
        """Return starting indices of DistributedArray on this processor"""
        return (0,)*self.rank + self._p0.substart

    @property
    def subcomm(self):
        """Return tuple of subcommunicators for DistributedArray"""
        return (MPI.COMM_SELF,)*self.rank + self._p0.subcomm

    @property
    def commsizes(self):
        """Return number of processors along each axis"""
        return [s.Get_size() for s in self.subcomm]

    @property
    def pencil(self):
        """Return pencil describing distribution of DistributedArray"""
        return self._p0

    @property
    def rank(self):
        """Return rank of DistributedArray"""
        return self._rank

    def __getitem__(self, i):
        if isinstance(i, int) and self.rank > 0:
            v0 = np.ndarray.__getitem__(self, i)
            v0._rank -= 1
            return v0
        v0 = np.ndarray.__getitem__(self, i)
        if isinstance(v0, DistributedArray):
            v0._rank = 0
        return v0

    def local_slice(self):
        """Local view into global array

        Returns
        -------
        List of slices
            Each item of the returned list is the slice along that axis,
            describing the view of the current array into the global array.

        Example
        -------
        Print local_slice of a global array of shape (16, 14, 12) using 4
        processors.

        >>> import subprocess
        >>> fx = open('ls_script.py', 'w')
        >>> h = fx.write('''
        ... from mpi4py import MPI
        ... from mpi4py_fft.distributedarray import DistributedArray
        ... comm = MPI.COMM_WORLD
        ... N = (16, 14, 12)
        ... z = DistributedArray(N, dtype=float, alignment=0)
        ... ls = comm.gather(z.local_slice())
        ... if comm.Get_rank() == 0:
        ...     for l in ls:
        ...         print(l)''')
        >>> fx.close()
        >>> print(subprocess.getoutput('mpirun -np 4 python ls_script.py'))
        [slice(0, 16, None), slice(0, 7, None), slice(0, 6, None)]
        [slice(0, 16, None), slice(0, 7, None), slice(6, 12, None)]
        [slice(0, 16, None), slice(7, 14, None), slice(0, 6, None)]
        [slice(0, 16, None), slice(7, 14, None), slice(6, 12, None)]
        """
        v = [slice(start, start+shape) for start, shape in zip(self._p0.substart,
                                                               self._p0.subshape)]
        return [slice(0, s) for s in self.shape[:self.rank]] + v

    def get_pencil_and_transfer(self, axis):
        """Return new pencil and transfer object for alignment in ``axis``

        Parameters
        ----------
        axis : int
            The new axis to align data with

        Returns
        -------
        2-tuple
            2-tuple where first item is a :class:`.Pencil` aligned in ``axis``.
            Second item is a :class:`.Transfer` object for executing the
            redistribution of data
        """
        p1 = self._p0.pencil(axis)
        return p1, self._p0.transfer(p1, self.dtype)

    def redistribute(self, axis):
        """Global redistribution of array into alignment in ``axis``

        Parameters
        ----------
        axis : int
            Align array along this axis

        Returns
        -------
        DistributedArray
            New DistributedArray, globally redistributed along ``axis``
        """
        p1, transfer = self.get_pencil_and_transfer(axis)
        z0 = DistributedArray(self.global_shape,
                              subcomm=p1.subcomm,
                              dtype=self.dtype,
                              alignment=axis,
                              rank=self.rank)
        transfer.forward(self, z0)
        return z0

def getDarray(pfft, forward_output=True, val=0, rank=0):
    """Return DistributedArray for instance of PFFT class

    Parameters
    ----------
    pfft : Instance of :class:`.PFFT` class
    forward_output: boolean, optional
        If False then create getDarray of shape/type for input to
        forward transform, otherwise create getDarray of shape/type for
        output from forward transform.
    val : int or float
        Value used to initialize array.
    rank: int
        Scalar has rank 0, vector 1 and matrix 2

    For more information, see `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from mpi4py_fft import PFFT, getDarray
    >>> FFT = PFFT(MPI.COMM_WORLD, [64, 64, 64])
    >>> u = getDarray(FFT, False, rank=1)
    >>> u_hat = getDarray(FFT, True, rank=1)

    """
    if forward_output is True:
        shape = pfft.forward.output_array.shape
        dtype = pfft.forward.output_array.dtype
        p0 = pfft.pencil[1]
    else:
        shape = pfft.forward.input_array.shape
        dtype = pfft.forward.input_array.dtype
        p0 = pfft.pencil[0]
    commsizes = [s.Get_size() for s in p0.subcomm]
    global_shape = tuple([s*p for s, p in zip(shape, commsizes)])

    if rank == 1:
        global_shape = (len(shape),) + global_shape
    elif rank == 2:
        global_shape = (len(shape), len(shape)) + global_shape
    else:
        assert rank == 0

    return DistributedArray(global_shape, subcomm=p0.subcomm, val=val,
                            dtype=dtype, rank=rank)
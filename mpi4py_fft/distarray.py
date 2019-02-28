import os
from numbers import Number
import numpy as np
from mpi4py import MPI
from .pencil import Pencil, Subcomm

comm = MPI.COMM_WORLD

class DistArray(np.ndarray):
    """Distributed Numpy array

    This Numpy array is part of a larger global array. Information about the
    distribution is contained in the attributes

    Parameters
    ----------
    global_shape : sequence of ints
        Shape of non-distributed global array
    subcomm : None, Subcomm instance or sequence of ints, optional
        Describes how to distribute the array
    val : Number or None, optional
        Initialize array with this number if buffer is not given
    dtype : np.dtype, optional
        Type of array
    buffer : np.ndarray, optional
        Array of correct shape
    alignment : None or int, optional
        Make sure array is aligned in this direction. Note that alignment does
        not take rank into consideration.
    rank : int, optional
        Rank of tensor (scalar is zero, vector one, matrix two)


    For more information, see `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_

    Note
    ----
    Tensors of rank higher than 0 are not distributed in the first ``rank``
    indices. For example,

    >>> from mpi4py_fft import DistArray
    >>> a = DistArray((3, 8, 8, 8), rank=1)
    >>> print(a.pencil.shape)
    (8, 8, 8)

    The array ``a`` cannot be distributed in the first axis of length 3 since
    rank is 1 and this first index represent the vector component. The ``pencil``
    attribute of ``a`` thus only considers the last three axes.

    Also note that the ``alignment`` keyword does not take rank into
    consideration. Setting alignment=2 for the array above means that the last
    axis will be aligned, also when rank>0.

    """
    def __new__(cls, global_shape, subcomm=None, val=None, dtype=np.float,
                buffer=None, alignment=None, rank=0):
        if rank > 0:
            assert global_shape[:rank] == (len(global_shape[rank:]),)*rank

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
                subcomm = [0] * len(global_shape[rank:])
                if alignment is not None:
                    subcomm[alignment] = 1
                else:
                    subcomm[-1] = 1
                    alignment = len(subcomm)-1
                subcomm = Subcomm(comm, subcomm)
        sizes = [s.Get_size() for s in subcomm]
        if alignment is not None:
            assert isinstance(alignment, (int, np.integer))
            assert sizes[alignment] == 1
        else:
            # Decide that alignment is the last axis with size 1
            alignment = np.flatnonzero(np.array(sizes) == 1)[-1]
        p0 = Pencil(subcomm, global_shape[rank:], axis=alignment)
        subshape = p0.subshape
        if rank > 0:
            subshape = global_shape[:rank] + subshape
        obj = np.ndarray.__new__(cls, subshape, dtype=dtype, buffer=buffer)
        if buffer is None and isinstance(val, Number):
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
        """Return alignment of local ``self`` array"""
        return self._p0.axis

    @property
    def global_shape(self):
        """Return global shape of ``self``"""
        return self.shape[:self.rank] + self._p0.shape

    @property
    def substart(self):
        """Return starting indices of local ``self`` array"""
        return (0,)*self.rank + self._p0.substart

    @property
    def subcomm(self):
        """Return tuple of subcommunicators for all axes of ``self``"""
        return (MPI.COMM_SELF,)*self.rank + self._p0.subcomm

    @property
    def commsizes(self):
        """Return number of processors along each axis of ``self``"""
        return [s.Get_size() for s in self.subcomm]

    @property
    def pencil(self):
        """Return pencil describing distribution of ``self``"""
        return self._p0

    @property
    def rank(self):
        """Return rank of ``self``"""
        return self._rank

    def __getitem__(self, i):
        # Return DistArray if the result is a component of a tensor
        # Otherwise return ndarray view
        if isinstance(i, int) and self.rank > 0:
            v0 = np.ndarray.__getitem__(self, i)
            v0._rank -= 1
            return v0
        if isinstance(i, tuple) and len(i) == 2 and self.rank == 2:
            v0 = np.ndarray.__getitem__(self, i)
            v0._rank = 0
            return v0
        return np.ndarray.__getitem__(self.v, i)

    @property
    def v(self):
        """ Return local ``self`` array as an ``ndarray`` object"""
        return self.__array__()

    def get_global_slice(self, gslice):
        """Return global slice of ``self``

        Parameters
        ----------
        gslice : sequence of slice(None) and ints
            The slice of the global array.

        Returns
        -------
        Numpy array
            The slice of the global array is returned on rank 0, whereas the
            remaining ranks return None

        Example
        -------
        >>> import subprocess
        >>> fx = open('gs_script.py', 'w')
        >>> h = fx.write('''
        ... from mpi4py import MPI
        ... from mpi4py_fft.distarray import DistArray
        ... comm = MPI.COMM_WORLD
        ... N = (6, 6, 6)
        ... z = DistArray(N, dtype=float, alignment=0)
        ... z[:] = comm.Get_rank()
        ... g = z.get_global_slice((0, slice(None), 0))
        ... if comm.Get_rank() == 0:
        ...     print(g)''')
        >>> fx.close()
        >>> print(subprocess.getoutput('mpirun -np 4 python gs_script.py'))
        [0. 0. 0. 2. 2. 2.]
        """
        # Note that this implementation uses h5py to take care of the local to
        # global MPI. We create a global file with MPI, but then open it without
        # MPI and only on rank 0.
        import h5py
        f = h5py.File('tmp.h5', 'w', driver="mpio", comm=comm)
        s = self.local_slice()
        sp = np.nonzero([isinstance(x, slice) for x in gslice])[0]
        sf = tuple(np.take(s, sp))
        N = self.global_shape
        f.require_dataset('0', shape=tuple(np.take(N, sp)), dtype=self.dtype)
        gslice = list(gslice)
        si = np.nonzero([isinstance(x, int) and not z == slice(None) for x, z in zip(gslice, s)])[0]
        on_this_proc = True
        for i in si:
            if gslice[i] >= s[i].start and gslice[i] < s[i].stop:
                gslice[i] -= s[i].start
            else:
                on_this_proc = False
        if on_this_proc:
            f["0"][sf] = self[tuple(gslice)]
        f.close()
        c = None
        if comm.Get_rank() == 0:
            h = h5py.File('tmp.h5', 'r')
            c = h['0'].__array__()
            h.close()
            os.remove('tmp.h5')
        return c

    def local_slice(self):
        """Return local view into global ``self`` array

        Returns
        -------
        List of slices
            Each item of the returned list is the slice along that axis,
            describing the view of the ``self`` array into the global array.

        Example
        -------
        Print local_slice of a global array of shape (16, 14, 12) using 4
        processors.

        >>> import subprocess
        >>> fx = open('ls_script.py', 'w')
        >>> h = fx.write('''
        ... from mpi4py import MPI
        ... from mpi4py_fft.distarray import DistArray
        ... comm = MPI.COMM_WORLD
        ... N = (16, 14, 12)
        ... z = DistArray(N, dtype=float, alignment=0)
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
        """Return pencil and transfer objects for alignment along ``axis``

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

    def redistribute(self, axis=None, darray=None):
        """Global redistribution of local ``self`` array

        Parameters
        ----------
        axis : int, optional
            Align local ``self`` array along this axis
        darray : :class:`.DistArray`, optional
            Copy data to this array of possibly different alignment

        Returns
        -------
        :class:`.DistArray` : darray
            The ``self`` array globally redistributed. If keyword ``darray`` is
            None then a new DistArray (aligned along ``axis``) is created
            and returned
        """
        if axis is None:
            assert isinstance(darray, np.ndarray)
            assert self.global_shape == darray.global_shape
            axis = darray.alignment
        p1, transfer = self.get_pencil_and_transfer(axis)
        if darray is None:
            darray = DistArray(self.global_shape,
                               subcomm=p1.subcomm,
                               dtype=self.dtype,
                               alignment=axis,
                               rank=self.rank)
        if self.rank == 0:
            transfer.forward(self, darray)
        elif self.rank == 1:
            for i in range(self.shape[0]):
                transfer.forward(self[i], darray[i])
        elif self.rank == 2:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    transfer.forward(self[i, j], darray[i, j])

        return darray

def newDistArray(pfft, forward_output=True, val=0, rank=0, view=False):
    """Return a :class:`.DistArray` for provided :class:`.PFFT` object

    Parameters
    ----------
    pfft : :class:`.PFFT` object
    forward_output: boolean, optional
        If False then create newDistArray of shape/type for input to
        forward transform, otherwise create newDistArray of shape/type for
        output from forward transform.
    val : int or float
        Value used to initialize array.
    rank: int
        Scalar has rank 0, vector 1 and matrix 2
    view : bool
        If True return view of the underlying Numpy array, i.e., return
        cls.view(np.ndarray). Note that the DistArray still will
        be accessible through the base attribute of the view.

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from mpi4py_fft import PFFT, newDistArray
    >>> FFT = PFFT(MPI.COMM_WORLD, [64, 64, 64])
    >>> u = newDistArray(FFT, False, rank=1)
    >>> u_hat = newDistArray(FFT, True, rank=1)

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
    global_shape = (len(shape),)*rank + global_shape
    z = DistArray(global_shape, subcomm=p0.subcomm, val=val, dtype=dtype,
                  rank=rank)
    return z.v if view else z

def Function(*args, **kwargs): #pragma: no cover
    import warnings
    warnings.warn("Function() is deprecated; use newDistArray().", FutureWarning)
    if 'tensor' in kwargs:
        kwargs['rank'] = 1
        del kwargs['tensor']
    return newDistArray(*args, **kwargs)

import numpy as np

from .libfft import FFT
from .pencil import Pencil
from .pencil import Subcomm


class Transform(object):
    """Class for performing any parallel transform, forward or backward

    Parameters
    ----------
    xfftn : list of serial transform objects
    transfer : list of global redistribution objects
    pencil : list of two pencil objects
        The two pencils represent the input and final output configuration of
        the distributed global arrays

    """
    def __init__(self, xfftn, transfer, pencil):
        assert len(xfftn) == len(transfer) + 1 and len(pencil) == 2
        self._xfftn = tuple(xfftn)
        self._transfer = tuple(transfer)
        self._pencil = tuple(pencil)

    @property
    def input_array(self):
        """Return input array of Transform"""
        return self._xfftn[0].input_array

    @property
    def output_array(self):
        """Return output array of Transform"""
        return self._xfftn[-1].output_array

    @property
    def input_pencil(self):
        """Return input pencil of Transform"""
        return self._pencil[0]

    @property
    def output_pencil(self):
        """Return output pencil of Transform"""
        return self._pencil[1]

    def __call__(self, input_array=None, output_array=None, **kw):
        """Compute transform

        Parameters
        ----------
        input_array : array, optional
        output_array : array, optional
        kw : dict
            parameters to serial transforms
            Note in particular that the keyword 'normalize'=True/False can be
            used to turn normalization on or off. Default is to enable
            normalization for forward transforms and disable it for backward.

        Note
        ----
        If input_array/output_array are not given, then use predefined arrays
        as planned with serial transform object _xfftn.

        """
        if input_array is not None:
            self.input_array[...] = input_array

        for i in range(len(self._transfer)):
            self._xfftn[i](**kw)
            arrayA = self._xfftn[i].output_array
            arrayB = self._xfftn[i+1].input_array
            self._transfer[i](arrayA, arrayB)
        self._xfftn[-1](**kw)

        if output_array is not None:
            output_array[...] = self.output_array
            return output_array
        else:
            return self.output_array


class PFFT(object):
    """Base class for parallel FFT transforms

    Parameters
    ----------
    comm : MPI communicator
    shape : sequence of ints, optional
        shape of input array planned for
    axes : None, int, sequence of ints or sequence of sequence of ints, optional
        axes to transform over.

        - None -> All axes are transformed
        - int -> Just one axis to transform over
        - sequence of ints -> e.g., (0, 1, 2) or (0, 2, 1)
        - sequence of sequence of ints -> e.g., ((0,), (1,)) or ((0,), (1, 2))
          For seq. of seq. of ints all but the last transformed sequence
          may be longer than 1. This corresponds to collapsing axes, where
          serial FFTs are performed for all collapsed axes in one single call
    dtype : np.dtype, optional
        Type of input array
    grid : sequence of ints, optional
        Define processor grid sizes. Non positive values act as wildcards to
        allow MPI compute optimal decompositions. The sequence is padded with
        ones to match the global transform dimension.
        Use ``(-1,)`` to get a slab decomposition on the first axis.
        Use ``(1, -1)`` to get a slab decomposition  on the second axis.
        Use ``(P, Q)`` or ``(P, Q, 1)`` to get a 3D transform with 2D-pencil
        decomposition on a PxQ processor grid with the last axis non distributed.
        Use ``(P, 1, Q)`` to get a 3D transform with 2D-pencil decomposition on
        a PxQ processor grid with the second to last axis non distributed.
    padding : bool, number or sequence of numbers, optional
        If False, then no padding. If number, then apply this number as padding
        factor for all axes. If sequence of numbers, then each number gives the
        padding for each axis. Must be same length as axes.
    collapse : bool, optional
        If True try to collapse several serial transforms into one
    backend : str, optional
        Choose backend for serial transforms (``fftw``, ``pyfftw``, ``numpy``,
        ``scipy``, ``mkl_fft``). Default is ``fftw``
    transforms : None or dict, optional
        Dictionary of axes to serial transforms (forward and backward) along
        those axes. For example::

            {(0, 1): (dctn, idctn), (2, 3): (dstn, idstn)}

        If missing the default is to use rfftn/irfftn for real input arrays and
        fftn/ifftn for complex input arrays. Real-to-real transforms can be
        configured using this dictionary and real-to-real transforms from the
        :mod:`.fftw.xfftn` module. See Examples.

    Other Parameters
    ----------------
    darray : DistArray object, optional
        Create PFFT using information contained in ``darray``, neglecting most
        optional Parameters above
    slab : bool or int, optional
        DEPRECATED. If True then distribute only one axis of the global array.

    Methods
    -------
    forward(input_array=None, output_array=None, **kw)
        Parallel forward transform. The method is an instance of the
        :class:`.Transform` class. See :meth:`.Transform.__call__`

        Parameters
        ----------
        input_array : array, optional
        output_array : array, optional
        kw : dict
            parameters to serial transforms

        Returns
        -------
        output_array : array

    backward(input_array=None, output_array=None, **kw)
        Parallel backward transform. The method is an instance of the
        :class:`.Transform` class. See :meth:`.Transform.__call__`

        Parameters
        ----------
        input_array : array, optional
        output_array : array, optional
        kw : dict
            parameters to serial transforms

        Returns
        -------
        output_array : array

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py import MPI
    >>> from mpi4py_fft import PFFT, newDistArray
    >>> N = np.array([12, 14, 15], dtype=int)
    >>> fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2))
    >>> u = newDistArray(fft, False)
    >>> u[:] = np.random.random(u.shape).astype(u.dtype)
    >>> u_hat = fft.forward(u)
    >>> uj = np.zeros_like(u)
    >>> uj = fft.backward(u_hat, uj)
    >>> assert np.allclose(uj, u)

    Now configure with real-to-real discrete cosine transform type 3

    >>> from mpi4py_fft.fftw import rfftn, irfftn, dctn, idctn
    >>> import functools
    >>> dct = functools.partial(dctn, type=3)
    >>> idct = functools.partial(idctn, type=3)
    >>> transforms = {(1, 2): (dct, idct)}
    >>> r2c = PFFT(MPI.COMM_WORLD, N, axes=((0,), (1, 2)), transforms=transforms)
    >>> u = newDistArray(r2c, False)
    >>> u[:] = np.random.random(u.shape).astype(u.dtype)
    >>> u_hat = r2c.forward(u)
    >>> uj = np.zeros_like(u)
    >>> uj = r2c.backward(u_hat, uj)
    >>> assert np.allclose(uj, u)

    """
    def __init__(self, comm, shape=None, axes=None, dtype=float, grid=None,
                 padding=False, collapse=False, backend='fftw',
                 transforms=None, darray=None, **kw):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        if shape is None:
            assert darray is not None
            shape = darray._p0.shape

        if axes is not None:
            axes = list(axes) if np.ndim(axes) else [axes]
        else:
            axes = list(range(len(shape)))
            if darray is not None:
                # Make sure aligned axis of darray is transformed first
                axes = list(np.roll(axes, len(shape)-1-darray.alignment))

        for i, ax in enumerate(axes):
            if isinstance(ax, (int, np.integer)):
                if ax < 0:
                    ax += len(shape)
                axes[i] = (ax,)
            else:
                assert isinstance(ax, (tuple, list))
                ax = list(ax)
                for j, a in enumerate(ax):
                    assert isinstance(a, int)
                    if a < 0:
                        a += len(shape)
                        ax[j] = a
                axes[i] = ax
            assert min(axes[i]) >= 0
            assert max(axes[i]) < len(shape)
            assert 0 < len(axes[i]) <= len(shape)
            assert sorted(axes[i]) == sorted(set(axes[i]))

        self.axes = axes
        shape = list(shape)

        if darray is None:
            dtype = np.dtype(dtype)
            assert dtype.char in 'fdgFDG'

            if padding is not False:
                assert len(padding) == len(shape)
                for ax in axes:
                    if len(ax) == 1 and padding[ax[0]] > 1.0+1e-6:
                        old = np.float(shape[ax[0]])
                        shape[ax[0]] = int(np.floor(shape[ax[0]]*padding[ax[0]]))
                        padding[ax[0]] = shape[ax[0]] / old

            self._input_shape = tuple(shape)
            assert len(shape) > 0
            assert min(shape) > 0

            slab = kw.pop('slab', False)

            if grid is not None:
                assert not isinstance(comm, Subcomm)
                assert slab is False
                grid = tuple(grid)
                assert len(grid) <= len(shape)
                dims = list(grid) + [1] * (len(shape) - len(grid))
                comm = Subcomm(comm, dims)

            if isinstance(comm, Subcomm):
                assert slab is False
                assert len(comm) == len(shape)
                assert np.all([comm[ax].Get_size() == 1 for ax in axes[-1]])
                self.subcomm = comm
            else:
                if slab is False or slab is None:
                    dims = [0] * len(shape)
                    for ax in axes[-1]:
                        dims[ax] = 1
                else: #pragma: no cover
                    if slab is True:
                        axis = (axes[-1][-1] + 1) % len(shape)
                    else:
                        axis = slab
                        if axis < 0:
                            axis = axis + len(shape)
                        assert 0 <= axis < len(shape)
                    dims = [1] * len(shape)
                    dims[axis] = comm.Get_size()

                self.subcomm = Subcomm(comm, dims)
        else:
            dtype = darray.dtype
            self.subcomm = darray.subcomm
            self._input_shape = tuple(shape)
            commsizes = darray.commsizes
            assert np.all([commsizes[ax] == 1 for ax in axes[-1]]), "Set keyword axes such that axes to transform first are aligned"

        self.collapse = collapse
        if collapse is True:
            groups = [[]]
            for ax in reversed(axes):
                if np.all([self.subcomm[axis].Get_size() == 1 for axis in ax]):
                    [groups[0].insert(0, axis) for axis in reversed(ax)]
                else:
                    groups.insert(0, ax)
            axes = groups

        self.axes = tuple(map(tuple, axes))
        self.xfftn = []
        self.transfer = []
        self.pencil = [None, None]

        axes = self.axes[-1]
        pencil = Pencil(self.subcomm, shape, axes[-1])
        xfftn = FFT(pencil.subshape, axes, dtype, padding, backend=backend,
                    transforms=transforms, **kw)
        self.xfftn.append(xfftn)
        self.pencil[0] = pencilA = pencil
        if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
            dtype = xfftn.forward.output_array.dtype
            shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
            pencilA = Pencil(self.subcomm, shape, axes[-1])

        for axes in reversed(self.axes[:-1]):
            pencilB = pencilA.pencil(axes[-1])
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = FFT(pencilB.subshape, axes, dtype, padding, backend=backend,
                        transforms=transforms, **kw)
            self.xfftn.append(xfftn)
            self.transfer.append(transAB)
            pencilA = pencilB
            if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
                dtype = xfftn.forward.output_array.dtype
                shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
                pencilA = Pencil(pencilB.subcomm, shape, axes[-1])

        self.pencil[1] = pencilA
        self._output_shape = tuple(shape)

        self.forward = Transform(
            [o.forward for o in self.xfftn],
            [o.forward for o in self.transfer],
            self.pencil)
        self.backward = Transform(
            [o.backward for o in self.xfftn[::-1]],
            [o.backward for o in self.transfer[::-1]],
            self.pencil[::-1])

    def destroy(self):
        if isinstance(self.subcomm, Subcomm):
            self.subcomm.destroy()
        for trans in self.transfer:
            trans.destroy()

    def shape(self, forward_output=True):
        """The local (to each processor) shape of data

        Parameters
        ----------
        forward_output : bool, optional
            Return shape of output array (spectral space) if True, else return
            shape of input array (physical space)
        """
        if forward_output is not True:
            return self.forward.input_pencil.subshape
        return self.backward.input_pencil.subshape

    def local_slice(self, forward_output=True):
        """The local view into the global data

        Parameters
        ----------
        forward_output : bool, optional
            Return local slices of output array (spectral space) if True, else
            return local slices of input array (physical space)

        """
        if forward_output is not True:
            ip = self.forward.input_pencil
            s = [slice(start, start+shape) for start, shape in zip(ip.substart,
                                                                   ip.subshape)]
        else:
            ip = self.backward.input_pencil
            s = [slice(start, start+shape) for start, shape in zip(ip.substart,
                                                                   ip.subshape)]
        return tuple(s)

    def global_shape(self, forward_output=False):
        """Return global shape of associated tensors

        Parameters
        ----------
        forward_output : bool, optional
            If True then return global shape of spectral space, i.e., the input
            to a backward transfer. If False then return shape of physical
            space, i.e., the input to a forward transfer.
        """
        if forward_output:
            return self._output_shape
        return self._input_shape

    @property
    def dimensions(self):
        """The number of dimensions for transformed arrays"""
        return len(self.forward.input_array.shape)

    def dtype(self, forward_output=False):
        """The type of transformed arrays

        Parameters
        ----------
        forward_output : bool, optional
            If True then return dtype of an array that is the result of a
            forward transform. Otherwise, return the dtype of an array that
            is input to a forward transform.
        """
        if forward_output:
            return self.forward.output_array.dtype
        return self.forward.input_array.dtype

from copy import copy
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
    shape : sequence of ints
        shape of input array planned for
    axes : None, int, sequence of ints or sequence of sequence of ints, optional
        axes to transform over.

        - None -> All axes are transformed
        - int -> Just one axis to transform over
        - sequence of ints -> e.g., (0, 1, 2) or (0, 2, 1)
        - sequence of sequence of ints -> e.g., ((0,), (1,)) or ((0,), (1, 2))
          For seq. of seq. of ints only the last inner sequence may be longer
          than 1. This corresponds to collapsing axes, where serial FFTs are
          performed for all collapsed axes in one single call
    dtype : np.dtype, optional
        Type of input array
    slab : bool, optional
        If True then distribute only one index of the global array
    padding : bool, number or sequence of numbers, optional
        If False, then no padding. If number, then apply this number as padding
        factor for all axes. If sequence of numbers, then each number gives the
        padding for each axis. Must be same length as axes.
    collapse : bool, optional
        If True try to collapse several serial transforms into one
    use_pyfftw : bool, optional
        Use pyfftw for serial transforms instead of local wrappers
    transforms : None or dict, optional
        Dictionary of axes to serial transforms (forward and backward) along
        those axes. For example::

            {(0, 1): (dctn, idctn), (2, 3): (dstn, idstn)}

        If missing the default is to use rfftn/irfftn for real input arrays and
        fftn/ifftn for complex input arrays. Real-to-real transforms can be
        configured using this dictionary and real-to-real transforms from the
        :mod:`.fftw.xfftn` module. See Examples.

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
    >>> from mpi4py_fft.mpifft import PFFT, Function
    >>> N = np.array([12, 14, 15], dtype=int)
    >>> fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2))
    >>> u = Function(fft, False)
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
    >>> u = Function(r2c, False)
    >>> u[:] = np.random.random(u.shape).astype(u.dtype)
    >>> u_hat = r2c.forward(u)
    >>> uj = np.zeros_like(u)
    >>> uj = r2c.backward(u_hat, uj)
    >>> assert np.allclose(uj, u)

    """
    def __init__(self, comm, shape, axes=None, dtype=float,
                 slab=False, padding=False, collapse=False,
                 use_pyfftw=False, transforms=None, **kw):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        if axes is not None:
            axes = list(axes) if np.ndim(axes) else [axes]
        else:
            axes = list(range(len(shape)))

        for i, ax in enumerate(axes):
            if isinstance(ax, int):
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

        dtype = np.dtype(dtype)
        assert dtype.char in 'fdgFDG'

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
            else:
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
        xfftn = FFT(pencil.subshape, axes, dtype, padding, use_pyfftw,
                    transforms, **kw)
        self.xfftn.append(xfftn)
        self.pencil[0] = pencilA = pencil
        if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
            dtype = xfftn.forward.output_array.dtype
            shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
            pencilA = Pencil(self.subcomm, shape, axes[-1])

        for axes in reversed(self.axes[:-1]):
            pencilB = pencilA.pencil(axes[-1])
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = FFT(pencilB.subshape, axes, dtype, padding, use_pyfftw,
                        transforms, **kw)
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
        self.subcomm.destroy()
        for trans in self.transfer:
            trans.destroy()

    def local_shape(self, forward_output=True):
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

    def shape(self, forward_output=False):
        """Return shape of tensor for space

        Parameters
        ----------
        forward_output : bool, optional
            If True then return shape of spectral space, i.e., the input to
            a backward transfer. If False then return shape of physical
            space, i.e., the input to a forward transfer.
        """
        if forward_output:
            return self._output_shape
        return self._input_shape

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


class Function(np.ndarray):
    """Distributed Numpy array for instance of PFFT class

    Basically just a Numpy array created with the shape according to the input
    PFFT instance

    Parameters
    ----------

    pfft : Instance of :class:`.PFFT` class
    forward_output: boolean, optional
        If False then create Function of shape/type for input to
        forward transform, otherwise create Function of shape/type for
        output from forward transform.
    val : int or float
        Value used to initialize array.
    tensor: int or tuple
        For tensorvalued Functions, e.g., tensor=(3) for a vector in 3D.

    For more information, see `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from mpi4py_fft.mpifft import PFFT, Function
    >>> FFT = PFFT(MPI.COMM_WORLD, [64, 64, 64])
    >>> u = Function(FFT, False, tensor=3)
    >>> u_hat = Function(FFT, True, tensor=3)

    """
    def __new__(cls, pfft, forward_output=True, val=0, tensor=None):

        shape = pfft.forward.input_array.shape
        dtype = pfft.forward.input_array.dtype
        if forward_output is True:
            shape = pfft.forward.output_array.shape
            dtype = pfft.forward.output_array.dtype

        if not tensor is None:
            tensor = list(tensor) if np.ndim(tensor) else [tensor]
            shape = tensor + list(shape)

        obj = np.ndarray.__new__(cls,
                                 shape,
                                 dtype=dtype)
        obj.fill(val)
        return obj

    def __init__(self, pfft, forward_output=True, val=0, tensor=None):
        pass

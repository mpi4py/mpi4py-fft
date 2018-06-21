from copy import copy
import numpy as np

from .libfft import FFT
from .pencil import Pencil
from .pencil import Subcomm


class Transform(object):
    """Class for performing any parallel transform

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
            Function values on quadrature mesh
        output_array : array, optional
            Expansion coefficients
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
    shape : list or tuple of ints
        shape of input array planned for
    axes : None, int or tuple of ints, optional
        axes to transform over. If None transform over all axes
    dtype : np.dtype, optional
        Type of input array
    slab : bool, optional
        If True then distribute only one index of the global array
    padding : bool, number or list of numbers, optional
        If False, then no padding. If number, then apply this number as padding
        factor for all axes. If list of numbers, then each number gives the
        padding for each axis. Must be same length as axes.
    collapse : bool, optional
        If True try to collapse several serial transforms into one object
    use_pyfftw : bool, optional
        Use pyfftw for serial transforms instead of local wrappers

    Methods
    -------
    forward
        Parallel forward transform. The method is an instance of the
        :class:`.Transform` class
    backward
        Parallel backward transform. The method is an instance of the
        :class:`.Transform` class
    """

    def __init__(self, comm, shape, axes=None, dtype=float,
                 slab=False, padding=False, collapse=False,
                 use_pyfftw=False, **kw):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        if axes is not None:
            axes = list(axes) if np.ndim(axes) else [axes]
            for i, axis in enumerate(axes):
                if axis < 0:
                    axes[i] = axis + len(shape)
        else:
            axes = list(range(len(shape)))
        assert min(axes) >= 0
        assert max(axes) < len(shape)
        assert 0 < len(axes) <= len(shape)
        assert sorted(axes) == sorted(set(axes))

        shape = list(shape)
        if padding is not False:
            assert len(padding) == len(shape)
            for axis in axes:
                old = np.float(shape[axis])
                shape[axis] = int(np.floor(shape[axis]*padding[axis]))
                padding[axis] = shape[axis] / old

        self._input_shape = copy(shape)
        assert len(shape) > 0
        assert min(shape) > 0

        dtype = np.dtype(dtype)
        assert dtype.char in 'fdgFDG'

        if isinstance(comm, Subcomm):
            assert slab is False
            assert len(comm) == len(shape)
            assert comm[axes[-1]].Get_size() == 1
            self.subcomm = comm
        else:
            if slab is False or slab is None:
                dims = [0] * len(shape)
                dims[axes[-1]] = 1
            else:
                if slab is True:
                    axis = (axes[-1] + 1) % len(shape)
                else:
                    axis = slab
                    if axis < 0:
                        axis = axis + len(shape)
                    assert 0 <= axis < len(shape)
                dims = [1] * len(shape)
                dims[axis] = comm.Get_size()
            self.subcomm = Subcomm(comm, dims)

        if padding is not False:
            collapse = False
        self.collapse = collapse

        if collapse is True:
            groups = [[]]
            for axis in reversed(axes):
                if self.subcomm[axis].Get_size() == 1:
                    groups[0].insert(0, axis)
                else:
                    groups.insert(0, [axis])
            self.axes = tuple(map(tuple, groups))
        else:
            self.axes = tuple((axis,) for axis in axes)

        self.xfftn = []
        self.transfer = []
        self.pencil = [None, None]

        axes = self.axes[-1]
        pencil = Pencil(self.subcomm, shape, axes[-1])
        xfftn = FFT(pencil.subshape, axes, dtype, padding, use_pyfftw, **kw)
        self.xfftn.append(xfftn)
        self.pencil[0] = pencilA = pencil
        if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
            dtype = xfftn.forward.output_array.dtype
            shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
            pencilA = Pencil(self.subcomm, shape, axes[-1])

        for axes in reversed(self.axes[:-1]):
            pencilB = pencilA.pencil(axes[-1])
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = FFT(pencilB.subshape, axes, dtype, padding, use_pyfftw, **kw)
            self.xfftn.append(xfftn)
            self.transfer.append(transAB)
            pencilA = pencilB
            if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
                shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
                pencilA = Pencil(pencilB.subcomm, shape, axes[-1])

        self.pencil[1] = pencilA
        self._output_shape = copy(shape)

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

    def local_shape(self, spectral=True):
        """The local (to each processor) shape of data

        Parameters
        ----------
        spectral : bool, optional
            Return shape of output array (spectral space) if True, else return
            shape of input array (physical space)
        """
        if not spectral:
            return self.forward.input_pencil.subshape
        return self.backward.input_pencil.subshape

    def local_slice(self, spectral=True):
        """The local view into the global data

        Parameters
        ----------
        spectral : bool, optional
            Return local slices of output array (spectral space) if True, else
            return local slices of input array (physical space)

        """
        if spectral is not True:
            ip = self.forward.input_pencil
            s = [slice(start, start+shape) for start, shape in zip(ip.substart,
                                                                   ip.subshape)]
        else:
            ip = self.backward.input_pencil
            s = [slice(start, start+shape) for start, shape in zip(ip.substart,
                                                                   ip.subshape)]
        return s

    def input_shape(self):
        """Return global shape of input array"""
        return self._input_shape

    def output_shape(self):
        """Return global shape of output array"""
        return self._output_shape


class Function(np.ndarray):
    """Distributed Numpy array for instance of PFFT class

    Basically just a Numpy array created with the shape according to the input
    PFFT instance

    Parameters
    ----------

    pfft : Instance of PFFT class
    forward_output: boolean.
        If False then create Function of shape/type for input to PFFT.forward,
        otherwise create Function of shape/type for output from PFFT.forward
    val : int or float
        Value used to initialize array
    tensor: int or tuple
        For tensorvalued Functions, e.g., tensor=(3) for a vector in 3D.

    For more information, see numpy.ndarray

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from mpi4py_fft.mpifft import PFFT, Function
    >>> FFT = PFFT(MPI.COMM_WORLD, [64, 64, 64])
    >>> u = Function(FFT, tensor=3)
    >>> uhat = Function(FFT, False, tensor=3)

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

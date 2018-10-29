import numpy as np
from . import fftw

def _Xfftn_plan_pyfftw(shape, axes, dtype, transforms, options):
    """Plan serial transforms using pyfftw

    Parameters
    ----------
    shape : list of ints
        shape of input array planned for
    axes : list of ints
        axes to transform over
    dtype : np.dtype
        type of input array
    options : dict
        arguments for planning serial transforms
    """
    import pyfftw
    opts = dict(
        avoid_copy=True,
        overwrite_input=True,
        auto_align_input=True,
        auto_contiguous=True,
        threads=1,
    )
    opts.update(options)

    transforms = {} if transforms is None else transforms
    if tuple(axes) in transforms:
        plan_fwd, plan_bck = transforms[tuple(axes)]
    else:
        if np.issubdtype(dtype, np.floating):
            plan_fwd = pyfftw.builders.rfftn
            plan_bck = pyfftw.builders.irfftn
        else:
            plan_fwd = pyfftw.builders.fftn
            plan_bck = pyfftw.builders.ifftn

    s = tuple(np.take(shape, axes))

    U = pyfftw.empty_aligned(shape, dtype=dtype)
    xfftn_fwd = plan_fwd(U, s=s, axes=axes, **opts)
    U.fill(0)
    if np.issubdtype(dtype, np.floating):
        del opts['overwrite_input']
    V = xfftn_fwd.output_array
    xfftn_bck = plan_bck(V, s=s, axes=axes, **opts)
    V.fill(0)

    xfftn_fwd.update_arrays(U, V)
    xfftn_bck.update_arrays(V, U)

    return (xfftn_fwd, xfftn_bck)

def _Xfftn_plan_mpi4py(shape, axes, dtype, transforms, options):
    """Plan serial transforms using local wrapper

    Parameters
    ----------
    shape : list of ints
        shape of input array planned for
    axes : list of ints
        axes to transform over
    dtype : np.dtype
        type of input array
    options : dict
        arguments for planning serial transforms
    """
    opts = dict(
        overwrite_input='FFTW_DESTROY_INPUT',
        planner_effort='FFTW_MEASURE',
        threads=1,
    )
    opts.update(options)
    flags = (fftw.flag_dict[opts['planner_effort']],
             fftw.flag_dict[opts['overwrite_input']])
    threads = opts['threads']

    transforms = {} if transforms is None else transforms
    if tuple(axes) in transforms:
        plan_fwd, plan_bck = transforms[tuple(axes)]
    else:
        if np.issubdtype(dtype, np.floating):
            plan_fwd = fftw.rfftn
            plan_bck = fftw.irfftn
        else:
            plan_fwd = fftw.fftn
            plan_bck = fftw.ifftn

    s = tuple(np.take(shape, axes))

    U = fftw.aligned(shape, dtype=dtype)
    xfftn_fwd = plan_fwd(U, s=s, axes=axes, threads=threads, flags=flags)
    U.fill(0)
    V = xfftn_fwd.output_array
    if np.issubdtype(dtype, np.floating):
        flags = (fftw.flag_dict[opts['planner_effort']],)

    xfftn_bck = plan_bck(V, s=s, axes=axes, threads=threads, flags=flags, output_array=U)

    return (xfftn_fwd, xfftn_bck)


class _Xfftn_wrap(object):

    # pylint: disable=too-few-public-methods

    __slots__ = ('_xfftn', '__doc__', '_input_array', '_output_array')

    def __init__(self, xfftn_obj, input_array, output_array):
        object.__setattr__(self, '_xfftn', xfftn_obj)
        object.__setattr__(self, '_input_array', input_array)
        object.__setattr__(self, '_output_array', output_array)
        object.__setattr__(self, '__doc__', xfftn_obj.__doc__)

    @property
    def input_array(self):
        return object.__getattribute__(self, '_input_array')

    @property
    def output_array(self):
        return object.__getattribute__(self, '_output_array')

    @property
    def xfftn(self):
        return object.__getattribute__(self, '_xfftn')

    def __call__(self, input_array=None, output_array=None, **options):
        if input_array is not None:
            self.input_array[...] = input_array
        self.xfftn(**options)
        if output_array is not None:
            output_array[...] = self.output_array
            return output_array
        else:
            return self.output_array

class FFTBase(object):
    """Base class for serial FFT transforms

    Parameters
    ----------
    shape : list or tuple of ints
        shape of input array planned for
    axes : None, int or tuple of ints, optional
        axes to transform over. If None transform over all axes
    dtype : np.dtype, optional
        Type of input array
    padding : bool, number or list of numbers
        If False, then no padding. If number, then apply this number as padding
        factor for all axes. If list of numbers, then each number gives the
        padding for each axis. Must be same length as axes.
    """

    def __init__(self, shape, axes=None, dtype=float, padding=False):
        shape = list(shape) if np.ndim(shape) else [shape]
        assert len(shape) > 0
        assert min(shape) > 0
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

        dtype = np.dtype(dtype)
        assert dtype.char in 'fdgFDG'
        self.shape = shape
        self.axes = axes
        self.dtype = dtype
        self.padding = padding
        self.real_transform = np.issubdtype(dtype, np.floating)
        self.padding_factor = 1

    def _truncation_forward(self, padded_array, trunc_array):
        axis = self.axes[-1]
        if self.padding_factor > 1.0+1e-8:
            trunc_array.fill(0)
            N0 = self.forward.output_array.shape[axis]
            if self.real_transform:
                N = trunc_array.shape[axis]
                s = [slice(None)]*trunc_array.ndim
                s[axis] = slice(0, N)
                trunc_array[:] = padded_array[tuple(s)]
                if N0 % 2 == 0:
                    s[axis] = N-1
                    s = tuple(s)
                    trunc_array[s] = trunc_array[s].real
                    trunc_array[s] *= 2
            else:
                N = trunc_array.shape[axis]
                su = [slice(None)]*trunc_array.ndim
                su[axis] = slice(0, N//2+1)
                trunc_array[tuple(su)] = padded_array[tuple(su)]
                su[axis] = slice(-(N//2), None)
                trunc_array[tuple(su)] += padded_array[tuple(su)]

    def _padding_backward(self, trunc_array, padded_array):
        axis = self.axes[-1]
        if self.padding_factor > 1.0+1e-8:
            padded_array.fill(0)
            N0 = self.forward.output_array.shape[axis]
            if self.real_transform:
                s = [slice(0, n) for n in trunc_array.shape]
                padded_array[tuple(s)] = trunc_array[:]
                N = trunc_array.shape[axis]
                if N0 % 2 == 0: # Symmetric Fourier interpolator
                    s[axis] = N-1
                    s = tuple(s)
                    padded_array[s] = padded_array[s].real
                    padded_array[s] *= 0.5
            else:
                N = trunc_array.shape[axis]
                su = [slice(None)]*trunc_array.ndim
                su[axis] = slice(0, N//2+1)
                padded_array[tuple(su)] = trunc_array[tuple(su)]
                su[axis] = slice(-(N//2), None)
                padded_array[tuple(su)] = trunc_array[tuple(su)]
                if N0 % 2 == 0:  # Use symmetric Fourier interpolator
                    su[axis] = N//2
                    padded_array[tuple(su)] *= 0.5
                    su[axis] = -(N//2)
                    padded_array[tuple(su)] *= 0.5


class FFT(FFTBase):
    """Class for serial FFT transforms

    Parameters
    ----------
    shape : list or tuple of ints
        shape of input array planned for
    axes : None, int or tuple of ints, optional
        axes to transform over. If None transform over all axes
    dtype : np.dtype, optional
        Type of input array
    padding : bool, number or list of numbers
        If False, then no padding. If number, then apply this number as padding
        factor for all axes. If list of numbers, then each number gives the
        padding for each axis. Must be same length as axes.
    kw : dict
        Parameters passed to serial transform object

    Methods
    -------
    forward(input_array=None, output_array=None, **options)
        Generic serial forward transform

        Parameters
        ----------
        input_array : array, optional
        output_array : array, optional
        options : dict
            parameters to serial transforms

        Returns
        -------
        output_array : array

    backward(input_array=None, output_array=None, **options)
        Generic serial backward transform

        Parameters
        ----------
        input_array : array, optional
        output_array : array, optional
        options : dict
            parameters to serial transforms

        Returns
        -------
        output_array : array

    """
    def __init__(self, shape, axes=None, dtype=float, padding=False,
                 use_pyfftw=False, transforms=None, **kw):
        FFTBase.__init__(self, shape, axes, dtype, padding)
        plan = _Xfftn_plan_pyfftw if use_pyfftw is True else _Xfftn_plan_mpi4py
        self.fwd, self.bck = plan(self.shape, self.axes, self.dtype, transforms, kw)
        U, V = self.fwd.input_array, self.fwd.output_array
        if use_pyfftw:
            self.M = 1./np.prod(np.take(self.shape, self.axes))
        else:
            self.M = self.fwd.get_normalization()
        self.padding_factor = 1.0
        if padding is not False:
            self.padding_factor = padding[self.axes[-1]] if np.ndim(padding) else padding
        if abs(self.padding_factor-1.0) > 1e-8:
            assert len(self.axes) == 1
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = _Xfftn_wrap(self._forward, U, trunc_array)
            self.backward = _Xfftn_wrap(self._backward, trunc_array, U)
        else:
            self.forward = _Xfftn_wrap(self._forward, U, V)
            self.backward = _Xfftn_wrap(self._backward, V, U)

    def _forward(self, **kw):
        self.fwd(None, None, **kw)
        self._truncation_forward(self.fwd.output_array, self.forward.output_array)
        self.forward._output_array *= self.M
        return self.forward.output_array

    def _backward(self, **kw):
        self._padding_backward(self.backward.input_array, self.bck.input_array)
        self.bck(None, None, normalise_idft=False, **kw)
        return self.backward.output_array

    def _get_truncarray(self, shape, dtype):
        axis = self.axes[-1]
        if not self.real_transform:
            shape = list(shape)
            shape[axis] = int(np.round(shape[axis] / self.padding_factor))
            return fftw.aligned(shape, dtype=dtype)

        shape = list(shape)
        shape[axis] = int(np.round(shape[axis] / self.padding_factor))
        shape[axis] = shape[axis]//2 + 1
        return fftw.aligned(shape, dtype=dtype)


class FFTNumPy(FFTBase): #pragma: no cover
    """Class for serial FFT transforms using Numpy FFT

    Parameters
    ----------
    shape : list or tuple of ints
        shape of input array planned for
    axes : None, int or tuple of ints, optional
        axes to transform over. If None transform over all axes
    dtype : np.dtype, optional
        Type of input array
    padding : bool, number or list of numbers
        If False, then no padding. If number, then apply this number as padding
        factor for all axes. If list of numbers, then each number gives the
        padding for each axis. Must be same length as axes.
    kw : dict
        Parameters passed to serial transform object

    forward(input_array=None, output_array=None, **options)
        Generic serial forward transform

        Parameters
        ----------
        input_array : array, optional
        output_array : array, optional
        options : dict
            parameters to serial transforms

        Returns
        -------
        output_array : array

    backward(input_array=None, output_array=None, **options)
        Generic serial backward transform

        Parameters
        ----------
        input_array : array, optional
        output_array : array, optional
        options : dict
            parameters to serial transforms

        Returns
        -------
        output_array : array

    """

    def __init__(self, shape, axes=None, dtype=float, padding=False, **kw):
        FFTBase.__init__(self, shape, axes, dtype, padding)
        typecode = self.dtype.char

        self.sizes = list(np.take(self.shape, self.axes))
        arrayA = np.zeros(self.shape, self.dtype)
        if self.real_transform:
            axis = self.axes[-1]
            self.shape[axis] = self.shape[axis]//2 + 1
            arrayB = np.zeros(self.shape, typecode.upper())
            fwd = np.fft.rfftn
            bck = np.fft.irfftn
        else:
            arrayB = np.zeros(self.shape, typecode)
            fwd = np.fft.fftn
            bck = np.fft.ifftn

        fwd.input_array = arrayA
        fwd.output_array = arrayB
        bck.input_array = arrayB
        bck.output_array = arrayA
        self.fwd, self.bck = fwd, bck

        self.padding_factor = 1
        if padding is not False:
            assert len(self.axes) == 1
            self.axis = self.axes[-1]
            self.padding_factor = padding[axes[-1]] if np.ndim(padding) else padding
            trunc_array = self._get_truncarray(shape, arrayB.dtype)
            self.forward = _Xfftn_wrap(self._forward, arrayA, trunc_array)
            self.backward = _Xfftn_wrap(self._backward, trunc_array, arrayA)
        else:
            self.forward = _Xfftn_wrap(self._forward, arrayA, arrayB)
            self.backward = _Xfftn_wrap(self._backward, arrayB, arrayA)

    def _forward(self, **kw):
        self.fwd.output_array[:] = self.fwd(self.fwd.input_array, s=self.sizes,
                                            axes=self.axes, **kw)
        self._truncation_forward(self.fwd.output_array, self.forward.output_array)
        return self.forward.output_array

    def _backward(self, **kw):
        self._padding_backward(self.backward.input_array, self.bck.input_array)
        self.backward.output_array[:] = self.bck(self.bck.input_array, s=self.sizes,
                                                 axes=self.axes, **kw)
        return self.backward.output_array

    def _get_truncarray(self, shape, dtype):
        axis = self.axes[-1]
        if not self.real_transform:
            shape = list(shape)
            shape[axis] = int(np.round(shape[axis] / self.padding_factor))
            return np.zeros(shape, dtype=dtype)

        shape = list(shape)
        shape[axis] = int(np.round(shape[axis] / self.padding_factor))
        shape[axis] = shape[axis]//2 + 1
        return np.zeros(shape, dtype=dtype)

#FFT = FFTNumPy

import numpy as np
import pyfftw


def _Xfftn_plan(shape, axes, dtype, options):
    assert pyfftw
    opts = dict(
        avoid_copy=True,
        overwrite_input=True,
        auto_align_input=True,
        auto_contiguous=True,
        threads=1,
    )
    opts.update(options)

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


class _Xfftn_wrap(object):

    # pylint: disable=too-few-public-methods

    __slots__ = ('_xfftn', 'axes', 'input_array', 'output_array')

    def __init__(self, xfftn_obj, axes, input_array, output_array):
        object.__setattr__(self, '_xfftn', xfftn_obj)
        object.__setattr__(self, 'axes', axes)
        object.__setattr__(self, 'input_array', input_array)
        object.__setattr__(self, 'output_array', output_array)

    def __call__(self, input_array=None, output_array=None, **options):
        xfftn = object.__getattribute__(self, '_xfftn')
        if input_array is not None:
            self.input_array[...] = input_array
        xfftn(**options)
        if output_array is not None:
            output_array[...] = self.output_array
            return output_array
        else:
            return self.output_array


class FFT(object):

    # pylint: disable=too-few-public-methods

    def __init__(self, shape, axes=None, dtype=float, padding=1, **kw):
        shape = list(shape) if np.ndim(shape) else [shape]
        assert len(shape) > 0
        assert min(shape) > 0
        self.padding_factor = padding

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

        self.fwd, self.bck = _Xfftn_plan(shape, axes, dtype, kw)
        self.real_transform = np.issubdtype(dtype, np.floating)
        self.axes = axes
        U, V = self.fwd.input_array, self.fwd.output_array

        if padding > 1.+1e-8:
            assert len(axes) == 1
            self.axis = axes[-1]
            trunc_array = self._get_truncarray(shape, dtype)
            self.forward = _Xfftn_wrap(self._forward, tuple(axes), U, trunc_array)
            self.backward = _Xfftn_wrap(self._backward, tuple(axes), trunc_array, U)
        else:
            self.forward = _Xfftn_wrap(self._forward, tuple(axes), U, V)
            self.backward = _Xfftn_wrap(self._backward, tuple(axes), V, U)

    def _get_truncarray(self, shape, dtype):
        if not self.real_transform:
            shape = list(shape)
            shape[self.axis] = int(shape[self.axis] / self.padding_factor)
            return pyfftw.empty_aligned(shape, dtype=dtype)
        else:
            shape = list(shape)
            shape[self.axis] = int(shape[self.axis] / self.padding_factor)
            shape[self.axis] = shape[self.axis]//2 + 1
            return pyfftw.empty_aligned(shape, dtype=np.complex)

    def _forward(self, **kw):
        self.fwd(None, None, **kw)
        self._truncation_forward(self.fwd.output_array, self.forward.output_array)
        return self.forward.output_array

    def _backward(self, **kw):
        self._padding_backward(self.backward.input_array, self.bck.input_array)
        self.bck(None, None, **kw)
        return self.backward.output_array

    def _truncation_forward(self, padded_array, trunc_array):
        if self.padding_factor > 1.0+1e-8:
            trunc_array.fill(0)
            if self.real_transform:
                N = trunc_array.shape[self.axis]
                s = [slice(None)]*trunc_array.ndim
                s[self.axis] = slice(0, N)
                trunc_array[:] = padded_array[s]
                trunc_array *= (1./self.padding_factor)
            else:
                N = trunc_array.shape[self.axis]
                su = [slice(None)]*trunc_array.ndim
                su[self.axis] = slice(0, N//2+1)
                trunc_array[su] = padded_array[su]
                su[self.axis] = slice(-N//2, None)
                trunc_array[su] += padded_array[su]
                trunc_array *= (1./self.padding_factor)

    def _padding_backward(self, trunc_array, padded_array):
        if self.padding_factor > 1.0+1e-8:
            padded_array.fill(0)
            if self.real_transform:
                s = [slice(0, n) for n in trunc_array.shape]
                padded_array[s] = trunc_array[:]
                padded_array *= self.padding_factor
            else:
                N = trunc_array.shape[self.axis]
                su = [slice(None)]*trunc_array.ndim
                su[self.axis] = slice(0, N//2)
                padded_array[su] = trunc_array[su]
                su[self.axis] = slice(-N//2, None)
                padded_array[su] = trunc_array[su]
                padded_array *= self.padding_factor


class FFTNumPy(object):

    # pylint: disable=too-few-public-methods

    class _Wrap(object):

        def __init__(self, xfftn, sizes, axes, in_array, out_array):
            # pylint: disable=too-many-arguments
            self.xfftn = xfftn
            self.sizes = sizes
            self.axes = axes
            self.input_array = in_array
            self.output_array = out_array

        def __call__(self, input_array=None, output_array=None, **kw):
            if input_array is None:
                input_array = self.input_array
            if output_array is None:
                output_array = self.output_array
            sizes, axes = self.sizes, self.axes
            output_array[...] = self.xfftn(input_array, s=sizes, axes=axes)
            return output_array

    def __init__(self, shape, axes=None, dtype=float):
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
        typecode = dtype.char

        sizes = list(np.take(shape, axes))
        arrayA = np.zeros(shape, dtype)
        if np.issubdtype(dtype, np.floating):
            axis = axes[-1]
            shape[axis] = shape[axis]//2 + 1
            arrayB = np.zeros(shape, typecode.upper())
            fwd = np.fft.rfftn
            bck = np.fft.irfftn
        else:
            arrayB = np.zeros(shape, typecode)
            fwd = np.fft.fftn
            bck = np.fft.ifftn

        self.forward = self._Wrap(fwd, sizes, axes, arrayA, arrayB)
        self.backward = self._Wrap(bck, sizes, axes, arrayB, arrayA)


# FFT = FFTNumPy

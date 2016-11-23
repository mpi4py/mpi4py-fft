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


def _Xfftn_exec(xfftn_obj, in_array, out_array, options):
    if in_array is not None:
        xfftn_obj.input_array[...] = in_array
    xfftn_obj(None, None, **options)
    if out_array is not None:
        out_array[...] = xfftn_obj.output_array
        return out_array
    else:
        return xfftn_obj.output_array


class _Xfftn_wrap(object):

    __slots__ = ('_obj',)

    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)

    def __getattribute__(self, name):
        obj = object.__getattribute__(self, '_obj')
        return getattr(obj, name)

    def __call__(self, input_array=None, output_array=None, **kw):
        xfftn_obj = object.__getattribute__(self, '_obj')
        return _Xfftn_exec(xfftn_obj, input_array, output_array, kw)


class FFT(object):

    def __init__(self, shape, axes=None, dtype=float, **kw):
        shape = tuple(shape) if np.ndim(shape) else (shape,)
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

        fwd, bck = _Xfftn_plan(shape, axes, dtype, kw)
        self.forward = _Xfftn_wrap(fwd)
        self.backward = _Xfftn_wrap(bck)


class FFTNumPy(object):

    class _Wrap(object):

        def __init__(self, xfftn, s, axes, in_array, out_array):
            self.xfftn = xfftn
            self.s = s
            self.axes = axes
            self.input_array = in_array
            self.output_array = out_array

        def __call__(self, input_array=None, output_array=None, **kw):
            if input_array is None:
                input_array = self.input_array
            if output_array is None:
                output_array = self.output_array
            s, axes = self.s, self.axes
            output_array[...] = self.xfftn(input_array, s=s, axes=axes)
            return output_array

    def __init__(self, shape, axes=None, dtype=float):
        shape = tuple(shape) if np.ndim(shape) else (shape,)
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

        s = tuple(np.take(shape, axes))
        arrayA = np.zeros(shape, dtype)
        if np.issubdtype(dtype, np.floating):
            axis = axes[-1]
            shape = list(shape)
            shape[axis] = shape[axis]//2 + 1
            arrayB = np.zeros(shape, dtype.char.upper())
            fwd = np.fft.rfftn
            bck = np.fft.irfftn
        else:
            arrayB = np.zeros(shape, dtype)
            fwd = np.fft.fftn
            bck = np.fft.ifftn

        self.forward = self._Wrap(fwd, s, axes, arrayA, arrayB)
        self.backward = self._Wrap(bck, s, axes, arrayB, arrayA)


# FFT = FFTNumPy

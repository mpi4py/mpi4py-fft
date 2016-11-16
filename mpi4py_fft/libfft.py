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

    U = pyfftw.empty_aligned(shape, dtype=dtype)
    xfftn_fwd = plan_fwd(U, axes=axes, **opts)
    U.fill(0)
    if np.issubdtype(dtype, np.floating):
        del opts['overwrite_input']
    V = xfftn_fwd.output_array
    xfftn_bck = plan_bck(V, axes=axes, **opts)
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
            axes = tuple(axes) if np.ndim(axes) else (axes,)
            assert 0 < len(axes) <= len(shape)
            assert min(axes) >= -len(shape)
            assert max(axes) < len(shape)
        else:
            axes = tuple(range(len(shape)))

        dtype = np.dtype(dtype)
        assert dtype.char in 'fdgFDG'
        assert not (dtype.char in 'fdg' and shape[axes[-1]] % 2 != 0)

        fwd, bck = _Xfftn_plan(shape, axes, dtype, kw)
        self.forward = _Xfftn_wrap(fwd)
        self.backward = _Xfftn_wrap(bck)



class FFTNumPy(object):

    class Wrapper(object):

        def __init__(self, xfftn, axes, in_array, out_array):
            self.xfftn = xfftn
            self.axes = axes
            self.input_array = in_array
            self.output_array = out_array

        def __call__(self, input_array=None, output_array=None, **kw):
            if input_array is None:
                input_array = self.input_array
            if output_array is None:
                output_array = self.output_array
            output_array[...] = self.xfftn(input_array, axes=self.axes)
            return output_array

    def __init__(self, shape, axes=None, dtype=float, **kw):
        shape = tuple(shape) if np.ndim(shape) else (shape,)
        assert len(shape) > 0
        assert min(shape) > 0

        if axes is not None:
            axes = tuple(axes) if np.ndim(axes) else (axes,)
            assert 0 < len(axes) <= len(shape)
            assert min(axes) >= -len(shape)
            assert max(axes) < len(shape)
        else:
            axes = tuple(range(len(shape)))

        dtype = np.dtype(dtype)
        assert dtype.char in 'fdgFDG'
        assert not (dtype.char in 'fdg' and shape[axes[-1]] % 2 != 0)

        arrayA = np.zeros(shape, dtype)
        if np.issubdtype(dtype, np.floating):
            shape = list(shape)
            shape[axis] = shape[axes[-1]]//2 + 1
            arrayB = np.zeros(shape, dtype.char.upper())
            fwd = np.fft.rfftn
            bck = np.fft.irfftn
        else:
            arrayB = np.zeros(shape, dtype)
            fwd = np.fft.fftn
            bck = np.fft.ifftn
        self.forward  = self.Wrapper(fwd, axes, arrayA, arrayB)
        self.backward = self.Wrapper(bck, axes, arrayB, arrayA)


#FFT = FFTNumPy

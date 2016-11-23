import numpy as np
from mpi4py import MPI  # pylint: disable=unused-import

from .libfft import FFT
from .pencil import Pencil
from .pencil import Subcomm


class Transform(object):

    def __init__(self, xfftn, transfer, pencil):
        assert len(xfftn) == len(transfer) + 1 and len(pencil) == 2
        self._xfftn = tuple(xfftn)
        self._transfer = tuple(transfer)
        self._pencil = tuple(pencil)

    @property
    def input_array(self):
        return self._xfftn[0].input_array

    @property
    def output_array(self):
        return self._xfftn[-1].output_array

    @property
    def input_pencil(self):
        return self._pencil[0]

    @property
    def output_pencil(self):
        return self._pencil[1]

    def __call__(self, input_array=None, output_array=None, **kw):
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

    # pylint: disable=too-few-public-methods

    def __init__(self, comm, shape, axes=None, dtype=float, **kw):
        shape = list(shape)
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

        self.axes = tuple(axes)
        self.subcomm = Subcomm(comm, len(shape)-1)
        self.xfftn = []
        self.transfer = []
        self.pencil = [None, None]

        axis = self.axes[-1]
        pencil = Pencil(self.subcomm, shape, axis)
        xfftn = FFT(pencil.subshape, axis, dtype, **kw)
        self.xfftn.append(xfftn)

        self.pencil[0] = pencilA = pencil
        if np.issubdtype(dtype, np.floating):
            shape[axis] = shape[axis]//2 + 1
            dtype = xfftn.forward.output_array.dtype
            pencilA = Pencil(self.subcomm, shape, axis)
        for axis in reversed(self.axes[:-1]):
            pencilB = pencilA.pencil(axis)
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = FFT(pencilB.subshape, axis, dtype, **kw)
            self.xfftn.append(xfftn)
            self.transfer.append(transAB)
            pencilA = pencilB
        self.pencil[1] = pencilA

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

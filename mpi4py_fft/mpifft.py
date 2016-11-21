import numpy as np
from mpi4py import MPI

from .libfft import FFT
from .pencil import Pencil
from .pencil import Subcomm


class Transform(object):

    def __init__(self, xfftnseq, transfer):
        assert len(xfftnseq) == len(transfer) + 1
        self.xfftnseq = xfftnseq
        self.transfer = transfer

    @property
    def input_array(self):
        return self.xfftnseq[0].input_array

    @property
    def output_array(self):
        return self.xfftnseq[-1].output_array

    def __call__(self, input_array=None, output_array=None, **kw):
        if input_array is not None:
            self.input_array[...] = input_array

        for i in range(len(self.transfer)):
            self.xfftnseq[i](**kw)
            arrayA = self.xfftnseq[i].output_array
            arrayB = self.xfftnseq[i+1].input_array
            self.transfer[i](arrayA, arrayB)
        self.xfftnseq[-1](**kw)

        if output_array is not None:
            output_array[...] = self.output_array
            return output_array
        else:
            return self.output_array


class PFFT(object):

    def __init__(self, comm, shape, axes=None, dtype=float, **kw):
        shape = list(shape)
        assert len(shape) > 0
        assert min(shape) > 0

        if axes is not None:
            axes = list(axes) if np.ndim(axes) else [axes]
            assert 0 < len(axes) <= len(shape)
            assert min(axes) >= -len(shape)
            assert max(axes) < len(shape)
            assert sorted(axes) == sorted(set(axes))
        else:
            axes = list(range(len(shape)))

        dtype = np.dtype(dtype)
        assert dtype.char in 'fdgFDG'

        self.axes = tuple(axes)
        self.subcomm = Subcomm(comm, len(shape)-1)
        self.xfftnseq = []
        self.transfer = []

        axis = self.axes[-1]
        pencilA = Pencil(self.subcomm, shape, axis)
        xfftn = FFT(pencilA.subshape, axis, dtype)

        if np.issubdtype(dtype, np.floating):
            if shape[axis] % 2 == 0:
                shape[axis] = shape[axis]//2 + 1
            else:
                shape[axis] = (shape[axis]+1)//2
            dtype = xfftn.forward.output_array.dtype
            pencilA = Pencil(self.subcomm, shape, axis)

        self.xfftnseq.append(xfftn)
        for axis in reversed(self.axes[:-1]):
            pencilB = pencilA.pencil(axis)
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = FFT(pencilB.subshape, axis, dtype)
            self.xfftnseq.append(xfftn)
            self.transfer.append(transAB)
            pencilA = pencilB

        self.forward = Transform(
            [o.forward for o in self.xfftnseq],
            [o.forward for o in self.transfer])
        self.backward = Transform(
            [o.backward for o in self.xfftnseq[::-1]],
            [o.backward for o in self.transfer[::-1]])

    def destroy(self):
        self.subcomm.destroy()
        for trans in self.transfer:
            trans.destroy()

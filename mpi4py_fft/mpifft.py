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

    def __init__(self, comm, shape, axes=None, dtype=float, padding=1, **kw):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        shape = [int(i*padding) for i in shape]
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

        if isinstance(comm, Subcomm):
            assert len(comm) == len(shape)
            assert comm[axes[-1]].Get_size() == 1
            self.subcomm = comm
        else:
            dims = [0] * len(shape)
            dims[axes[-1]] = 1
            self.subcomm = Subcomm(comm, dims)

        collapse = False # kw.pop('collapse', True)

        if padding > 1+1e-8:
            collapse = False

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
        xfftn = FFT(pencil.subshape, axes, dtype, padding, **kw)
        self.xfftn.append(xfftn)
        self.pencil[0] = pencilA = pencil
        if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
            dtype = xfftn.forward.output_array.dtype
            shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
            pencilA = Pencil(self.subcomm, shape, axes[-1])

        for axes in reversed(self.axes[:-1]):
            pencilB = pencilA.pencil(axes[-1])
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = FFT(pencilB.subshape, axes, dtype, padding, **kw)
            self.xfftn.append(xfftn)
            self.transfer.append(transAB)
            pencilA = pencilB
            if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
                dtype = xfftn.forward.output_array.dtype
                shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
                pencilA = Pencil(pencilB.subcomm, shape, axes[-1])

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

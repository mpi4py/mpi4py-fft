import numpy as np
from mpi4py import MPI  # pylint: disable=unused-import

from .libfft import FFT
from .pencil import Pencil
from .pencil import Subcomm
from .padder import Padder

class Transform(object):

    def __init__(self, xfftn, padder, transfer, pencil):
        assert len(xfftn) == len(transfer) + 1 and len(pencil) == 2
        self._xfftn = tuple(xfftn)
        self._padder = tuple(padder)
        self._transfer = tuple(transfer)
        self._pencil = tuple(pencil)

    @property
    def input_array(self):
        if self._padder and self._xfftn[0].direction == 'FFTW_BACKWARD':
            return self._padder[0].input_array
        return self._xfftn[0].input_array

    @property
    def output_array(self):
        if self._padder and self._xfftn[0].direction == 'FFTW_FORWARD':
            return self._padder[-1].output_array
        return self._xfftn[-1].output_array

    @property
    def input_pencil(self):
        return self._pencil[0]

    @property
    def output_pencil(self):
        return self._pencil[1]

    def __call__(self, input_array=None, output_array=None, **kw):

        if not self._padder:
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
            return self.output_array

        elif self._xfftn[0].direction == 'FFTW_BACKWARD':

            if input_array is not None:
                self.input_array[...] = input_array

            for i in range(len(self._transfer)):
                self._padder[i]()
                self._xfftn[i](**kw)
                arrayA = self._xfftn[i].output_array
                arrayB = self._padder[i+1].input_array
                self._transfer[i](arrayA, arrayB)

            self._padder[-1]()
            self._xfftn[-1](**kw)

            if output_array is not None:
                output_array[...] = self.output_array
                return output_array
            return self.output_array


        elif self._xfftn[0].direction == 'FFTW_FORWARD':

            if input_array is not None:
                self.input_array[...] = input_array

            for i in range(len(self._transfer)):
                self._xfftn[i](**kw)
                self._padder[i]()
                arrayA = self._padder[i].output_array
                arrayB = self._xfftn[i+1].input_array
                self._transfer[i](arrayA, arrayB)

            self._xfftn[-1](**kw)
            self._padder[-1]()

            if output_array is not None:
                output_array[...] = self.output_array
                return output_array
            return self.output_array


class PFFT(object):

    # pylint: disable=too-few-public-methods

    def __init__(self, comm, shape, axes=None, dtype=float, padding=False, **kw):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
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

        slab = False
        if isinstance(comm, Subcomm):
            assert len(comm) == len(shape)
            assert comm[axes[-1]].Get_size() == 1
            self.subcomm = comm
        else:
            if slab:
                dims = [1] * len(shape)
                dims[0] = 0
            else:
                dims = [0] * len(shape)
                dims[axes[-1]] = 1
            self.subcomm = Subcomm(comm, dims)

        self.padding = padding
        if padding is True:
            real = False
            for i, s in enumerate(shape):
                shape[i] = 3*s//2

        collapse = False # kw.pop('collapse', True)
        if collapse and not padding:
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
        self.padder = []
        self.transfer = []
        self.pencil = [None, None]

        axes = self.axes[-1]
        pencil = Pencil(self.subcomm, shape, axes[-1])
        xfftn = FFT(pencil.subshape, axes, dtype, **kw)
        self.xfftn.append(xfftn)
        self.pencil[0] = pencilA = pencil

        if padding:
            if np.issubdtype(dtype, np.floating):
                dtype = xfftn.forward.output_array.dtype
                shape[axes[-1]] = shape[axes[-1]]//3 + 1
                real = True

            else:
                shape[axes[-1]] = 2*shape[axes[-1]]//3

            pencilA = Pencil(self.subcomm, shape, axes[-1])
            padder = Padder(padded_array=xfftn.forward.output_array,
                            trunc_shape=pencilA.subshape, axis=axes[-1],
                            real=real, scale=1.5)
            self.padder.append(padder)

        else:
            if np.issubdtype(dtype, np.floating):
                dtype = xfftn.forward.output_array.dtype
                shape[axes[-1]] = shape[axes[-1]]//2 + 1
                pencilA = Pencil(self.subcomm, shape, axes[-1])

        for axes in reversed(self.axes[:-1]):
            pencilB = pencilA.pencil(axes[-1])
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = FFT(pencilB.subshape, axes, dtype, **kw)
            self.xfftn.append(xfftn)
            if padding:
                trunc_shape = list(xfftn.forward.output_array.shape)
                trunc_shape[axes[-1]] = 2*trunc_shape[axes[-1]]//3
                padder = Padder(padded_array=xfftn.forward.output_array,
                                trunc_shape=tuple(trunc_shape),
                                axis=axes[-1], scale=1.5)
                self.padder.append(padder)

            self.transfer.append(transAB)
            pencilA = pencilB
            if padding:
                shape[axes[-1]] = trunc_shape[axes[-1]]
                pencilA = Pencil(pencilB.subcomm, shape, axes[-1])

        self.pencil[1] = pencilA

        self.forward = Transform(
            [o.forward for o in self.xfftn],
            [o.forward for o in self.padder],
            [o.forward for o in self.transfer],
            self.pencil)
        self.backward = Transform(
            [o.backward for o in self.xfftn[::-1]],
            [o.backward for o in self.padder[::-1]],
            [o.backward for o in self.transfer[::-1]],
            self.pencil[::-1])

    def destroy(self):
        self.subcomm.destroy()
        for trans in self.transfer:
            trans.destroy()

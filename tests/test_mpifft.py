from __future__ import print_function
import numpy as np
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT

abstol = dict(f=2e-3, d=1e-10, g=1e-12)

def allclose(a, b):
    atol = abstol[a.dtype.char.lower()]
    return np.allclose(a, b, rtol=0, atol=atol)

def random_like(array):
    shape = array.shape
    dtype = array.dtype
    return np.random.random(shape).astype(dtype)

def test_mpifft():
    from itertools import product

    comm = MPI.COMM_WORLD
    dims  = (2, 3, 4)
    sizes = (9, 13)
    types = 'fFdD' # + 'gG'

    for typecode in types:
        for dim in dims:
            for shape in product(*([sizes]*dim)):

                if dim < 3:
                    n = min(shape)
                    if typecode in 'fdg':
                        n //=2; n+=1
                    if n < comm.size:
                        continue

                padding = False
                for axes in [None, (-1,), (-2,),
                             (-1,-2,), (-2,-1),
                             (-1,0), (0,-1)]:

                    fft = PFFT(comm, shape, axes=axes, dtype=typecode,
                               padding=padding)

                    if comm.rank == 0:
                        grid = [c.size for c in fft.subcomm]
                        print('grid:{} shape:{} typecode:{} axes:{}'
                              .format(grid, shape, typecode, axes,))

                    assert len(fft.axes) == len(fft.xfftn)
                    assert len(fft.axes) == len(fft.transfer) + 1
                    assert fft.axes[-1] == fft.forward._xfftn[0].axes
                    assert fft.axes[-1] == fft.backward._xfftn[-1].axes
                    assert fft.axes[0] == fft.forward._xfftn[-1].axes
                    assert fft.axes[0] == fft.backward._xfftn[0].axes
                    assert (fft.forward.input_pencil.subshape ==
                            fft.forward.input_array.shape)
                    assert (fft.forward.output_pencil.subshape ==
                            fft.forward.output_array.shape)
                    assert (fft.backward.input_pencil.subshape ==
                            fft.backward.input_array.shape)
                    assert (fft.backward.output_pencil.subshape ==
                            fft.backward.output_array.shape)
                    ax = -1 if axes is None else axes[-1]
                    assert fft.forward.input_pencil.substart[ax] == 0
                    assert fft.backward.output_pencil.substart[ax] == 0
                    ax = 0 if axes is None else axes[0]
                    assert fft.forward.output_pencil.substart[ax] == 0
                    assert fft.backward.input_pencil.substart[ax] == 0

                    U = random_like(fft.forward.input_array)

                    if 1:
                        F = fft.forward(U)
                        V = fft.backward(F)
                        assert allclose(V, U)
                    else:
                        fft.forward.input_array[...] = U
                        fft.forward()
                        fft.backward()
                        V = fft.backward.output_array
                        assert allclose(V, U)

                    fft.destroy()

                padding = [1.5]*len(shape)
                for axes in [None, (-1,), (-2,),
                             (-1,-2,), (-2,-1),
                             (-1,0), (0,-1)]:

                    fft = PFFT(comm, shape, axes=axes, dtype=typecode,
                               padding=padding)

                    if comm.rank == 0:
                        grid = [c.size for c in fft.subcomm]
                        print('grid:{} shape:{} typecode:{} axes:{}'
                              .format(grid, shape, typecode, axes,))

                    assert len(fft.axes) == len(fft.xfftn)
                    assert len(fft.axes) == len(fft.transfer) + 1
                    assert fft.axes[-1] == fft.forward._xfftn[0].axes
                    assert fft.axes[-1] == fft.backward._xfftn[-1].axes
                    assert fft.axes[0] == fft.forward._xfftn[-1].axes
                    assert fft.axes[0] == fft.backward._xfftn[0].axes
                    assert (fft.forward.input_pencil.subshape ==
                            fft.forward.input_array.shape)
                    assert (fft.forward.output_pencil.subshape ==
                            fft.forward.output_array.shape)
                    assert (fft.backward.input_pencil.subshape ==
                            fft.backward.input_array.shape)
                    assert (fft.backward.output_pencil.subshape ==
                            fft.backward.output_array.shape)
                    ax = -1 if axes is None else axes[-1]
                    assert fft.forward.input_pencil.substart[ax] == 0
                    assert fft.backward.output_pencil.substart[ax] == 0
                    ax = 0 if axes is None else axes[0]
                    assert fft.forward.output_pencil.substart[ax] == 0
                    assert fft.backward.input_pencil.substart[ax] == 0

                    U = random_like(fft.forward.input_array)
                    F = fft.forward(U)

                    if 1:
                        Fc = F.copy()
                        V = fft.backward(F)
                        F = fft.forward(V)
                        assert allclose(F, Fc)
                    else:
                        fft.backward.input_array[...] = F
                        fft.backward()
                        fft.forward()
                        V = fft.forward.output_array
                        assert allclose(F, V)

                    fft.destroy()

if __name__ == '__main__':
    test_mpifft()


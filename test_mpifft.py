from __future__ import print_function
import numpy as np
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT

abstol = dict(f=1e-6, d=1e-14, g=1e-15)

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
    sizes = (10, 13, 16)
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

                for axes in [None, (-1,), (-2,),
                             (-1,-2,), (-2,-1),
                             (-1,0), (0,-1)]:
                    if (axes is None and
                        shape[-1] % 2 and
                        typecode in 'fdg'):
                        continue
                    if (axes is not None and
                        shape[axes[-1]] % 2 and
                        typecode in 'fdg'):
                        continue

                    fft = PFFT(comm, shape, axes=axes, dtype=typecode)

                    if comm.rank == 0:
                        grid = [c.size for c in fft.subcomm]
                        print('grid:{} shape:{} typecode:{} axes:{}'
                              .format(grid, shape, typecode, axes,))

                    U = random_like(fft.forward.input_array)

                    if 1:
                        F = fft.forward(U)
                        V = fft.backward(F)
                        assert allclose(U, V)
                    else:
                        fft.forward.input_array[...] = U
                        fft.forward()
                        fft.backward()
                        V = fft.backward.output_array
                        assert allclose(U, V)

                    fft.destroy()


if __name__ == '__main__':
    test_mpifft()

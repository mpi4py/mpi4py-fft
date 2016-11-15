from __future__ import print_function
import numpy as np
from mpi4py_fft.libfft import FFT

abstol = dict(f=1e-6, d=1e-14, g=1e-15)

def allclose(a, b):
    atol = abstol[a.dtype.char.lower()]
    return np.allclose(a, b, rtol=0, atol=atol)

def test_libfft():
    from itertools import product

    dims  = (1, 2, 3, 4)
    sizes = (7, 8, 9)
    types = 'fdgFDG'
    types = 'fF'

    for typecode in types:
        for dim in dims:
            for shape in product(*([sizes]*dim)):
                allaxes = tuple(reversed(range(dim)))
                for i in range(dim):
                    for j in range(i+1, dim):

                        axes = allaxes[i:j]
                        if (shape[axes[-1]] % 2 and
                            typecode in 'fdg'):
                            continue

                        #print shape, axes, typecode
                        fft = FFT(shape, axes, dtype=typecode)
                        A = fft.forward.input_array
                        B = fft.forward.output_array

                        A[...] = np.random.random(shape).astype(typecode)
                        X = A.copy()

                        B.fill(0)
                        fft.forward(A, B)

                        A.fill(0)
                        fft.backward(B, A)

                        assert allclose(A, X)


if __name__ == '__main__':
    test_libfft()

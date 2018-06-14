from __future__ import print_function
from time import time
import numpy as np
from mpi4py_fft.libfft import FFT

abstol = dict(f=5e-5, d=1e-14, g=1e-15)

def allclose(a, b):
    atol = abstol[a.dtype.char.lower()]
    return np.allclose(a, b, rtol=0, atol=atol)

def test_libfft():
    from itertools import product

    dims  = (1, 2, 3, 4)
    sizes = (7, 8, 9)
    types = 'fdgFDG'

    for use_pyfftw in (True, False):
        t0 = 0
        for typecode in types:
            for dim in dims:
                for shape in product(*([sizes]*dim)):
                    allaxes = tuple(reversed(range(dim)))
                    for i in range(dim):
                        for j in range(i+1, dim):

                            axes = allaxes[i:j]

                            #print(shape, axes, typecode)
                            fft = FFT(shape, axes, dtype=typecode,
                                      use_pyfftw=use_pyfftw)
                            A = fft.forward.input_array
                            B = fft.forward.output_array

                            A[...] = np.random.random(A.shape).astype(typecode)
                            X = A.copy()

                            B.fill(0)
                            t0 -= time()
                            B = fft.forward(A, B)
                            t0 += time()

                            A.fill(0)
                            t0 -= time()
                            A = fft.backward(B, A)
                            t0 += time()
                            assert allclose(A, X)
        print(use_pyfftw, t0)
    # Padding is different because the physical space is padded and as such
    # difficult to initialize. We solve this problem by making one extra
    # transform
    for use_pyfftw in (True, False):
        for padding in (1.5, 2.0):
            for typecode in types:
                for dim in dims:
                    for shape in product(*([sizes]*dim)):
                        allaxes = tuple(reversed(range(dim)))
                        for i in range(dim):
                            axis = allaxes[i]
                            shape = list(shape)
                            shape[axis] = int(shape[axis]*padding)

                            #print(shape, axis, typecode)
                            fft = FFT(shape, axis, dtype=typecode,
                                      padding=padding, use_pyfftw=use_pyfftw)
                            A = fft.forward.input_array
                            B = fft.forward.output_array

                            A[...] = np.random.random(A.shape).astype(typecode)

                            B.fill(0)
                            B = fft.forward(A, B)
                            X = B.copy()

                            A.fill(0)
                            A = fft.backward(B, A)

                            B.fill(0)
                            B = fft.forward(A, B)
                            assert allclose(B, X)


if __name__ == '__main__':
    test_libfft()

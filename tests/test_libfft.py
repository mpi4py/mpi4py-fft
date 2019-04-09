from __future__ import print_function
from time import time
import importlib
import functools
import numpy as np
from mpi4py_fft import fftw
from mpi4py_fft.libfft import FFT

has_backend = {'fftw': True}
for backend in ('pyfftw', 'mkl_fft', 'scipy', 'numpy'):
    has_backend[backend] = True
    try:
        importlib.import_module(backend)
    except ImportError:
        has_backend[backend] = False

abstol = dict(f=5e-5, d=1e-14, g=1e-14)

def allclose(a, b):
    atol = abstol[a.dtype.char.lower()]
    return np.allclose(a, b, rtol=0, atol=atol)

def test_libfft():
    from itertools import product

    dims = (1, 2, 3)
    sizes = (7, 8, 9)
    types = ''
    for t in 'fd':
        if fftw.get_fftw_lib(t):
            types += t+t.upper()

    for backend in ('pyfftw', 'mkl_fft', 'scipy', 'numpy', 'fftw'):
        if has_backend[backend] is False:
            continue
        t0 = 0
        for typecode in types:
            for dim in dims:
                for shape in product(*([sizes]*dim)):
                    allaxes = tuple(reversed(range(dim)))
                    for i in range(dim):
                        for j in range(i+1, dim):
                            for axes in (None, allaxes[i:j]):
                                #print(shape, axes, typecode)
                                fft = FFT(shape, axes, dtype=typecode, backend=backend,
                                          planner_effort='FFTW_ESTIMATE')
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
        print('backend: ', backend, t0)
    # Padding is different because the physical space is padded and as such
    # difficult to initialize. We solve this problem by making one extra
    # transform
    for backend in ('pyfftw', 'mkl_fft', 'scipy', 'numpy', 'fftw'):
        if has_backend[backend] is False:
            continue
        for padding in (1.5, 2.0):
            for typecode in types:
                for dim in dims:
                    for shape in product(*([sizes]*dim)):
                        allaxes = tuple(reversed(range(dim)))
                        for i in range(dim):
                            axis = allaxes[i]
                            axis -= len(shape)
                            shape = list(shape)
                            shape[axis] = int(shape[axis]*padding)

                            #print(shape, axis, typecode, backend)
                            fft = FFT(shape, axis, dtype=typecode, backend=backend,
                                      padding=padding, planner_effort='FFTW_ESTIMATE')
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
                            assert allclose(B, X), np.linalg.norm(B-X)

    for backend in ('pyfftw', 'mkl_fft', 'scipy', 'numpy', 'fftw'):
        if has_backend[backend] is False:
            continue

        if backend == 'fftw':
            dctn = functools.partial(fftw.dctn, type=3)
            idctn = functools.partial(fftw.idctn, type=3)
            transforms = {(1,): (dctn, idctn),
                          (0, 1): (dctn, idctn)}
        elif backend == 'pyfftw':
            import pyfftw
            transforms = {(1,): (pyfftw.builders.rfftn, pyfftw.builders.irfftn),
                          (0, 1): (pyfftw.builders.rfftn, pyfftw.builders.irfftn)}
        elif backend == 'numpy':
            transforms = {(1,): (np.fft.rfftn, np.fft.irfftn),
                          (0, 1): (np.fft.rfftn, np.fft.irfftn)}
        elif backend == 'mkl_fft':
            import mkl_fft
            transforms = {(1,): (mkl_fft._numpy_fft.rfftn, mkl_fft._numpy_fft.irfftn),
                          (0, 1): (mkl_fft._numpy_fft.rfftn, mkl_fft._numpy_fft.irfftn)}
        elif backend == 'scipy':
            from scipy.fftpack import fftn, ifftn
            transforms = {(1,): (fftn, ifftn),
                          (0, 1): (fftn, ifftn)}

        for axis in ((1,), (0, 1)):
            fft = FFT(shape, axis, backend=backend, transforms=transforms)
            A = fft.forward.input_array
            B = fft.forward.output_array
            A[...] = np.random.random(A.shape)
            X = A.copy()
            B.fill(0)
            B = fft.forward(A, B)
            A.fill(0)
            A = fft.backward(B, A)
            assert allclose(A, X)

if __name__ == '__main__':
    test_libfft()

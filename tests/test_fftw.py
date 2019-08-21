from __future__ import print_function
from time import time
import numpy as np
from scipy.fftpack import dctn as scipy_dctn
from scipy.fftpack import dstn as scipy_dstn
import scipy.fftpack # pylint: disable=unused-import
from mpi4py_fft import fftw

has_pyfftw = True
try:
    import pyfftw
except ImportError:
    has_pyfftw = False

abstol = dict(f=5e-4, d=1e-12, g=1e-14)

kinds = {'dst4': fftw.FFTW_RODFT11, # no scipy to compare with
         'dct4': fftw.FFTW_REDFT11, # no scipy to compare with
         'dst3': fftw.FFTW_RODFT01,
         'dct3': fftw.FFTW_REDFT01,
         'dct2': fftw.FFTW_REDFT10,
         'dst2': fftw.FFTW_RODFT10,
         'dct1': fftw.FFTW_REDFT00,
         'dst1': fftw.FFTW_RODFT00}

rkinds = {val: key for key, val in kinds.items()}

def allclose(a, b):
    atol = abstol[a.dtype.char.lower()]
    return np.allclose(a, b, rtol=0, atol=atol)

def test_fftw():
    from itertools import product

    dims = (1, 2, 3)
    sizes = (7, 8, 10)
    types = ''
    for t in 'fdg':
        if fftw.get_fftw_lib(t):
            types += t
    fflags = (fftw.FFTW_ESTIMATE, fftw.FFTW_DESTROY_INPUT)
    iflags = (fftw.FFTW_ESTIMATE, fftw.FFTW_DESTROY_INPUT)

    for threads in (1, 2):
        for typecode in types:
            for dim in dims:
                for shape in product(*([sizes]*dim)):
                    allaxes = tuple(reversed(range(dim)))
                    for i in range(dim):
                        for j in range(i+1, dim):
                            axes = allaxes[i:j]
                            #print(shape, axes, typecode, threads)
                            # r2c - c2r
                            input_array = fftw.aligned(shape, dtype=typecode)
                            outshape = list(shape)
                            outshape[axes[-1]] = shape[axes[-1]]//2+1
                            output_array = fftw.aligned(outshape, dtype=typecode.upper())
                            oa = output_array if typecode == 'd' else None # Test for both types of signature
                            rfftn = fftw.rfftn(input_array, None, axes, threads, fflags, output_array=oa)
                            A = np.random.random(shape).astype(typecode)
                            input_array[:] = A
                            B = rfftn()
                            assert id(B) == id(rfftn.output_array)
                            if has_pyfftw:
                                B2 = pyfftw.interfaces.numpy_fft.rfftn(input_array, axes=axes)
                                assert allclose(B, B2), np.linalg.norm(B-B2)
                            ia = input_array if typecode == 'd' else None
                            sa = np.take(input_array.shape, axes) if shape[axes[-1]] % 2 == 1 else None
                            irfftn = fftw.irfftn(output_array, sa, axes, threads, iflags, output_array=ia)
                            irfftn.input_array[...] = B
                            A2 = irfftn(normalize=True)
                            assert allclose(A, A2), np.linalg.norm(A-A2)
                            hfftn = fftw.hfftn(output_array, sa, axes, threads, fflags, output_array=ia)
                            hfftn.input_array[...] = B
                            AC = hfftn().copy()
                            ihfftn = fftw.ihfftn(input_array, None, axes, threads, iflags, output_array=oa)
                            A2 = ihfftn(AC, implicit=False, normalize=True)
                            assert allclose(A2, B), print(np.linalg.norm(A2-B))

                            # c2c
                            input_array = fftw.aligned(shape, dtype=typecode.upper())
                            output_array = fftw.aligned_like(input_array)
                            oa = output_array if typecode=='d' else None
                            fftn = fftw.fftn(input_array, None, axes, threads, fflags, output_array=oa)
                            C = np.random.random(shape).astype(typecode.upper())
                            fftn.input_array[...] = C
                            D = fftn().copy()
                            ifftn = fftw.ifftn(input_array, None, axes, threads, iflags, output_array=oa)
                            ifftn.input_array[...] = D
                            C2 = ifftn(normalize=True)
                            assert allclose(C, C2), np.linalg.norm(C-C2)
                            if has_pyfftw:
                                D2 = pyfftw.interfaces.numpy_fft.fftn(C, axes=axes)
                                assert allclose(D, D2), np.linalg.norm(D-D2)

                            # r2r
                            input_array = fftw.aligned(shape, dtype=typecode)
                            output_array = fftw.aligned_like(input_array)
                            oa = output_array if typecode =='d' else None
                            for type in (1, 2, 3, 4):
                                dct = fftw.dctn(input_array, None, axes, type, threads, fflags, output_array=oa)
                                B = dct(A).copy()
                                idct = fftw.idctn(input_array, None, axes, type, threads, iflags, output_array=oa)
                                A2 = idct(B, implicit=True, normalize=True)
                                assert allclose(A, A2), np.linalg.norm(A-A2)
                                if typecode is not 'g' and not type is 4:
                                    B2 = scipy_dctn(A, axes=axes, type=type)
                                    assert allclose(B, B2), np.linalg.norm(B-B2)

                                dst = fftw.dstn(input_array, None, axes, type, threads, fflags, output_array=oa)
                                B = dst(A).copy()
                                idst = fftw.idstn(input_array, None, axes, type, threads, iflags, output_array=oa)
                                A2 = idst(B, implicit=True, normalize=True)
                                assert allclose(A, A2), np.linalg.norm(A-A2)
                                if typecode is not 'g' and not type is 4:
                                    B2 = scipy_dstn(A, axes=axes, type=type)
                                    assert allclose(B, B2), np.linalg.norm(B-B2)

                            # Different r2r transforms along all axes. Just pick
                            # any naxes transforms and compare with scipy
                            naxes = len(axes)
                            kds = np.random.randint(3, 11, size=naxes) # get naxes random transforms
                            tsf = [rkinds[k] for k in kds]
                            T = fftw.get_planned_FFT(input_array, input_array.copy(), axes=axes,
                                                     kind=kds, threads=threads, flags=fflags)
                            C = T(A)
                            TI = fftw.get_planned_FFT(input_array.copy(), input_array.copy(), axes=axes,
                                                      kind=list([fftw.inverse[kd] for kd in kds]),
                                                      threads=threads, flags=iflags)

                            C2 = TI(C)
                            M = fftw.get_normalization(kds, input_array.shape, axes)
                            assert allclose(C2*M, A)
                            # Test vs scipy for transforms available in scipy
                            if typecode is not 'g' and not any(f in kds for f in (fftw.FFTW_RODFT11, fftw.FFTW_REDFT11)):
                                for m, ts in enumerate(tsf):
                                    A = eval('scipy.fftpack.'+ts[:-1])(A, axis=axes[m], type=int(ts[-1]))
                                assert allclose(C, A), np.linalg.norm(C-A)

def test_wisdom():
    # Test a simple export/import call
    fftw.export_wisdom('newwisdom.dat')
    fftw.import_wisdom('newwisdom.dat')
    fftw.forget_wisdom()

def test_timelimit():
    limit = 0.01
    input_array = fftw.aligned((128, 128), dtype='d')
    t0 = time()
    fftw.rfftn(input_array, flags=fftw.FFTW_PATIENT)
    t1 = time()-t0
    fftw.forget_wisdom()
    fftw.set_timelimit(limit)
    t0 = time()
    fftw.rfftn(input_array, flags=fftw.FFTW_PATIENT)
    t2 = time()-t0
    assert t1 > t2
    assert abs(t2-limit) < limit, print(abs(t2-limit), limit)
    fftw.cleanup()

if __name__ == '__main__':
    test_fftw()
    test_wisdom()
    test_timelimit()

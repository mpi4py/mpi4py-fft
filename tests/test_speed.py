import numpy as np
import pyfftw
from scipy.fftpack import dct, dst, dctn, dstn
from mpi4py_fft import fftw
from time import time

try:
    fftw.xfftn.import_wisdom('wisdom.dat')
except AssertionError:
    pass

N = (50, 60, 71)
loops = 20
axis = 1
threads = 1
flags = (fftw.FFTW_MEASURE, fftw.FFTW_DESTROY_INPUT)

# Transform complex to complex
A = pyfftw.byte_align(np.random.random(N).astype('D'))

input_array = np.zeros_like(A)
output_array = np.zeros_like(A)

ptime = [[], []]
ftime = [[], []]
for axis in (0, 1, 2):

    # pyfftw
    fft = pyfftw.builders.fftn(input_array, axes=(axis,), threads=threads, overwrite_input=True)
    t0 = time()
    for i in range(loops):
        C = fft(A)
    ptime[0].append(time()-t0)

    # mine
    fft = fftw.fftn(input_array, output_array, (axis,), threads, flags)

    t0 = time()
    for i in range(loops):
        C2 = fft(A)
    ftime[0].append(time()-t0)
    assert np.allclose(C, C2)

    # pyfftw
    ifft = pyfftw.builders.ifftn(output_array, axes=(axis,), threads=threads, overwrite_input=True)
    t0 = time()
    for i in range(loops):
        B = ifft(C)
    ptime[1].append(time()-t0)

    # mine
    ifft = fftw.ifftn(output_array, input_array, (axis,), threads, flags)

    t0 = time()
    for i in range(loops):
        B2 = ifft(C, normalize_idft=True)
    ftime[1].append(time()-t0)
    assert np.allclose(B, B2), np.linalg.norm(B-B2)


print("Timing forward transform axes 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ptime[0]))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ftime[0]))
print("Timing backward transform axes 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ptime[1]))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ftime[1]))


# Transform complex to complex
A = pyfftw.byte_align(np.random.random(N).astype('d'))

input_array = np.zeros_like(A)

ptime = [[], []]
ftime = [[], []]
for axis in (0, 1, 2):

    # pyfftw
    rfft = pyfftw.builders.rfftn(input_array, axes=(axis,), threads=threads)
    t0 = time()
    for i in range(loops):
        C = rfft(A)
    ptime[0].append(time()-t0)

    # mine
    rfft = fftw.rfftn(input_array, C.copy(), (axis,), threads, flags)
    t0 = time()
    for i in range(loops):
        C2 = rfft(A)
    ftime[0].append(time()-t0)
    assert np.allclose(C, C2)

    # pyfftw
    irfft = pyfftw.builders.irfftn(C.copy(), axes=(axis,), s=(input_array.shape[axis],), threads=threads)
    t0 = time()
    for i in range(loops):
        C2[:] = C       # Because irfft is overwriting input
        D = irfft(C2)
    ptime[1].append(time()-t0)

    # mine
    irfft = fftw.irfftn(C.copy(), input_array, (axis,), threads, flags)
    t0 = time()
    for i in range(loops):
        C2[:] = C
        D2 = irfft(C2, normalize_idft=True)
    ftime[1].append(time()-t0)
    assert np.allclose(D, D2), np.linalg.norm(D-D2)

print("Timing real forward transform axes 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ptime[0]))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ftime[0]))
print("Timing real backward transform axes 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ptime[1]))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ftime[1]))

fftw.xfftn.export_wisdom('wisdom.dat')

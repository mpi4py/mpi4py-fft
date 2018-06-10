import numpy as np
import pyfftw
import scipy.fftpack as sp
from mpi4py_fft import fftw
from time import time

try:
    fftw.xfftn.import_wisdom('wisdom.dat')
except AssertionError:
    pass

N = (64, 64, 64)
loops = 50
axis = 1
threads = 1
implicit = True
flags = (fftw.FFTW_PATIENT, fftw.FFTW_DESTROY_INPUT)

def empty_aligned(shape, n=32, dtype=np.dtype('d', align=True)):
    a = np.empty(np.prod(shape)*dtype.itemsize+n, dtype=np.dtype('uint8'))
    offset = a.ctypes.data % n
    offset = 0 if offset == 0 else (n - offset)
    return np.frombuffer(a[offset:offset+n].data, dtype=dtype).reshape(shape)

# Transform complex to complex
#A = pyfftw.byte_align(np.random.random(N).astype('D'))
A = np.random.random(N).astype(np.dtype('D', align=True))

input_array = np.zeros_like(A)
output_array = np.zeros_like(A)

ptime = [[], []]
ftime = [[], []]
stime = [[], []]
for axis in ((1, 2), 0, 1, 2):

    axes = axis if np.ndim(axis) else [axis]

    # pyfftw
    fft = pyfftw.builders.fftn(input_array, axes=axes, threads=threads,
                               overwrite_input=True)
    t0 = time()
    for i in range(loops):
        C = fft(A)
    ptime[0].append(time()-t0)

    # us
    fft = fftw.fftn(input_array, output_array, axes, threads, flags)
    t0 = time()
    for i in range(loops):
        C2 = fft(A, implicit=implicit)
    ftime[0].append(time()-t0)
    assert np.allclose(C, C2)

    # scipy
    C3 = sp.fftn(A, axes=axes) # scipy is caching, so call once before
    t0 = time()
    for i in range(loops):
        C3 = sp.fftn(A, axes=axes)
    stime[0].append(time()-t0)

    # pyfftw
    ifft = pyfftw.builders.ifftn(output_array, axes=axes, threads=threads,
                                 overwrite_input=True)
    t0 = time()
    for i in range(loops):
        B = ifft(C, normalise_idft=True)
    ptime[1].append(time()-t0)

    # us
    ifft = fftw.ifftn(output_array, input_array, axes, threads, flags)
    t0 = time()
    for i in range(loops):
        B2 = ifft(C, normalize_idft=True, implicit=implicit)
    ftime[1].append(time()-t0)
    assert np.allclose(B, B2), np.linalg.norm(B-B2)

    # scipy
    B3 = sp.ifftn(C, axes=axes) # scipy is caching, so call once before
    t0 = time()
    for i in range(loops):
        B3 = sp.ifftn(C, axes=axes)
    stime[1].append(time()-t0)


print("Timing forward transform axes (1, 2), 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*ptime[0]))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*ftime[0]))
print("scipy   {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*stime[0]))
print("Timing backward transform axes (1, 2), 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*ptime[1]))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*ftime[1]))
print("scipy   {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*stime[1]))


# Transform real to complex
# Not scipy because they do not have rfftn

#A = pyfftw.byte_align(np.random.random(N).astype('d'))
A = np.random.random(N).astype(np.dtype('d', align=True))

input_array = np.zeros_like(A)

ptime = [[], []]
ftime = [[], []]
for axis in ((1, 2), 0, 1, 2):

    axes = axis if np.ndim(axis) else [axis]

    # pyfftw
    rfft = pyfftw.builders.rfftn(input_array, axes=axes, threads=threads)
    t0 = time()
    for i in range(loops):
        C = rfft(A)
    ptime[0].append(time()-t0)

    # us
    rfft = fftw.rfftn(input_array, C.copy(), axes, threads, flags)
    t0 = time()
    for i in range(loops):
        C2 = rfft(A, implicit=implicit)
    ftime[0].append(time()-t0)
    assert np.allclose(C, C2)

    # pyfftw
    irfft = pyfftw.builders.irfftn(C.copy(), s=np.take(input_array.shape, axes),
                                   axes=axes, threads=threads)
    t0 = time()
    for i in range(loops):
        C2[:] = C       # Because irfft is overwriting input
        D = irfft(C2, normalise_idft=True)
    ptime[1].append(time()-t0)

    # us
    irfft = fftw.irfftn(C.copy(), input_array, axes, threads, flags)
    t0 = time()
    for i in range(loops):
        C2[:] = C
        D2 = irfft(C2, normalize_idft=True, implicit=implicit)
    ftime[1].append(time()-t0)
    assert np.allclose(D, D2), np.linalg.norm(D-D2)

print("Timing real forward transform axes (1, 2), 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*ptime[0]))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*ftime[0]))
print("Timing real backward transform axes (1, 2), 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*ptime[1]))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e} {3:2.4e}".format(*ftime[1]))

fftw.xfftn.export_wisdom('wisdom.dat')

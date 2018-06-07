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
axis = 1
threads = 1
flags = (fftw.FFTW_MEASURE, fftw.FFTW_DESTROY_INPUT)

# Transform complex to complex
A = pyfftw.byte_align(np.random.random(N).astype('D'))

input_array = np.zeros_like(A)
output_array = np.zeros_like(A)

ptime = []
ftime = []
for axis in (0, 1, 2):

    # pyfftw
    fft = pyfftw.builders.fftn(input_array, axes=(axis,), threads=threads, overwrite_input=True)
    t0 = time()
    for i in range(100):
        C = fft(A)
    ptime.append(time()-t0)

    # mine
    fft = fftw.fftn(input_array, output_array, (axis,), threads, flags)

    t0 = time()
    for i in range(100):
        C2 = fft(A)
    ftime.append(time()-t0)
    assert np.allclose(C, C2)

print("Timing forward transform axes 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ptime))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ftime))


# Transform complex to complex
A = pyfftw.byte_align(np.random.random(N).astype('d'))

input_array = np.zeros_like(A)

ptime = []
ftime = []
for axis in (0, 1, 2):

    # pyfftw
    fft = pyfftw.builders.rfftn(input_array, axes=(axis,), threads=threads)
    t0 = time()
    for i in range(100):
        C = fft(A)
    ptime.append(time()-t0)

    # mine
    fft = fftw.rfftn(input_array, C.copy(), (axis,), threads, flags)

    t0 = time()
    for i in range(100):
        C2 = fft(A)
    ftime.append(time()-t0)
    assert np.allclose(C, C2)

print("Timing backward transform axes 0, 1, 2")
print("pyfftw  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ptime))
print("mpi4py  {0:2.4e}  {1:2.4e}  {2:2.4e}".format(*ftime))


#from IPython import embed; embed()
fftw.xfftn.export_wisdom('wisdom.dat')

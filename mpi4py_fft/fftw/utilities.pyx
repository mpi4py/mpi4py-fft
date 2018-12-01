#cython: language_level=3

cimport numpy as np
import numpy as np
from libc.stdint cimport intptr_t

cpdef enum:
    FFTW_FORWARD  = -1
    FFTW_R2HC     = 0
    FFTW_BACKWARD = 1
    FFTW_HC2R     = 1
    FFTW_DHT      = 2
    FFTW_REDFT00  = 3
    FFTW_REDFT01  = 4
    FFTW_REDFT10  = 5
    FFTW_REDFT11  = 6
    FFTW_RODFT00  = 7
    FFTW_RODFT01  = 8
    FFTW_RODFT10  = 9
    FFTW_RODFT11  = 10

cpdef enum:
    C2C_FORWARD = -1
    C2C_BACKWARD = 1
    R2C = -2
    C2R = 2

cpdef enum:
    FFTW_MEASURE = 0
    FFTW_DESTROY_INPUT = 1
    FFTW_UNALIGNED = 2
    FFTW_CONSERVE_MEMORY = 4
    FFTW_EXHAUSTIVE = 8
    FFTW_PRESERVE_INPUT = 16
    FFTW_PATIENT = 32
    FFTW_ESTIMATE = 64
    FFTW_WISDOM_ONLY = 2097152

cpdef int get_alignment(array):
   """Return alignment assuming highest allowed is 32

    Parameters
    ----------
    array : array
    """
   cdef int i, n
   cdef intptr_t addr = <intptr_t>np.PyArray_DATA(array)
   for i in range(5, -1, -1):
       n = 1 << i
       if addr % n == 0:
           break
   return n

cpdef aligned(shape, n=32, dtype=np.dtype('d'), fill=None):
    """Returned array with byte-alignment according to n

    Parameters
    ----------
    shape : sequence of ints
        The shape of the array to be created
    n : int, optional
        The chosen byte-alignment
    dtype : np.dtype, optional
        The type of the returned array
    fill : None or number, optional
        If number then fill returned array with this number, otherwise return
        empty array

    Returns
    -------
    array
        byte-aligned array

    """
    dtype = np.dtype(dtype)
    M = np.prod(shape)*dtype.itemsize
    a = np.empty(M+n, dtype=np.dtype('uint8'))
    offset = a.ctypes.data % n
    offset = 0 if offset == 0 else (n - offset)
    b = np.frombuffer(a[offset:(offset+M)].data, dtype=dtype).reshape(shape)
    if fill is not None:
        assert isinstance(fill, int)
        b[...] = fill
    return b

cpdef aligned_like(z, fill=None):
    """Return array with byte-alignment, shape and type like array z

    Parameters
    ----------
    z : array
        An array with shape and type we want to recreate
    fill : None or number, optional
        If number then fill returned array with this number, otherwise return
        empty array

    Returns
    -------
    array
        byte-aligned array

    """
    n = get_alignment(z)
    return aligned(z.shape, n=n, dtype=z.dtype, fill=fill)

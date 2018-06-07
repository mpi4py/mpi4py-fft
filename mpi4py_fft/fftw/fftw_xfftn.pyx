cimport fftw_xfftn
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cpdef int export_wisdom(const char *filename):
    return fftw_export_wisdom_to_filename(filename)

cpdef int import_wisdom(const char *filename):
    return fftw_import_wisdom_from_filename(filename)

cdef fftw_plan* _fftw_planxfftn(int      ndims,
                                int      sizesA[],
                                void     *arrayA,
                                int      sizesB[],
                                void     *arrayB,
                                int      naxes,
                                int      axes[],
                                int      kind[],
                                unsigned flags):
    return <fftw_plan *>fftw_planxfftn(ndims, sizesA, <void *> arrayA,
                                       sizesB, <void *> arrayB, naxes, 
                                       axes, kind, flags)

cdef class FFT:
    """
    Unified class for performing FFTs on multidimensional arrays

    The class can be used for any type of transform defined in the user manual
    of `FFTW <http://www.fftw.org/fftw3_doc>`_. 

    Parameters
    ----------
    input_array : array
        real or complex input array
    output_array : array
        real or complex output array
    axes : tuple if ints
        The axes to transform over, starting from the last
    kind : int or tuple of ints
        Any one of
        - FFTW_FORWARD (-1)
        - FFTW_R2HC (0)
        - FFTW_BACKWARD (1)
        - FFTW_HC2R (1)
        - FFTW_DHT (2)
        - FFTW_REDFT00 (3)
        - FFTW_REDFT01 (4)
        - FFTW_REDFT10 (5)
        - FFTW_REDFT11 (6)
        - FFTW_RODFT00 (7)
        - FFTW_RODFT01 (8)
        - FFTW_RODFT10 (9)
        - FFTW_RODFT11 (10)
    threads : int
        Number of threads to use in transforms
    flags : int or tuple of ints
        Any one of, but not necessarily for all transforms or all combinations
        - FFTW_MEASURE (0)
        - FFTW_DESTROY_INPUT (1)
        - FFTW_UNALIGNED (2)
        - FFTW_CONSERVE_MEMORY (4)
        - FFTW_EXHAUSTIVE (8)
        - FFTW_PRESERVE_INPUT (16)
        - FFTW_PATIENT (32)
        - FFTW_ESTIMATE (64)
        - FFTW_WISDOM_ONLY (2097152)
    normalization : int
        Normalization factor

    """
    cdef void *_plan
    cdef np.ndarray _input_array
    cdef np.ndarray _output_array
    cdef fftw_real _M

    def __cinit__(self, input_array, output_array, axes=(-1,), 
                  kind=FFTW_FORWARD, int threads=1,
                  flags=FFTW_MEASURE, int normalization=1):
        cdef int ndims = len(input_array.shape)
        cdef int naxes = len(axes)
        cdef int flag, i
        cdef unsigned allflags
        cdef int *szA = <int *> malloc(ndims * sizeof(int))
        cdef int *szB = <int *> malloc(ndims * sizeof(int))
        cdef int *axs = <int *> malloc(naxes * sizeof(int))
        cdef int *knd = <int *> malloc(naxes * sizeof(int))
        cdef void *_in = <void *>np.PyArray_DATA(input_array)
        cdef void *_out = <void *>np.PyArray_DATA(output_array)        

        fftw_plan_with_nthreads(threads)
        flags = [flags] if isinstance(flags, int) else flags
        kind = [kind] if isinstance(kind, int) else kind
        axes = list(axes)
        for i in range(naxes):
            if axes[i] < 0:
                axes[i] = axes[i] + ndims
        
        allflags = flags[0]
        for flag in flags[1:]:
            allflags |= flag

        self._input_array = input_array 
        self._output_array = output_array
        self._M = 1./normalization
        for i in range(ndims):
            szA[i] = input_array.shape[i]
            szB[i] = output_array.shape[i]
        for i in range(naxes):
            axs[i] = axes[i]
        for i in range(len(kind)):
            knd[i] = kind[i]

        self._plan = _fftw_planxfftn(ndims, szA, _in, szB, _out, naxes, axs,
                                     knd, allflags)
        free(szA)
        free(szB)
        free(axs)
        free(knd)

    def __call__(self, input_array=None, output_array=None, **kw):
        if input_array is not None:
            self._input_array[...] = input_array
        fftw_execute(<fftw_plan *>self._plan)
        if kw.get('normalize_idft', False):
            self._output_array *= self._M
        if output_array is not None:
            output_array[...] = self._output_array
            return output_array
        return self._output_array

    def __dealloc__(self):
        self.destroy()

    def destroy(self):
        fftw_destroy_plan(<fftw_plan *>self._plan)

    @property
    def input_array(self):
        return self._input_array

    @property
    def output_array(self):
        return self._output_array


cimport fftw_xfftn
#cython: language_level=3
cimport numpy as np
from .utilities import *
import numpy as np
from libc.stdint cimport intptr_t
from libc.stdlib cimport malloc, free

cpdef int alignment_of(input_array):
    cdef np.ndarray _input_array = input_array
    return fftw_alignment_of(<fftw_real *>np.PyArray_DATA(_input_array))

cpdef int export_wisdom(const char *filename):
    return fftw_export_wisdom_to_filename(filename)

cpdef int import_wisdom(const char *filename):
    return fftw_import_wisdom_from_filename(filename)

cpdef void forget_wisdom():
    fftw_forget_wisdom()

cpdef void set_timelimit(fftw_real limit):
    fftw_set_timelimit(limit)

cpdef void cleanup():
    fftw_cleanup()
    fftw_cleanup_threads()

cdef void _fftw_execute_dft(void *plan, void *_in, void *_out) nogil:
    fftw_execute_dft(<fftw_plan>plan, <fftw_complex *>_in, <fftw_complex *>_out)

cdef void _fftw_execute_dft_r2c(void *plan, void *_in, void *_out) nogil:
    fftw_execute_dft_r2c(<fftw_plan>plan, <fftw_real *>_in, <fftw_complex *>_out)

cdef void _fftw_execute_dft_c2r(void *plan, void *_in, void *_out) nogil:
    fftw_execute_dft_c2r(<fftw_plan>plan, <fftw_complex *>_in, <fftw_real *>_out)

cdef void _fftw_execute_r2r(void *plan, void *_in, void *_out) nogil:
    fftw_execute_r2r(<fftw_plan>plan, <fftw_real *>_in, <fftw_real *>_out)

cdef generic_function _get_execute_function(kind):
    if kind in (C2C_FORWARD, C2C_BACKWARD):
        return _fftw_execute_dft
    elif kind == R2C:
        return _fftw_execute_dft_r2c
    elif kind == C2R:
        return _fftw_execute_dft_c2r
    return _fftw_execute_r2r

cdef class FFT:
    """
    Unified class for FFTs of multidimensional arrays

    This class is used for any type of transform defined in the user manual
    of `FFTW <http://www.fftw.org/fftw3_doc>`_.

    Parameters
    ----------
    input_array : array
        real or complex input array
    output_array : array
        real or complex output array
    axes : sequence of ints, optional
        The axes to transform over, starting from the last
    kind : int or sequence of ints, optional
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
    threads : int, optional
        Number of threads to use in transforms
    flags : int or sequence of ints, optional
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
    normalization : float, optional
        Normalization factor

    """
    cdef void *_plan
    cdef np.ndarray _input_array
    cdef np.ndarray _output_array
    cdef fftw_real _M
    cdef int kind
    cdef tuple input_shape
    cdef tuple output_shape
    cdef tuple input_strides
    cdef tuple output_strides

    def __cinit__(self, input_array, output_array, axes=(-1,),
                  kind=FFTW_FORWARD, int threads=1,
                  flags=FFTW_MEASURE, fftw_real normalization=1.0):
        cdef int ndims = len(input_array.shape)
        cdef int naxes = len(axes)
        cdef int flag, i
        cdef unsigned allflags
        cdef int *sz_in = <int *> malloc(ndims * sizeof(int))
        cdef int *sz_out = <int *> malloc(ndims * sizeof(int))
        cdef int *axs = <int *> malloc(naxes * sizeof(int))
        cdef int *knd = <int *> malloc(naxes * sizeof(int))
        cdef void *_in = <void *>np.PyArray_DATA(input_array)
        cdef void *_out = <void *>np.PyArray_DATA(output_array)
        self.input_shape = input_array.shape
        self.output_shape = output_array.shape
        self.input_strides = input_array.strides
        self.output_strides = output_array.strides

        fftw_plan_with_nthreads(threads)
        flags = [flags] if isinstance(flags, int) else flags
        kind = [kind] if isinstance(kind, int) else kind
        self.kind = kind[0]
        axes = list(axes)
        for i in range(naxes):
            if axes[i] < 0:
                axes[i] = axes[i] + ndims

        allflags = flags[0]
        for flag in flags[1:]:
            allflags |= flag

        self._input_array = input_array
        self._output_array = output_array
        self._M = normalization
        for i in range(ndims):
            sz_in[i] = input_array.shape[i]
            sz_out[i] = output_array.shape[i]
        for i in range(naxes):
            axs[i] = axes[i]
        for i in range(len(kind)):
            knd[i] = kind[i]
        self._plan = fftw_planxfftn(ndims, sz_in, _in, sz_out, _out, naxes,
                                    axs, knd, allflags)
        free(sz_in)
        free(sz_out)
        free(axs)
        free(knd)

    def __dealloc__(self):
        self.destroy()

    def destroy(self):
        fftw_destroy_plan(<fftw_plan>self._plan)

    @property
    def input_array(self):
        return self._input_array

    @property
    def output_array(self):
        return self._output_array

    def print_plan(self):
        fftw_print_plan(<fftw_plan>self._plan)

    def update_arrays(self, input_array, output_array):
        assert self.input_shape == input_array.shape
        assert self.input_strides == input_array.strides
        assert self._input_array.dtype == input_array.dtype
        assert (<intptr_t>np.PyArray_DATA(input_array) %
                get_alignment(self._input_array) == 0)
        assert self.output_shape == output_array.shape
        assert self.output_strides == output_array.strides
        assert self._output_array.dtype == output_array.dtype
        assert (<intptr_t>np.PyArray_DATA(output_array) %
                get_alignment(self._output_array) == 0)
        self._input_array = input_array
        self._output_array = output_array

    def get_normalization(self):
        """Return the internally set normalization factor"""
        return self._M

    def __call__(self, input_array=None, output_array=None, implicit=True,
                 normalize=False, **kw):
        """
        Signature::

            __call__(input_array=None, output_array=None, implicit=True, normalize=False, **kw)

        Compute transform and return output array

        Parameters
        ----------
        input_array : array, optional
            If not provided, then use internally stored array
        output_array : array, optional
            If not provided, then use internally stored array
        implicit : bool, optional
            If True, then use an implicit method that acts by applying the plan
            directly on the given input array.
            If False, then use an explicit method that first copies the given
            input_array into the internal _input_array.
            The explicit method is generally safer, because it always preserves
            the provided input_array. The implicit method can be faster because
            it may be done without any copying. However, the contents of the
            input_array may be destroyed during computation. So use with care!
        normalize : bool, optional
            If True, normalize transform with internally stored normalization
            factor. The internally set normalization factor is possible to
            obtain through :func:`FFT.get_normalization`
        kw : dict, optional

        Note
        ----
        If the transform has been planned with FFTW_PRESERVE_INPUT, then both
        the two methods (implicit=True/False) will preserve the provided
        input_array. If not planned with this flag, then the implicit=True
        method may cause the input_array to be overwritten during computation.

        """
        if implicit:
            return self._apply_implicit(input_array, output_array, normalize, **kw)
        return self._apply_explicit(input_array, output_array, normalize, **kw)

    def _apply_explicit(self, input_array, output_array, normalize, **kw):
        """Apply plan with explicit (and safe) update of work arrays"""
        if input_array is not None:
            self._input_array[...] = input_array
        with nogil:
            fftw_execute(<fftw_plan>self._plan)
        if normalize:
            self._output_array *= self._M
        if output_array is not None:
            output_array[...] = self._output_array
            return output_array
        return self._output_array

    def _apply_implicit(self, input_array, output_array, normalize, **kw):
        """Apply plan with direct use of work arrays if possible

        This version of apply will use the provided input and output arrays
        instead of the original (self._input_array, self._output_array) that
        were used to plan the transform. Since planning takes the alignment of
        arrays into consideration, we need to make sure that the alignment of
        the new arrays match the originals. Other than that we also make sure
        that the new arrays have the correct shape, strides and type.
        """
        cdef void *_in
        cdef void *_out
        cdef generic_function apply_plan = _get_execute_function(self.kind)

        if input_array is not None:
            try:
                assert self.input_shape == input_array.shape
                assert self.input_strides == input_array.strides
                assert self._input_array.dtype == input_array.dtype
                assert (<intptr_t>np.PyArray_DATA(input_array) %
                        get_alignment(self._input_array) == 0)
            except AssertionError:
                self._input_array[...] = input_array
                input_array = self._input_array
        else:
            input_array = self._input_array

        if output_array is not None:
            assert self.output_shape == output_array.shape
            assert self.output_strides == output_array.strides
            assert self._output_array.dtype == output_array.dtype
            assert (<intptr_t>np.PyArray_DATA(output_array) %
                    get_alignment(self._output_array) == 0), \
                "output_array has wrong alignment"
        else:
            output_array = self._output_array

        _in = <void *>np.PyArray_DATA(input_array)
        _out = <void *>np.PyArray_DATA(output_array)
        with nogil:
            apply_plan(<fftw_plan>self._plan, _in, _out)
        if normalize:
            output_array *= self._M

        return output_array

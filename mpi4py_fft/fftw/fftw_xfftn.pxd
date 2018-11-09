cdef extern from "fftw3.h":

    ctypedef struct fftw_complex_struct:
        pass

    ctypedef fftw_complex_struct *fftw_complex

    ctypedef struct fftw_plan_struct:
        pass

    ctypedef fftw_plan_struct *fftw_plan

    void fftw_destroy_plan(fftw_plan)

    void fftw_execute_dft(fftw_plan, void *_in, void *_out) nogil

    void fftw_execute_dft_c2r(fftw_plan, void *_in, void *_out) nogil

    void fftw_execute_dft_r2c(fftw_plan, void *_in, void *_out) nogil

    void fftw_execute_r2r(fftw_plan, void *_in, void *_out) nogil

    void fftw_execute(fftw_plan) nogil

    void fftw_init_threads()

    void fftw_plan_with_nthreads(int n)

    int fftw_export_wisdom_to_filename(const char *filename)

    int fftw_import_wisdom_from_filename(const char *filename)

    void fftw_forget_wisdom()

    void fftw_set_timelimit(double seconds)

    void fftw_cleanup()

    void fftw_cleanup_threads()

    int fftw_alignment_of(void *_in)

    void fftw_print_plan(fftw_plan)


cdef extern from "fftw_planxfftn.h":

    ctypedef double fftw_real

    fftw_plan fftw_planxfftn(int      ndims,
                             int      sizes_in[],
                             void     *_in,
                             int      sizes_out[],
                             void     *_out,
                             int      naxes,
                             int      axes[],
                             int      kind[],
                             unsigned flags)

ctypedef void (*generic_function)(void *plan, void *_in, void *_out) nogil

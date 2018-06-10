cdef extern from "fftw3.h":

    ctypedef struct fftw_complex_struct:
        pass

    ctypedef fftw_complex_struct *fftw_complex

    ctypedef struct fftw_plan_struct:
        pass

    ctypedef fftw_plan_struct *fftw_plan

    void fftw_destroy_plan(fftw_plan *plan)

    void fftw_execute_dft(fftw_plan *plan, void *_in, void *_out) nogil
    
    void fftw_execute_dft_c2r(fftw_plan *plan, void *_in, void *_out) nogil
    
    void fftw_execute_dft_r2c(fftw_plan *plan, void *_in, void *_out) nogil
    
    void fftw_execute_r2r(fftw_plan *plan, void *_in, void *_out) nogil

    void fftw_execute(fftw_plan *plan) nogil

    void fftw_init_threads()

    void fftw_plan_with_nthreads(int n)

    int fftw_export_wisdom_to_filename(const char *filename)

    int fftw_import_wisdom_from_filename(const char *filename)

    void fftw_forget_wisdom()

    void fftw_set_timelimit(double seconds)

    void fftw_cleanup()

    void fftw_cleanup_threads()


cdef extern from "fftw_planxfftn.h":

    ctypedef double fftw_real

    fftw_plan fftw_planxfftn(int      ndims,
                             int      sizesA[],
                             void     *arrayA,
                             int      sizesB[],
                             void     *arrayB,
                             int      naxes,
                             int      axes[],
                             int      kind[],
                             unsigned flags)

ctypedef void (*fftw_execute_function)(void *_plan, void *_in, void *_out) nogil

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

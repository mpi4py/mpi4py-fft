#include <complex.h>
#include <fftw3.h>

#ifndef fftw_planxfftn_h 
#define fftw_planxfftn_h

typedef double fftw_real;

fftw_plan fftw_planxfftn(int      ndims,
                         int      sizes_in[ndims],
                         void     *_in,
                         int      sizes_out[ndims],
                         void     *_out,
                         int      naxes,
                         int      axes[naxes],
                         int      kind[naxes],
                         unsigned flags);

#endif

#include <complex.h>
#include <fftw3.h>

#ifndef fftw_planxfftn_h 
#define fftw_planxfftn_h

typedef double fftw_real;

fftw_plan fftw_planxfftn(int      ndims,
                         int      sizesA[ndims],
                         void     *arrayA,
                         int      sizesB[ndims],
                         void     *arrayB,
                         int      naxes,
                         int      axes[naxes],
                         int      kind,
                         unsigned flags);

#endif

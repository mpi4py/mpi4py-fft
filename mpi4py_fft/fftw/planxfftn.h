#include <complex.h>
#include <fftw3.h>

#ifndef PLANXFFTN_H
#define PLANXFFTN_H

typedef double fftw_real;
typedef float fftwf_real;
typedef long double fftwl_real;

fftw_plan fftw_planxfftn(int      ndims,
                         int      sizesA[ndims],
                         void     *arrayA,
                         int      sizesB[ndims],
                         void     *arrayB,
                         int      naxes,
                         int      axes[naxes],
                         int      kind,
                         unsigned flags);

fftwf_plan fftwf_planxfftn(int      ndims,
                           int      sizesA[ndims],
                           void     *arrayA,
                           int      sizesB[ndims],
                           void     *arrayB,
                           int      naxes,
                           int      axes[naxes],
                           int      kind,
                           unsigned flags);

fftwl_plan fftwl_planxfftn(int      ndims,
                           int      sizesA[ndims],
                           void     *arrayA,
                           int      sizesB[ndims],
                           void     *arrayB,
                           int      naxes,
                           int      axes[naxes],
                           int      kind,
                           unsigned flags);

#endif

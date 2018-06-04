#include <complex.h>
#include <fftw3.h>

#ifndef PLANXFFTN_H
#define PLANXFFTN_H

typedef double real;

fftw_plan planxfftn(int      ndims,
                    int      sizesA[ndims],
                    void     *arrayA,
                    int      sizesB[ndims],
                    void     *arrayB,
                    int      naxes,
                    int      axes[naxes],
                    int      kind,
                    unsigned flags);

#endif

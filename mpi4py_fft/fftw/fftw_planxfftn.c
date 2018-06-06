#include "fftw_planxfftn.h"

enum {
  C2C_FORWARD = FFTW_FORWARD,
  C2C_BACKWARD = FFTW_BACKWARD,
  R2C = FFTW_FORWARD-1,
  C2R = FFTW_BACKWARD+1,
};

fftw_plan fftw_planxfftn(int      ndims,
                         int      sizesA[ndims],
                         void     *arrayA,
                         int      sizesB[ndims],
                         void     *arrayB,
                         int      naxes,
                         int      axes[naxes],
                         int      kind[naxes],
                         unsigned flags)
{
  fftw_iodim ranks[ndims], dims[ndims];
  int stridesA[ndims], stridesB[ndims], markers[ndims];
  int *sizes = (kind[0] != C2R) ? sizesA : sizesB;

  stridesA[ndims-1] = 1;
  stridesB[ndims-1] = 1;
  for (int i = ndims-2; i >= 0; i--) {
    stridesA[i] = sizesA[i+1] * stridesA[i+1];
    stridesB[i] = sizesB[i+1] * stridesB[i+1];
  }

  for (int i = 0; i < ndims; i++)
    markers[i] = 0;
  for (int i = 0; i < naxes; i++) {
    int axis = axes[i];
    ranks[i].n = sizes[axis];
    ranks[i].is = stridesA[axis];
    ranks[i].os = stridesB[axis];
    markers[axis] = 1;
  }
  for (int i = 0, j = 0; i < ndims; i++) {
    if (markers[i]) continue;
    dims[j].n = sizes[i];
    dims[j].is = stridesA[i];
    dims[j].os = stridesB[i];
    j++;
  }

  switch (kind[0]) {
  case C2C_FORWARD:
  case C2C_BACKWARD:
    return fftw_plan_guru_dft(naxes, ranks,
                              ndims-naxes, dims,
                              (fftw_complex *)arrayA,
                              (fftw_complex *)arrayB,
                              kind[0], flags);
  case R2C:
    return fftw_plan_guru_dft_r2c(naxes, ranks,
                                  ndims-naxes, dims,
                                  (fftw_real *)arrayA,
                                  (fftw_complex *)arrayB,
                                  flags);
  case C2R:
    return fftw_plan_guru_dft_c2r(naxes, ranks,
                                  ndims-naxes, dims,
                                  (fftw_complex *)arrayA,
                                  (fftw_real *)arrayB,
                                  flags);
  }
  return fftw_plan_guru_r2r(naxes, ranks,
                          ndims-naxes, dims,
                          (fftw_real *)arrayA,
                          (fftw_real *)arrayB,
                          kind, flags);
}


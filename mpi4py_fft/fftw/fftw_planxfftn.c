#include "fftw_planxfftn.h"

enum {
  C2C_FORWARD = FFTW_FORWARD,
  C2C_BACKWARD = FFTW_BACKWARD,
  R2C = FFTW_FORWARD-1,
  C2R = FFTW_BACKWARD+1,
};

fftw_plan fftw_planxfftn(int      ndims,
                         int      sizes_in[ndims],
                         void     *_in,
                         int      sizes_out[ndims],
                         void     *_out,
                         int      naxes,
                         int      axes[naxes],
                         int      kind[naxes],
                         unsigned flags)
{
  fftw_iodim ranks[ndims], dims[ndims];
  int i, j, axis;
  int strides_in[ndims], strides_out[ndims], markers[ndims];
  int *sizes = (kind[0] != C2R) ? sizes_in : sizes_out;

  strides_in[ndims-1] = 1;
  strides_out[ndims-1] = 1;
  for (i = ndims-2; i >= 0; i--) {
    strides_in[i] = sizes_in[i+1] * strides_in[i+1];
    strides_out[i] = sizes_out[i+1] * strides_out[i+1];
  }

  for (i = 0; i < ndims; i++)
    markers[i] = 0;
  for (i = 0; i < naxes; i++) {
    axis = axes[i];
    ranks[i].n = sizes[axis];
    ranks[i].is = strides_in[axis];
    ranks[i].os = strides_out[axis];
    markers[axis] = 1;
  }
  for (i = 0, j = 0; i < ndims; i++) {
    if (markers[i]) continue;
    dims[j].n = sizes[i];
    dims[j].is = strides_in[i];
    dims[j].os = strides_out[i];
    j++;
  }

  switch (kind[0]) {
  case C2C_FORWARD:
  case C2C_BACKWARD:
    return fftw_plan_guru_dft(naxes, ranks,
                              ndims-naxes, dims,
                              (fftw_complex *)_in,
                              (fftw_complex *)_out,
                              kind[0], flags);
  case R2C:
    return fftw_plan_guru_dft_r2c(naxes, ranks,
                                  ndims-naxes, dims,
                                  (fftw_real *)_in,
                                  (fftw_complex *)_out,
                                  flags);
  case C2R:
    return fftw_plan_guru_dft_c2r(naxes, ranks,
                                  ndims-naxes, dims,
                                  (fftw_complex *)_in,
                                  (fftw_real *)_out,
                                  flags);
  default:
    return fftw_plan_guru_r2r(naxes, ranks,
                              ndims-naxes, dims,
                              (fftw_real *)_in,
                              (fftw_real *)_out,
                              (fftw_r2r_kind *)kind,
                              flags);
  }
}


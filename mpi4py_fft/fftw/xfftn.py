import numpy as np
from . import fftwf_xfftn, fftw_xfftn, fftwl_xfftn
from .fftw_xfftn import FFTW_FORWARD, FFTW_BACKWARD, FFTW_REDFT00, FFTW_REDFT01, \
    FFTW_REDFT10, FFTW_REDFT11, FFTW_RODFT00, FFTW_RODFT01, FFTW_RODFT10, \
    FFTW_RODFT11, FFTW_MEASURE, FFTW_DESTROY_INPUT, FFTW_UNALIGNED, \
    FFTW_CONSERVE_MEMORY, FFTW_EXHAUSTIVE, FFTW_PRESERVE_INPUT, FFTW_PATIENT, \
    FFTW_ESTIMATE, FFTW_WISDOM_ONLY, C2C_FORWARD, C2C_BACKWARD, R2C, C2R
from .fftw_xfftn import flag_dict

#__all__ = ['fftn', 'ifftn', 'rfftn', 'irfftn', 'dct', 'idct']

FFT = {
    'F': fftwf_xfftn.FFT,
    'D': fftw_xfftn.FFT,
    'G': fftwl_xfftn.FFT}

def fftn(input_array, output_array, axes=(-1,),
         threads=1, flags=(FFTW_MEASURE,)):
    kind = FFTW_FORWARD
    assert input_array.dtype.char in 'fdgFDG'
    assert np.all(input_array.shape == output_array.shape), "Arrays must be of same shape"
    dtype = input_array.dtype.char.upper()
    return FFT[dtype](input_array, output_array, axes, kind, threads, flags, 1)

def ifftn(input_array, output_array, axes=(-1,),
          threads=1, flags=(FFTW_MEASURE,)):
    kind = FFTW_BACKWARD
    assert input_array.dtype.char in 'FDG'
    assert np.all(input_array.shape == output_array.shape), "Arrays must be of same shape"
    s = input_array.shape
    M = 1
    for axis in axes:
        M *= s[axis]
    dtype = input_array.dtype.char.upper()
    return FFT[dtype](input_array, output_array, axes, kind, threads, flags, M)

def rfftn(input_array, output_array, axes=(-1,),
          threads=1, flags=(FFTW_MEASURE,)):
    kind = R2C
    assert input_array.dtype.char in 'fdg'
    assert np.all(input_array.shape[axes[-1]]//2+1 == output_array.shape[axes[-1]]), "Output array must have shape N//2+1 along first transformed axis"
    dtype = input_array.dtype.char.upper()
    return FFT[dtype](input_array, output_array, axes, kind, threads, flags, 1)

def irfftn(input_array, output_array, axes=(-1,),
           threads=1, flags=(FFTW_MEASURE,)):
    kind = C2R
    assert input_array.dtype.char in 'FDG'
    s = output_array.shape
    M = 1
    for axis in axes:
        M *= s[axis]
    dtype = input_array.dtype.char.upper()
    return FFT[dtype](input_array, output_array, axes, kind, threads, flags, M)

def dct(input_array, output_array, axes=(-1,), type=2,
        threads=1, flags=(FFTW_MEASURE,)):
    assert input_array.dtype.char in 'fdg'
    assert np.all(input_array.shape == output_array.shape), "Arrays must be of same shape"
    if type == 1:
        kind = FFTW_REDFT00
    elif type == 2:
        kind = FFTW_REDFT10  # inverse is type 3
    elif type == 3:
        kind = FFTW_REDFT01  # inverse is type 2
    dtype = input_array.dtype.char.upper()
    return FFT[dtype](input_array, output_array, axes, kind, threads, flags, 1)

def idct(input_array, output_array, axes=(-1,), type=2,
         threads=1, flags=(FFTW_MEASURE,)):
    assert input_array.dtype.char in 'fdg'
    assert np.all(input_array.shape == output_array.shape), "Arrays must be of same shape"
    s = input_array.shape
    M = 1
    if type == 1:
        kind = FFTW_REDFT00
        for axis in axes:
            M *= 2*(s[axis]-1)
    elif type == 2:
        kind = FFTW_REDFT01
        for axis in axes:
            M *= 2*s[axis]
    elif type == 3:
        kind = FFTW_REDFT10
        for axis in axes:
            M *= 2*s[axis]
    dtype = input_array.dtype.char.upper()
    return FFT[dtype](input_array, output_array, axes, kind, threads, flags, M)


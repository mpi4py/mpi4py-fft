import six
import numpy as np
from . import fftwf_xfftn, fftw_xfftn, fftwl_xfftn
from .fftw_xfftn import FFTW_FORWARD, FFTW_BACKWARD, FFTW_REDFT00, FFTW_REDFT01, \
    FFTW_REDFT10, FFTW_REDFT11, FFTW_RODFT00, FFTW_RODFT01, FFTW_RODFT10, \
    FFTW_RODFT11, FFTW_MEASURE, FFTW_DESTROY_INPUT, FFTW_UNALIGNED, \
    FFTW_CONSERVE_MEMORY, FFTW_EXHAUSTIVE, FFTW_PRESERVE_INPUT, FFTW_PATIENT, \
    FFTW_ESTIMATE, FFTW_WISDOM_ONLY, C2C_FORWARD, C2C_BACKWARD, R2C, C2R, \
    FFTW_R2HC, FFTW_HC2R, FFTW_DHT

flag_dict = {key: val for key, val in six.iteritems(locals())
             if key.startswith('FFTW_')}

fftlib = {
    'F': fftwf_xfftn,
    'D': fftw_xfftn,
    'G': fftwl_xfftn
}

def FFT(input_array, output_array, axes=(-1,), kind=FFTW_FORWARD, threads=1,
        flags=(FFTW_MEASURE,), normalize=1):
    dtype = input_array.dtype.char
    _fft = fftlib[dtype.upper()]
    return _fft.FFT(input_array, output_array, axes, kind, threads, flags, normalize)

def fftn(input_array, output_array, axes=(-1,), threads=1,
         flags=(FFTW_MEASURE,)):
    kind = FFTW_FORWARD
    assert input_array.dtype.char in 'fdgFDG'
    assert np.all(input_array.shape == output_array.shape), "Arrays must be of same shape"
    return FFT(input_array, output_array, axes, kind, threads, flags, 1)

def ifftn(input_array, output_array, axes=(-1,), threads=1,
          flags=(FFTW_MEASURE,)):
    kind = FFTW_BACKWARD
    assert input_array.dtype.char in 'FDG'
    assert np.all(input_array.shape == output_array.shape), "Arrays must be of same shape"
    s = input_array.shape
    M = 1
    for axis in axes:
        M *= s[axis]
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def rfftn(input_array, output_array, axes=(-1,), threads=1,
          flags=(FFTW_MEASURE,)):
    kind = R2C
    assert input_array.dtype.char in 'fdg'
    assert np.all(input_array.shape[axes[-1]]//2+1 == output_array.shape[axes[-1]]), \
    "Wrong shape of arrays. Shape in spectral space must be N//2+1 (N is shape in physical space)"
    return FFT(input_array, output_array, axes, kind, threads, flags, 1)

def irfftn(input_array, output_array, axes=(-1,), threads=1,
           flags=(FFTW_MEASURE,)):
    kind = C2R
    assert input_array.dtype.char in 'FDG'
    s = output_array.shape
    assert np.all(output_array.shape[axes[-1]]//2+1 == input_array.shape[axes[-1]]), \
    "Wrong shape of arrays. Shape in spectral space must be N//2+1 (N is shape in physical space)"
    M = 1
    for axis in axes:
        M *= s[axis]
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def dct(input_array, output_array, axes=(-1,), type=2, threads=1,
        flags=(FFTW_MEASURE,)):
    assert input_array.dtype.char in 'fdg'
    assert np.all(input_array.shape == output_array.shape), "Arrays must be of same shape"
    if type == 1:
        kind = FFTW_REDFT00
    elif type == 2:
        kind = FFTW_REDFT10  # inverse is type 3
    elif type == 3:
        kind = FFTW_REDFT01  # inverse is type 2
    kind = [kind]*len(axes)
    return FFT(input_array, output_array, axes, kind, threads, flags, 1)

def idct(input_array, output_array, axes=(-1,), type=2, threads=1,
         flags=(FFTW_MEASURE,)):
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
    kind = [kind]*len(axes)
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def dst(input_array, output_array, axes=(-1,), type=2, threads=1,
        flags=(FFTW_MEASURE,)):
    assert input_array.dtype.char in 'fdg'
    assert np.all(input_array.shape == output_array.shape), "Arrays must be of same shape"
    if type == 1:
        kind = FFTW_RODFT00
    elif type == 2:
        kind = FFTW_RODFT10  # inverse is type 3
    elif type == 3:
        kind = FFTW_RODFT01  # inverse is type 2
    kind = [kind]*len(axes)
    return FFT(input_array, output_array, axes, kind, threads, flags, 1)

def idst(input_array, output_array, axes=(-1,), type=2, threads=1,
         flags=(FFTW_MEASURE,)):
    assert input_array.dtype.char in 'fdg'
    assert np.all(input_array.shape == output_array.shape), "Arrays must be of same shape"
    s = input_array.shape
    M = 1
    if type == 1:
        kind = FFTW_RODFT00
        for axis in axes:
            M *= 2*(s[axis]+1)
    elif type == 2:
        kind = FFTW_RODFT01
        for axis in axes:
            M *= 2*s[axis]
    elif type == 3:
        kind = FFTW_RODFT10
        for axis in axes:
            M *= 2*s[axis]
    kind = [kind]*len(axes)
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def ihfftn(input_array, output_array, axes=(-1,), threads=1,
           flags=(FFTW_MEASURE,)):
    kind = R2C
    assert input_array.dtype.char in 'fdg'
    assert np.all(input_array.shape[axes[-1]]//2+1 == output_array.shape[axes[-1]]), "Output array must have shape N//2+1 along first transformed axis"
    s = input_array.shape
    M = 1
    for axis in axes:
        M *= s[axis]
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def hfftn(input_array, output_array, axes=(-1,), threads=1,
          flags=(FFTW_MEASURE,)):
    kind = C2R
    assert input_array.dtype.char in 'FDG'
    return FFT(input_array, output_array, axes, kind, threads, flags, 1)

def export_wisdom(filename):
    """Export FFTW wisdom

    Parameters
    ----------
    filename : str
        Name of file used to export wisdom to

    Notes
    -----
    Wisdom is stored for all three precisions, float, double and long double,
    using, respectively, prefix 'F_', 'D_' and 'G_'. Wisdom is
    imported using :mod:`.import_wisdom`.

    See also
    --------
    :mod:`.import_wisdom`

    """
    e = []
    for key, lib in six.iteritems(fftlib):
        e.append(lib.export_wisdom(bytes(key+'_'+filename, 'utf-8')))
    assert np.all(np.array(e) == 1), "Not able to export wisdom {}".format(filename)

def import_wisdom(filename):
    """Import FFTW wisdom

    Parameters
    ----------
    filename : str
        Name of file used to import wisdom from

    Notes
    -----
    Wisdom is imported for all three precisions, float, double and long double,
    using, respectively, prefix 'F_', 'D_' and 'G_'. Wisdom is
    exported using :mod:`.export_wisdom`.

    See also
    --------
    :mod:`.export_wisdom`

    """
    e = []
    for key, lib in six.iteritems(fftlib):
        e.append(lib.import_wisdom(bytes(key+'_'+filename, 'utf-8')))
    assert np.all(np.array(e) == 1), "Not able to import wisdom {}".format(filename)


import six
import numpy as np
from . import fftwf_xfftn, fftw_xfftn, fftwl_xfftn
from .utilities import FFTW_FORWARD, FFTW_MEASURE

fftlib = {
    'F': fftwf_xfftn,
    'D': fftw_xfftn,
    'G': fftwl_xfftn}

def get_planned_FFT(input_array, output_array, axes=(-1,), kind=FFTW_FORWARD,
                    threads=1, flags=(FFTW_MEASURE,), normalize=1):
    """Return instance of transform class

    Returned class instance is either :class:`.fftwf_xfftn.FFT`,
    :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT` depending on the type
    of the input array

    Parameters
    ----------
    input_array : array
        real or complex input array
    output_array : array
        real or complex output array
    axes : sequence of ints, optional
        The axes to transform over, starting from the last
    kind : int or sequence of ints, optional
        Any one of

            - FFTW_FORWARD (-1)
            - FFTW_R2HC (0)
            - FFTW_BACKWARD (1)
            - FFTW_HC2R (1)
            - FFTW_DHT (2)
            - FFTW_REDFT00 (3)
            - FFTW_REDFT01 (4)
            - FFTW_REDFT10 (5)
            - FFTW_REDFT11 (6)
            - FFTW_RODFT00 (7)
            - FFTW_RODFT01 (8)
            - FFTW_RODFT10 (9)
            - FFTW_RODFT11 (10)
    threads : int, optional
        Number of threads to use in transforms
    flags : int or sequence of ints, optional
        Any one of, but not necessarily for all transforms or all combinations

            - FFTW_MEASURE (0)
            - FFTW_DESTROY_INPUT (1)
            - FFTW_UNALIGNED (2)
            - FFTW_CONSERVE_MEMORY (4)
            - FFTW_EXHAUSTIVE (8)
            - FFTW_PRESERVE_INPUT (16)
            - FFTW_PATIENT (32)
            - FFTW_ESTIMATE (64)
            - FFTW_WISDOM_ONLY (2097152)
    normalization : int, optional
        Normalization factor

    """
    dtype = input_array.dtype.char
    _fft = fftlib[dtype.upper()]
    return _fft.FFT(input_array, output_array, axes, kind, threads, flags,
                    normalize)

def export_wisdom(filename):
    """Export FFTW wisdom

    Parameters
    ----------
    filename : str
        Name of file used to export wisdom to

    Note
    ----
    Wisdom is stored for all three precisions, float, double and long double,
    using, respectively, prefix ``F_``, ``D_`` and ``G_``. Wisdom is
    imported using :func:`.import_wisdom`.

    See also
    --------
    :func:`.import_wisdom`

    """
    e = []
    for key, lib in six.iteritems(fftlib):
        e.append(lib.export_wisdom(bytearray(key+'_'+filename, 'utf-8')))
    assert np.all(np.array(e) == 1), "Not able to export wisdom {}".format(filename)

def import_wisdom(filename):
    """Import FFTW wisdom

    Parameters
    ----------
    filename : str
        Name of file used to import wisdom from

    Note
    ----
    Wisdom is imported for all three precisions, float, double and long double,
    using, respectively, prefix ``F_``, ``D_`` and ``G_``. Wisdom is
    exported using :func:`.export_wisdom`.

    See also
    --------
    :func:`.export_wisdom`

    """
    e = []
    for key, lib in six.iteritems(fftlib):
        e.append(lib.import_wisdom(bytearray(key+'_'+filename, 'utf-8')))
    assert np.all(np.array(e) == 1), "Not able to import wisdom {}".format(filename)



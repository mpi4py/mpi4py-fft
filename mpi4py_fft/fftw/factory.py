#pylint: disable=no-name-in-module

import numpy as np
from mpi4py import MPI
from .utilities import FFTW_FORWARD, FFTW_MEASURE

def get_fftw_lib(dtype):
    """Return compiled fftw module interfacing the FFTW library

    Parameters
    ----------
    dtype : dtype
        Data precision

    Returns
    -------
    Module or ``None``
        Module can be either :mod:`.fftwf_xfftn`, :mod:`.fftw_xfftn` or
        :mod:`.fftwl_xfftn`, depending on precision.
    """

    dtype = np.dtype(dtype).char.upper()
    if dtype == 'G':
        try:
            from . import fftwl_xfftn
            return fftwl_xfftn
        except ImportError: #pragma: no cover
            return None
    elif dtype == 'D':
        try:
            from . import fftw_xfftn
            return fftw_xfftn
        except ImportError: #pragma: no cover
            return None
    elif dtype == 'F':
        try:
            from . import fftwf_xfftn
            return fftwf_xfftn
        except ImportError: #pragma: no cover
            return None

fftlib = {}
for t in 'fdg':
    fftw_lib = get_fftw_lib(t)
    if fftw_lib is not None:
        fftlib[t.upper()] = fftw_lib

comm = MPI.COMM_WORLD

def get_planned_FFT(input_array, output_array, axes=(-1,), kind=FFTW_FORWARD,
                    threads=1, flags=(FFTW_MEASURE,), normalization=1.0):
    """Return instance of transform class

    Parameters
    ----------
    input_array : array
        real or complex input array
    output_array : array
        real or complex output array
    axes : sequence of ints, optional
        The axes to transform over, starting from the last
    kind : int or sequence of ints, optional
        Any one of (or possibly several for real-to-real)

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
    normalization : float, optional
        Normalization factor

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for the desired transforms

    """
    dtype = input_array.dtype.char
    assert dtype.upper() in fftlib
    _fft = fftlib[dtype.upper()]
    return _fft.FFT(input_array, output_array, axes, kind, threads, flags,
                    normalization)

def export_wisdom(filename):
    """Export FFTW wisdom

    Parameters
    ----------
    filename : str
        Name of file used to export wisdom to

    Note
    ----
    Wisdom is stored for all precisions available: float, double and long
    double, using, respectively, prefix ``Fn_``, ``Dn_`` and ``Gn_``, where
    n is the rank of the processor.
    Wisdom is imported using :func:`.import_wisdom`, which must be called
    with the same MPI configuration as used with :func:`.export_wisdom`.

    See also
    --------
    :func:`.import_wisdom`

    """
    rank = str(comm.Get_rank())
    e = []
    for key, lib in fftlib.items():
        e.append(lib.export_wisdom(bytearray(key+rank+'_'+filename, 'utf-8')))
    assert np.all(np.array(e) == 1), "Not able to export wisdom {}".format(filename)

def import_wisdom(filename):
    """Import FFTW wisdom

    Parameters
    ----------
    filename : str
        Name of file used to import wisdom from

    Note
    ----
    Wisdom is imported for all available precisions: float, double and long
    double, using, respectively, prefix ``Fn_``, ``Dn_`` and ``Gn_``, where
    n is the rank of the processor.
    Wisdom is exported using :func:`.export_wisdom`.
    Note that importing wisdom only works when using the same MPI configuration
    as used with :func:`.export_wisdom`.


    See also
    --------
    :func:`.export_wisdom`

    """
    rank = str(comm.Get_rank())
    e = []
    for key, lib in fftlib.items():
        e.append(lib.import_wisdom(bytearray(key+rank+'_'+filename, 'utf-8')))
    assert np.all(np.array(e) == 1), "Not able to import wisdom {}".format(filename)

def forget_wisdom():
    for lib in fftlib.values():
        lib.forget_wisdom()

def set_timelimit(limit):
    """Set time limit for planning

    Parameters
    ----------
    limit : number
        The new time limit set for planning of serial transforms
    """
    for lib in fftlib.values():
        lib.set_timelimit(limit) # limit's precision handled by cython

def cleanup():
    for lib in fftlib.values():
        lib.cleanup()

import six
import numpy as np
from . import fftwf_xfftn, fftw_xfftn, fftwl_xfftn
from .utilities import FFTW_FORWARD, FFTW_BACKWARD, FFTW_REDFT00, FFTW_REDFT01, \
    FFTW_REDFT10, FFTW_REDFT11, FFTW_RODFT00, FFTW_RODFT01, FFTW_RODFT10, \
    FFTW_RODFT11, FFTW_MEASURE, FFTW_DESTROY_INPUT, FFTW_UNALIGNED, \
    FFTW_CONSERVE_MEMORY, FFTW_EXHAUSTIVE, FFTW_PRESERVE_INPUT, FFTW_PATIENT, \
    FFTW_ESTIMATE, FFTW_WISDOM_ONLY, C2C_FORWARD, C2C_BACKWARD, R2C, C2R, \
    FFTW_R2HC, FFTW_HC2R, FFTW_DHT, get_alignment, aligned, aligned_like

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

def fftn(input_array, s=None, axes=(-1,), threads=1,
         flags=(FFTW_MEASURE,), output_array=None):
    """Return complex-to-complex forward transform object

    Parameters
    ----------
    input_array : complex array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    threads : int, optional
        Number of threads used in computing FFT.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : complex array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for complex-to-complex transforms

    Note
    ----
    This routine does not compute the fftn, it merely returns an instance of
    a class that can do it.
    The contents of the input_array may be overwritten during planning. Make
    sure to keep a copy if needed.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import fftn as plan_fftn
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, aligned
    >>> A = aligned(4, dtype='D')
    >>> fftn = plan_fftn(A, flags=(FFTW_ESTIMATE,))
    >>> A[:] = 1, 2, 3, 4
    >>> B = fftn()
    >>> print(B)
    [10.+0.j -2.+2.j -2.+0.j -2.-2.j]
    >>> assert id(A) == id(fftn.input_array)
    >>> assert id(B) == id(fftn.output_array)

    """
    kind = FFTW_FORWARD
    assert input_array.dtype.char in 'FDG'
    if output_array is None:
        n = get_alignment(input_array)
        output_array = aligned(input_array.shape, n,
                               input_array.dtype.char.upper())
    else:
        assert input_array.shape == output_array.shape
        assert output_array.dtype.char == input_array.dtype.char.upper()
    sz = input_array.shape
    M = 1.0
    for axis in axes:
        M *= sz[axis]
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def ifftn(input_array, s=None, axes=(-1,), threads=1,
          flags=(FFTW_MEASURE,), output_array=None):
    """
    Return complex-to-complex inverse transform object

    Parameters
    ----------
    input_array : array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT.
    threads : int, optional
        Number of threads used in computing FFT.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for complex-to-complex inverse
        transforms

    Note
    ----
    This routine does not compute the ifftn, it merely returns an instance of
    a class that can do it.
    The contents of the input_array may be overwritten during planning. Make
    sure that you keep a copy if needed.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import ifftn as plan_ifftn
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, FFTW_PRESERVE_INPUT, aligned
    >>> A = aligned(4, dtype='D')
    >>> ifftn = plan_ifftn(A, flags=(FFTW_ESTIMATE, FFTW_PRESERVE_INPUT))
    >>> A[:] = 1, 2, 3, 4
    >>> B = ifftn()
    >>> print(B)
    [10.+0.j -2.-2.j -2.+0.j -2.+2.j]
    >>> assert id(B) == id(ifftn.output_array)
    >>> assert id(A) == id(ifftn.input_array)

    """
    kind = FFTW_BACKWARD
    assert input_array.dtype.char in 'FDG'
    if output_array is None:
        output_array = aligned_like(input_array)
    else:
        assert input_array.shape == output_array.shape
    sz = input_array.shape
    M = 1.0
    for axis in axes:
        M *= sz[axis]
    return FFT(input_array, output_array, axes, kind, threads, flags, 1.0/M)

def rfftn(input_array, s=None, axes=(-1,), threads=1,
          flags=(FFTW_MEASURE,), output_array=None):
    """Return real-to-complex transform object

    Parameters
    ----------
    input_array : real array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the real to complex FFT.
    threads : int, optional
        Number of threads used in computing FFT.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for real-to-complex transforms

    Note
    ----
    This routine does not compute the rfftn, it merely returns an instance of
    a class that can do it.
    The contents of the input_array may be overwritten during planning. Make
    sure that you keep a copy if needed.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import rfftn as plan_rfftn
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, aligned
    >>> A = aligned(4, dtype='d')
    >>> rfftn = plan_rfftn(A, flags=(FFTW_ESTIMATE,))
    >>> A[:] = 1, 2, 3, 4
    >>> B = rfftn()
    >>> print(B)
    [10.+0.j -2.+2.j -2.+0.j]
    >>> assert id(A) == id(rfftn.input_array)
    >>> assert id(B) == id(rfftn.output_array)

    """
    kind = R2C
    assert input_array.dtype.char in 'fdg'
    if output_array is None:
        sz = list(input_array.shape)
        sz[axes[-1]] = input_array.shape[axes[-1]]//2+1
        D = input_array.dtype.char.upper()
        _fft = fftlib[D]
        n = _fft.get_alignment(input_array)
        output_array = aligned(sz, n=n, dtype=np.dtype(D))
    else:
        assert input_array.shape[axes[-1]]//2+1 == output_array.shape[axes[-1]]
    M = 1.0
    sz = input_array.shape
    for axis in axes:
        M *= sz[axis]
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def irfftn(input_array, s=None, axes=(-1,), threads=1,
           flags=(FFTW_MEASURE,), output_array=None):
    """Return inverse complex-to-real transform object

    Parameters
    ----------
    input_array : array
    s : sequence of ints, optional
        Shape of output array along each of the transformed axes. Must be same
        length as axes (len(s) == len(axes)). If not given it is assumed that
        the shape of the output along the first transformed axis (i.e.,
        axes[-1]) is an even number. It is not possible to determine exactly,
        because for a real transform the output of a real array of length N is
        N//2+1. However, both N=4 and N=5 gives 4//2+1=3 and 5//2+1=3, so it is
        not possible to determine whether 4 or 5 is correct. Hence it must be
        given.
    axes : sequence of ints, optional
        Axes over which to compute the real to complex FFT.
    threads : int, optional
        Number of threads used in computing FFT.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for complex-to-real transforms

    Note
    ----
    This routine does not compute the irfftn, it merely returns an instance of
    a class that can do it.
    The irfftn is not possible to use with the FFTW_PRESERVE_INPUT flag.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import irfftn as plan_irfftn
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, aligned
    >>> A = aligned(4, dtype='D')
    >>> irfftn = plan_irfftn(A, flags=(FFTW_ESTIMATE,)) # no shape given for output
    >>> A[:] = 1, 2, 3, 4
    >>> B = irfftn()
    >>> print(B)
    [15. -4.  0. -1.  0. -4.]
    >>> irfftn = plan_irfftn(A, s=(7,), flags=(FFTW_ESTIMATE,)) # output shape given
    >>> B = irfftn()
    >>> print(B)
    [19.         -5.04891734 -0.30797853 -0.64310413 -0.64310413 -0.30797853
     -5.04891734]
    >>> assert id(B) == id(irfftn.output_array)
    >>> assert id(A) == id(irfftn.input_array)

    """
    kind = C2R
    assert input_array.dtype.char in 'FDG'
    assert FFTW_PRESERVE_INPUT not in flags
    sz = list(input_array.shape)
    if s is not None:
        assert len(axes) == len(s)
        for q, axis in zip(s, axes):
            sz[axis] = q
    else:
        sz[axes[-1]] = 2*sz[axes[-1]]-2
    if output_array is None:
        _fft = fftlib[input_array.dtype.char]
        n = _fft.get_alignment(input_array)
        output_array = aligned(sz, n=n, dtype=np.dtype(input_array.dtype.char.lower()))
    else:
        assert list(output_array.shape) == sz

    assert sz[axes[-1]]//2+1 == input_array.shape[axes[-1]]
    M = 1.0
    sz = output_array.shape
    for axis in axes:
        M /= sz[axis]
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def dctn(input_array, s=None, axes=(-1,), type=2, threads=1,
         flags=(FFTW_MEASURE,), output_array=None):
    """Return discrete cosine transform object

    Parameters
    ----------
    input_array : array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the real-to-real dct.
    threads : int, optional
        Number of threads used in computing dct.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for real-to-real dct transforms
        of given type

    Note
    ----
    This routine does not compute the dct, it merely returns an instance of
    a class that can do it.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import dctn as plan_dct
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, aligned
    >>> A = aligned(4, dtype='d')
    >>> dct = plan_dct(A, flags=(FFTW_ESTIMATE,))
    >>> A[:] = 1, 2, 3, 4
    >>> B = dct()
    >>> print(B)
    [20.         -6.30864406  0.         -0.44834153]
    >>> assert id(A) == id(dct.input_array)
    >>> assert id(B) == id(dct.output_array)

    """
    assert input_array.dtype.char in 'fdg'
    if output_array is None:
        output_array = aligned_like(input_array)
    else:
        assert input_array.shape == output_array.shape
    sz = input_array.shape
    M = 1
    if type == 1:
        kind = FFTW_REDFT00
        for axis in axes:
            M *= 2*(sz[axis]-1)
    elif type == 2:
        kind = FFTW_REDFT10  # inverse is type 3
        for axis in axes:
            M *= 2*sz[axis]
    elif type == 3:
        kind = FFTW_REDFT01  # inverse is type 2
        for axis in axes:
            M *= 2*sz[axis]
    elif type == 4:
        kind = FFTW_REDFT11
        for axis in axes:
            M *= 2*sz[axis]
    kind = [kind]*len(axes)
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def idctn(input_array, s=None, axes=(-1,), type=2, threads=1,
          flags=(FFTW_MEASURE,), output_array=None):
    """Return inverse discrete cosine transform object

    Parameters
    ----------
    input_array : array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the real-to-real idct.
    threads : int, optional
        Number of threads used in computing idct.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for real-to-real idct transforms
        of given type

    Note
    ----
    This routine does not compute the idct, it merely returns an instance of
    a class that can do it.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import idctn as plan_idct
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, aligned
    >>> A = aligned(4, dtype='d')
    >>> idct = plan_idct(A, flags=(FFTW_ESTIMATE,))
    >>> A[:] = 1, 2, 3, 4
    >>> B = idct()
    >>> print(B)
    [11.99962628 -9.10294322  2.61766184 -1.5143449 ]
    >>> assert id(A) == id(idct.input_array)
    >>> assert id(B) == id(idct.output_array)

    """
    assert input_array.dtype.char in 'fdg'
    if output_array is None:
        output_array = aligned_like(input_array)
    else:
        assert input_array.shape == output_array.shape
    sz = input_array.shape
    M = 1
    if type == 1:
        kind = FFTW_REDFT00
        for axis in axes:
            M *= 2*(sz[axis]-1)
    elif type == 2:
        kind = FFTW_REDFT01
        for axis in axes:
            M *= 2*sz[axis]
    elif type == 3:
        kind = FFTW_REDFT10
        for axis in axes:
            M *= 2*sz[axis]
    elif type == 4:
        kind = FFTW_REDFT11
        for axis in axes:
            M *= 2*sz[axis]
    kind = [kind]*len(axes)
    return FFT(input_array, output_array, axes, kind, threads, flags, 1./M)

def dstn(input_array, s=None, axes=(-1,), type=2, threads=1,
         flags=(FFTW_MEASURE,), output_array=None):
    """Return discrete sine transform object

    Parameters
    ----------
    input_array : array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the real-to-real dst.
    threads : int, optional
        Number of threads used in computing dst.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for real-to-real dst transforms
        of given type

    Note
    ----
    This routine does not compute the dst, it merely returns an instance of
    a class that can do it.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import dstn as plan_dst
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, aligned
    >>> A = aligned(4, dtype='d')
    >>> dst = plan_dst(A, flags=(FFTW_ESTIMATE,))
    >>> A[:] = 1, 2, 3, 4
    >>> B = dst()
    >>> print(B)
    [13.06562965 -5.65685425  5.411961   -4.        ]
    >>> assert id(A) == id(dst.input_array)
    >>> assert id(B) == id(dst.output_array)

    """
    assert input_array.dtype.char in 'fdg'
    if output_array is None:
        output_array = aligned_like(input_array)
    else:
        assert input_array.shape == output_array.shape
    sz = input_array.shape
    M = 1
    if type == 1:
        kind = FFTW_RODFT00
        for axis in axes:
            M *= 2*(sz[axis]+1)
    elif type == 2:
        kind = FFTW_RODFT10  # inverse is type 3
        for axis in axes:
            M *= 2*sz[axis]
    elif type == 3:
        kind = FFTW_RODFT01  # inverse is type 2
        for axis in axes:
            M *= 2*sz[axis]
    elif type == 4:
        kind = FFTW_RODFT11
        for axis in axes:
            M *= 2*sz[axis]
    kind = [kind]*len(axes)
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

def idstn(input_array, s=None, axes=(-1,), type=2, threads=1,
          flags=(FFTW_MEASURE,), output_array=None):
    """Return inverse discrete sine transform object

    Parameters
    ----------
    input_array : array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the real-to-real inverse dst.
    threads : int, optional
        Number of threads used in computing inverse dst.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for real-to-real idst transforms
        of given type

    Note
    ----
    This routine does not compute the idst, it merely returns an instance of
    a class that can do it.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import idstn as plan_idst
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, aligned
    >>> A = aligned(4, dtype='d')
    >>> idst = plan_idst(A, flags=(FFTW_ESTIMATE,))
    >>> A[:] = 1, 2, 3, 4
    >>> B = idst()
    >>> print(B)
    [13.13707118 -1.6199144   0.72323135 -0.51978306]
    >>> assert id(A) == id(idst.input_array)
    >>> assert id(B) == id(idst.output_array)

    """
    assert input_array.dtype.char in 'fdg'
    if output_array is None:
        output_array = aligned_like(input_array)
    else:
        assert input_array.shape == output_array.shape
    sz = input_array.shape
    M = 1
    if type == 1:
        kind = FFTW_RODFT00
        for axis in axes:
            M *= 2*(sz[axis]+1)
    elif type == 2:
        kind = FFTW_RODFT01
        for axis in axes:
            M *= 2*sz[axis]
    elif type == 3:
        kind = FFTW_RODFT10
        for axis in axes:
            M *= 2*sz[axis]
    elif type == 4:
        kind = FFTW_RODFT11
        for axis in axes:
            M *= 2*sz[axis]
    kind = [kind]*len(axes)
    return FFT(input_array, output_array, axes, kind, threads, flags, 1./M)

def ihfftn(input_array, s=None, axes=(-1,), threads=1,
           flags=(FFTW_MEASURE,), output_array=None):
    """Return inverse transform object for an array with Hermitian symmetry

    Parameters
    ----------
    input_array : array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the ihfftn.
    threads : int, optional
        Number of threads used in computing ihfftn.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for real-to-complex ihfftn
        transforms

    Note
    ----
    This routine does not compute the ihfttn, it merely returns an instance of
    a class that can do it.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import ihfftn as plan_ihfftn
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, aligned
    >>> A = aligned(4, dtype='d')
    >>> ihfftn = plan_ihfftn(A, flags=(FFTW_ESTIMATE,))
    >>> A[:] = 1, 2, 3, 4
    >>> B = ihfftn()
    >>> print(B)
    [10.+0.j -2.+2.j -2.+0.j]
    >>> assert id(A) == id(ihfftn.input_array)
    >>> assert id(B) == id(ihfftn.output_array)

    """
    kind = R2C
    assert input_array.dtype.char in 'fdg'
    if output_array is None:
        sz = list(input_array.shape)
        sz[axes[-1]] = input_array.shape[axes[-1]]//2+1
        _fft = fftlib[input_array.dtype.char.upper()]
        n = _fft.get_alignment(input_array)
        output_array = aligned(sz, n=n, dtype=np.dtype(input_array.dtype.char.upper()))
    else:
        assert input_array.shape[axes[-1]]//2+1 == output_array.shape[axes[-1]]
    sz = input_array.shape
    M = 1
    for axis in axes:
        M *= sz[axis]
    return FFT(input_array, output_array, axes, kind, threads, flags, 1./M)

def hfftn(input_array, s=None, axes=(-1,), threads=1,
          flags=(FFTW_MEASURE,), output_array=None):
    """Return transform object for an array with Hermitian symmetry

    Parameters
    ----------
    input_array : array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the hfftn.
    threads : int, optional
        Number of threads used in computing hfftn.
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Returns
    -------
    :class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`
        An instance of the return type configured for complex-to-real hfftn
        transforms

    Note
    ----
    This routine does not compute the hfttn, it merely returns an instance of
    a class that can do it.

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py_fft.fftw import hfftn as plan_hfftn
    >>> from mpi4py_fft.fftw import FFTW_ESTIMATE, aligned
    >>> A = aligned(4, dtype='D')
    >>> hfftn = plan_hfftn(A, flags=(FFTW_ESTIMATE,)) # no shape given for output
    >>> A[:] = 1, 2, 3, 4
    >>> B = hfftn()
    >>> print(B)
    [15. -4.  0. -1.  0. -4.]
    >>> hfftn = plan_hfftn(A, s=(7,), flags=(FFTW_ESTIMATE,)) # output shape given
    >>> B = hfftn()
    >>> print(B)
    [19.         -5.04891734 -0.30797853 -0.64310413 -0.64310413 -0.30797853
     -5.04891734]
    >>> assert id(B) == id(hfftn.output_array)
    >>> assert id(A) == id(hfftn.input_array)

    """
    kind = C2R
    assert input_array.dtype.char in 'FDG'
    sz = list(input_array.shape)
    if s is not None:
        assert len(axes) == len(s)
        for q, axis in zip(s, axes):
            sz[axis] = q
    else:
        sz[axes[-1]] = 2*sz[axes[-1]]-2
    if output_array is None:
        _fft = fftlib[input_array.dtype.char]
        n = _fft.get_alignment(input_array)
        output_array = aligned(sz, n=n, dtype=np.dtype(input_array.dtype.char.lower()))
    else:
        assert list(output_array.shape) == sz
    assert sz[axes[-1]]//2+1 == input_array.shape[axes[-1]]
    M = 1
    for axis in axes:
        M *= sz[axis]
    return FFT(input_array, output_array, axes, kind, threads, flags, M)

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

inverse = {
    FFTW_RODFT11: FFTW_RODFT11,
    FFTW_REDFT11: FFTW_REDFT11,
    FFTW_RODFT01: FFTW_RODFT10,
    FFTW_RODFT10: FFTW_RODFT01,
    FFTW_REDFT01: FFTW_REDFT10,
    FFTW_REDFT10: FFTW_REDFT01,
    FFTW_RODFT00: FFTW_RODFT00,
    FFTW_REDFT00: FFTW_REDFT00,
    rfftn: irfftn,
    irfftn: rfftn,
    fftn: ifftn,
    ifftn: fftn,
    dctn: idctn,
    idctn: dctn,
    dstn: idstn,
    idstn: dstn,
    hfftn: ihfftn,
    ihfftn: hfftn
}


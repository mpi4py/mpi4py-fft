#pylint: disable=no-name-in-module,unused-import
import numpy as np
from .factory import get_planned_FFT
from .utilities import FFTW_FORWARD, FFTW_BACKWARD, FFTW_REDFT00, FFTW_REDFT01, \
    FFTW_REDFT10, FFTW_REDFT11, FFTW_RODFT00, FFTW_RODFT01, FFTW_RODFT10, \
    FFTW_RODFT11, FFTW_MEASURE, FFTW_DESTROY_INPUT, FFTW_UNALIGNED, \
    FFTW_CONSERVE_MEMORY, FFTW_EXHAUSTIVE, FFTW_PRESERVE_INPUT, FFTW_PATIENT, \
    FFTW_ESTIMATE, FFTW_WISDOM_ONLY, C2C_FORWARD, C2C_BACKWARD, R2C, C2R, \
    FFTW_R2HC, FFTW_HC2R, FFTW_DHT, get_alignment, aligned, aligned_like

flag_dict = {key: val for key, val in locals().items()
             if key.startswith('FFTW_')}

dct_type = {
    1: FFTW_REDFT00,
    2: FFTW_REDFT10,
    3: FFTW_REDFT01,
    4: FFTW_REDFT11}

idct_type = {
    1: FFTW_REDFT00,
    2: FFTW_REDFT01,
    3: FFTW_REDFT10,
    4: FFTW_REDFT11}

dst_type = {
    1: FFTW_RODFT00,
    2: FFTW_RODFT10,
    3: FFTW_RODFT01,
    4: FFTW_RODFT11}

idst_type = {
    1: FFTW_RODFT00,
    2: FFTW_RODFT01,
    3: FFTW_RODFT10,
    4: FFTW_RODFT11}

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
    M = np.prod(np.take(input_array.shape, axes))
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, 1.0/M)

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
    M = np.prod(np.take(input_array.shape, axes))
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, 1.0/M)

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
        dtype = input_array.dtype.char
        n = get_alignment(input_array)
        output_array = aligned(sz, n=n, dtype=np.dtype(dtype.upper()))
    else:
        assert input_array.shape[axes[-1]]//2+1 == output_array.shape[axes[-1]]
    M = np.prod(np.take(input_array.shape, axes))
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, 1.0/M)

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
        dtype = input_array.dtype.char
        n = get_alignment(input_array)
        output_array = aligned(sz, n=n, dtype=np.dtype(dtype.lower()))
    else:
        assert list(output_array.shape) == sz

    assert sz[axes[-1]]//2+1 == input_array.shape[axes[-1]]
    M = np.prod(np.take(output_array.shape, axes))
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, 1.0/M)

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
    type : int, optional
        Type of `dct <http://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transform-Kinds.html>`_

            - 1 - FFTW_REDFT00
            - 2 - FFTW_REDFT10,
            - 3 - FFTW_REDFT01,
            - 4 - FFTW_REDFT11
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
    kind = dct_type[type]
    kind = [kind]*len(axes)
    M = get_normalization(kind, input_array.shape, axes)
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, M)

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
    type : int, optional
        Type of `idct <http://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transform-Kinds.html>`_

            - 1 - FFTW_REDFT00
            - 2 - FFTW_REDFT01
            - 3 - FFTW_REDFT10
            - 4 - FFTW_REDFT11
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
    kind = idct_type[type]
    kind = [kind]*len(axes)
    M = get_normalization(kind, input_array.shape, axes)
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, M)

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
    type : int, optional
        Type of `dst <http://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transform-Kinds.html>`_

            - 1 - FFTW_RODFT00
            - 2 - FFTW_RODFT10
            - 3 - FFTW_RODFT01
            - 4 - FFTW_RODFT11
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
    kind = dst_type[type]
    kind = [kind]*len(axes)
    M = get_normalization(kind, input_array.shape, axes)
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, M)

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
    type : int, optional
        Type of `idst <http://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transform-Kinds.html>`_

            - 1 - FFTW_RODFT00
            - 2 - FFTW_RODFT01
            - 3 - FFTW_RODFT10
            - 4 - FFTW_RODFT11
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
    kind = idst_type[type]
    kind = [kind]*len(axes)
    M = get_normalization(kind, input_array.shape, axes)
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, M)

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
        dtype = input_array.dtype.char
        sz = list(input_array.shape)
        sz[axes[-1]] = input_array.shape[axes[-1]]//2+1
        n = get_alignment(input_array)
        output_array = aligned(sz, n=n, dtype=np.dtype(dtype.upper()))
    else:
        assert input_array.shape[axes[-1]]//2+1 == output_array.shape[axes[-1]]
    M = get_normalization(kind, input_array.shape, axes)
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, M)

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
        dtype = input_array.dtype.char
        n = get_alignment(input_array)
        output_array = aligned(sz, n=n, dtype=np.dtype(dtype.lower()))
    else:
        assert list(output_array.shape) == sz
    assert sz[axes[-1]]//2+1 == input_array.shape[axes[-1]]
    M = get_normalization(kind, sz, axes)
    return get_planned_FFT(input_array, output_array, axes, kind, threads,
                           flags, M)

def get_normalization(kind, shape, axes):
    """Return normalization factor for multidimensional transform

    The normalization factor is, for Fourier transforms::

        1./np.prod(np.take(shape, axes))

    where shape is the global shape of the array that is input to the
    forward transform, and axes are the axes transformed over.

    For real-to-real transforms the normalization factor for each axis is

        - REDFT00 - 2(N-1)
        - REDFT01 - 2N
        - REDFT10 - 2N
        - REDFT11 - 2N
        - RODFT00 - 2(N+1)
        - RODFT01 - 2N
        - RODFT10 - 2N
        - RODFT11 - 2N

    where N is the length of the input array along that axis.

    Parameters
    ----------
    kind : sequence of ints
        The kind of transform along each axis
    shape : sequence of ints
        The shape of the global transformed array (input to the forward
        transform)
    axes : sequence of ints
        The axes transformed over

    Note
    ----
    The returned normalization factor is the *inverse* of the product of the
    normalization factors for the axes it is transformed over.

    """
    kind = [kind]*len(axes) if isinstance(kind, int) else kind
    assert len(kind) == len(axes)
    M = 1
    for knd, axis in zip(kind, axes):
        N = shape[axis]
        if knd == FFTW_RODFT00:
            M *= 2*(N+1)
        elif knd == FFTW_REDFT00:
            M *= 2*(N-1)
        elif knd in (FFTW_RODFT01, FFTW_RODFT10, FFTW_RODFT11,
                     FFTW_REDFT01, FFTW_REDFT10, FFTW_REDFT11):
            M *= 2*N
        else:
            M *= N
    return 1./M

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

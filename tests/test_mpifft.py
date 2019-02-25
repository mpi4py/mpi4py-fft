from __future__ import print_function
import functools
import numpy as np
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT
from mpi4py_fft.pencil import Subcomm
from mpi4py_fft.distributedarray import DistributedArray, newDarray
from mpi4py_fft import fftw

has_pyfftw = True
try:
    import pyfftw
except ImportError:
    has_pyfftw = False

abstol = dict(f=0.1, d=2e-10, g=1e-10)

def allclose(a, b):
    atol = abstol[a.dtype.char.lower()]
    return np.allclose(a, b, rtol=0, atol=atol)

def random_like(array):
    shape = array.shape
    dtype = array.dtype
    return np.random.random(shape).astype(dtype)

def random_true_or_false(comm):
    r = 0
    if comm.rank == 0:
        r = np.random.randint(2)
    r = comm.bcast(r)
    return r

def test_r2r():
    N = (5, 6, 7, 8, 9)
    dctn = functools.partial(fftw.dctn, type=3)
    idctn = functools.partial(fftw.idctn, type=3)
    dstn = functools.partial(fftw.dstn, type=3)
    idstn = functools.partial(fftw.idstn, type=3)
    fft = PFFT(MPI.COMM_WORLD, N, axes=((0,), (1, 2), (3, 4)), slab=True,
               transforms={(1, 2): (dctn, idctn), (3, 4): (dstn, idstn)})

    A = newDarray(fft, forward_output=False)
    A[:] = np.random.random(A.shape)
    C = fftw.aligned_like(A)
    B = fft.forward(A)
    C = fft.backward(B, C)
    assert np.allclose(A, C)

def test_mpifft():
    from itertools import product

    comm = MPI.COMM_WORLD
    dims = (2, 3, 4,)
    sizes = (16, 17)
    types = ''
    for t in 'fdg':
        if fftw.get_fftw_lib(t):
            types += t+t.upper()
    for typecode in types:
        for dim in dims:
            for shape in product(*([sizes]*dim)):

                if dim < 3:
                    n = min(shape)
                    if typecode in 'fdg':
                        n //= 2
                        n += 1
                    if n < comm.size:
                        continue
                for slab in (True, False):
                    padding = False
                    for collapse in (True, False):
                        for use_pyfftw in (False, True):
                            if has_pyfftw is False and use_pyfftw is True:
                                continue
                            transforms = None
                            if dim < 3:
                                allaxes = [None, (-1,), (-2,),
                                           (-1, -2,), (-2, -1),
                                           (-1, 0), (0, -1),
                                           ((0,), (1,))]
                            elif dim < 4:
                                allaxes = [None, ((0,), (1, 2)),
                                           ((0,), (-2, -1))]
                            elif dim > 3:
                                allaxes = [None, ((0,), (1,), (2,), (3,)),
                                           ((0,), (1, 2, 3)),
                                           ((0,), (1,), (2, 3))]
                                dctn = functools.partial(fftw.dctn, type=3)
                                idctn = functools.partial(fftw.idctn, type=3)

                                if not typecode in 'FDG':
                                    if use_pyfftw:
                                        transforms = {(3,): (pyfftw.builders.rfftn, pyfftw.builders.irfftn),
                                                      (2, 3): (pyfftw.builders.rfftn, pyfftw.builders.irfftn),
                                                      (1, 2, 3): (pyfftw.builders.rfftn, pyfftw.builders.irfftn),
                                                      (0, 1, 2, 3): (pyfftw.builders.rfftn, pyfftw.builders.irfftn)}
                                    else:
                                        transforms = {(3,): (dctn, idctn),
                                                      (2, 3): (dctn, idctn),
                                                      (1, 2, 3): (dctn, idctn),
                                                      (0, 1, 2, 3): (dctn, idctn)}
                            for axes in allaxes:
                                _slab = slab
                                # Test also the slab is number interface
                                if slab is True and axes is not None:
                                    ax = axes[-1] if isinstance(axes[-1], int) else axes[-1][-1]
                                    _slab = (ax+1) % len(shape)
                                    if random_true_or_false(comm) == 1:
                                        _slab -= len(shape) # Test neg axes interface
                                _comm = comm
                                # Test also the comm is Subcomm interfaces
                                # For PFFT the Subcomm needs to be as long as shape
                                if len(shape) > 2 and axes is None and slab is False:
                                    _dims = [0] * len(shape)
                                    _dims[-1] = 1 # distribute all but last axis (axes is None)
                                    _comm = comm
                                    if random_true_or_false(comm) == 1:
                                        # then test Subcomm with a MPI.CART argument
                                        _dims = MPI.Compute_dims(comm.Get_size(), _dims)
                                        _comm = comm.Create_cart(_dims)
                                        _dims = None
                                    _comm = Subcomm(_comm, _dims)
                                #print(typecode, shape, axes, collapse)
                                fft = PFFT(_comm, shape, axes=axes, dtype=typecode,
                                           padding=padding, slab=_slab, collapse=collapse,
                                           use_pyfftw=use_pyfftw, transforms=transforms)

                                #if comm.rank == 0:
                                #    grid = [c.size for c in fft.subcomm]
                                #    print('grid:{} shape:{} typecode:{} use_pyfftw:{} axes:{}'
                                #          .format(grid, shape, typecode, use_pyfftw, axes))
                                assert fft.dtype(True) == fft.forward.output_array.dtype
                                assert fft.dtype(False) == fft.forward.input_array.dtype
                                assert len(fft.axes) == len(fft.xfftn)
                                assert len(fft.axes) == len(fft.transfer) + 1
                                assert (fft.forward.input_pencil.subshape ==
                                        fft.forward.input_array.shape)
                                assert (fft.forward.output_pencil.subshape ==
                                        fft.forward.output_array.shape)
                                assert (fft.backward.input_pencil.subshape ==
                                        fft.backward.input_array.shape)
                                assert (fft.backward.output_pencil.subshape ==
                                        fft.backward.output_array.shape)
                                assert np.alltrue(np.array(fft.shape(True)) == np.array(fft.forward.output_pencil.shape))
                                assert np.alltrue(np.array(fft.shape(False)) == np.array(fft.forward.input_pencil.shape))
                                ax = -1 if axes is None else axes[-1] if isinstance(axes[-1], int) else axes[-1][-1]
                                assert fft.forward.input_pencil.substart[ax] == 0
                                assert fft.backward.output_pencil.substart[ax] == 0
                                ax = 0 if axes is None else axes[0] if isinstance(axes[0], int) else axes[0][0]
                                assert fft.forward.output_pencil.substart[ax] == 0
                                assert fft.backward.input_pencil.substart[ax] == 0
                                assert fft.dimensions() == len(shape)

                                U = random_like(fft.forward.input_array)

                                if random_true_or_false(comm) == 1:
                                    F = fft.forward(U)
                                    V = fft.backward(F)
                                    assert allclose(V, U)
                                else:
                                    fft.forward.input_array[...] = U
                                    fft.forward()
                                    fft.backward()
                                    V = fft.backward.output_array
                                    assert allclose(V, U)

                                fft.destroy()

                    padding = [1.5]*len(shape)
                    for use_pyfftw in (True, False):
                        if has_pyfftw is False and use_pyfftw is True:
                            continue
                        if dim < 3:
                            allaxes = [None, (-1,), (-2,),
                                       (-1, -2,), (-2, -1),
                                       (-1, 0), (0, -1),
                                       ((0,), (1,))]
                        elif dim < 4:
                            allaxes = [None, ((0,), (1,), (2,)),
                                       ((0,), (-2,), (-1,))]
                        elif dim > 3:
                            allaxes = [None, (0, 1, -2, -1),
                                       ((0,), (1,), (2,), (3,))]

                        for axes in allaxes:

                            fft = PFFT(comm, shape, axes=axes, dtype=typecode,
                                       padding=padding, slab=slab, use_pyfftw=use_pyfftw)

                            #if comm.rank == 0:
                            #    grid = [c.size for c in fft.subcomm]
                            #    print('grid:{} shape:{} typecode:{} use_pyfftw:{} axes:{}'
                            #          .format(grid, shape, typecode, use_pyfftw, axes))

                            assert len(fft.axes) == len(fft.xfftn)
                            assert len(fft.axes) == len(fft.transfer) + 1
                            assert (fft.forward.input_pencil.subshape ==
                                    fft.forward.input_array.shape)
                            assert (fft.forward.output_pencil.subshape ==
                                    fft.forward.output_array.shape)
                            assert (fft.backward.input_pencil.subshape ==
                                    fft.backward.input_array.shape)
                            assert (fft.backward.output_pencil.subshape ==
                                    fft.backward.output_array.shape)
                            ax = -1 if axes is None else axes[-1] if isinstance(axes[-1], int) else axes[-1][-1]
                            assert fft.forward.input_pencil.substart[ax] == 0
                            assert fft.backward.output_pencil.substart[ax] == 0
                            ax = 0 if axes is None else axes[0] if isinstance(axes[0], int) else axes[0][0]
                            assert fft.forward.output_pencil.substart[ax] == 0
                            assert fft.backward.input_pencil.substart[ax] == 0

                            U = random_like(fft.forward.input_array)
                            F = fft.forward(U)

                            if random_true_or_false(comm) == 1:
                                Fc = F.copy()
                                V = fft.backward(F)
                                F = fft.forward(V)
                                assert allclose(F, Fc)
                            else:
                                fft.backward.input_array[...] = F
                                fft.backward()
                                fft.forward()
                                V = fft.forward.output_array
                                assert allclose(F, V)

                            fft.destroy()

if __name__ == '__main__':
    test_mpifft()
    test_r2r()

from mpi4py import MPI
from mpi4py_fft.pencil import Transfer, CustomMPITransfer, Pencil, Subcomm
import numpy as np

transfer_classes = [CustomMPITransfer]
xps = {CustomMPITransfer: np}

try:
    import cupy as cp
    from mpi4py_fft.pencil import NCCLTransfer
    transfer_classes += [NCCLTransfer]
    xps[NCCLTransfer] = cp
except ModuleNotFoundError:
    pass


def get_args(comm, shape, dtype):
    subcomm = Subcomm(comm=comm, dims=None)
    pencilA = Pencil(subcomm, shape, 0)
    pencilB = Pencil(subcomm, shape, 1)

    kwargs = {
        'comm': comm,
        'shape': shape,
        'subshapeA': pencilA.subshape,
        'subshapeB': pencilB.subshape,
        'axisA': 0,
        'axisB': 1,
        'dtype': dtype,
    }
    return kwargs


def get_arrays(kwargs, xp):
    arrayA = xp.zeros(shape=kwargs['subshapeA'], dtype=kwargs['dtype'])
    arrayB = xp.zeros(shape=kwargs['subshapeB'], dtype=kwargs['dtype'])

    arrayA[:] = xp.random.random(arrayA.shape).astype(arrayA.dtype)
    return arrayA, arrayB
 

def single_test_all_to_allw(transfer_class, shape, dtype, comm=None, xp=None):
    comm = comm if comm else MPI.COMM_WORLD
    kwargs = get_args(comm, shape, dtype)
    arrayA, arrayB = get_arrays(kwargs, xp)
    arrayB_ref = arrayB.copy()

    transfer = transfer_class(**kwargs)
    reference_transfer = Transfer(**kwargs)

    transfer.Alltoallw(arrayA, transfer._subtypesA, arrayB, transfer._subtypesB)
    reference_transfer.Alltoallw(arrayA, transfer._subtypesA, arrayB_ref, transfer._subtypesB)
    assert xp.allclose(arrayB, arrayB_ref), f'Did not get the same result from `alltoallw` with {transfer_class.__name__} transfer class as MPI implementation on rank {comm.rank}!'

    comm.Barrier()
    if comm.rank == 0:
        print(f'{transfer_class.__name__} passed alltoallw test with shape {shape} and dtype {dtype}')


def single_test_forward_backward(transfer_class, shape, dtype, comm=None, xp=None):
    comm = comm if comm else MPI.COMM_WORLD
    kwargs = get_args(comm, shape, dtype)
    arrayA, arrayB = get_arrays(kwargs, xp)
    arrayA_ref = arrayA.copy()

    transfer = transfer_class(**kwargs)

    transfer.forward(arrayA, arrayB)
    transfer.backward(arrayB, arrayA)
    assert xp.allclose(arrayA, arrayA_ref), f'Did not get the same result when transferring back and forth with {transfer_class.__name__} transfer class on rank {comm.rank}!'

    comm.Barrier()
    if comm.rank == 0:
        print(f'{transfer_class.__name__} passed forward/backward test with shape {shape} and dtype {dtype}')


def test_transfer_class():
    dims = (2, 3)
    sizes = (7, 8, 9, 128)
    dtypes = 'fFdD'

    shapes = [[size] * dim for size in sizes for dim in dims] + [[32, 256, 129]]

    for transfer_class in transfer_classes:
        for shape in shapes:
            for dtype in dtypes:
                single_test_all_to_allw(transfer_class, shape, dtype, xp=xps[transfer_class])
                single_test_forward_backward(transfer_class, shape, dtype, xp=xps[transfer_class])


if __name__ == '__main__':
    test_transfer_class()

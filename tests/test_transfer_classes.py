from mpi4py import MPI
from mpi4py_fft.pencil import Transfer, CustomMPITransfer, Pencil, Subcomm
import numpy as np


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


def get_arrays(kwargs):
    arrayA = np.zeros(shape=kwargs['subshapeA'], dtype=kwargs['dtype'])
    arrayB = np.zeros(shape=kwargs['subshapeB'], dtype=kwargs['dtype'])

    arrayA[:] = np.random.random(arrayA.shape).astype(arrayA.dtype)
    return arrayA, arrayB
 

def single_test_all_to_allw(transfer_class, shape, dtype, comm=None):
    comm = comm if comm else MPI.COMM_WORLD
    kwargs = get_args(comm, shape, dtype)
    arrayA, arrayB = get_arrays(kwargs)
    arrayB_ref = arrayB.copy()

    transfer = transfer_class(**kwargs)
    reference_transfer = Transfer(**kwargs)

    transfer.Alltoallw(arrayA, transfer._subtypesA, arrayB, transfer._subtypesB)
    reference_transfer.Alltoallw(arrayA, transfer._subtypesA, arrayB_ref, transfer._subtypesB)
    assert np.allclose(arrayB, arrayB_ref), f'Did not get the same result from `alltoallw` with {transfer_class.__name__} transfer class as MPI implementation on rank {comm.rank}!'

    comm.Barrier()
    if comm.rank == 0:
        print(f'{transfer_class.__name__} passed alltoallw test with shape {shape} and dtype {dtype}')


def single_test_forward_backward(transfer_class, shape, dtype, comm=None):
    comm = comm if comm else MPI.COMM_WORLD
    kwargs = get_args(comm, shape, dtype)
    arrayA, arrayB = get_arrays(kwargs)
    arrayA_ref = arrayA.copy()

    transfer = transfer_class(**kwargs)

    transfer.forward(arrayA, arrayB)
    transfer.backward(arrayB, arrayA)
    assert np.allclose(arrayA, arrayA_ref), f'Did not get the same result when transferring back and forth with {transfer_class.__name__} transfer class on rank {comm.rank}!'

    comm.Barrier()
    if comm.rank == 0:
        print(f'{transfer_class.__name__} passed forward/backward test with shape {shape} and dtype {dtype}')


def test_transfer_class():
    dims = (2, 3)
    sizes = (7, 8, 9, 128)
    dtypes = 'fFdD'
    transfer_class = CustomMPITransfer

    shapes = [[size] * dim for size in sizes for dim in dims] + [[32, 256, 129]]

    for shape in shapes:
        for dtype in dtypes:
            single_test_all_to_allw(transfer_class, shape, dtype)
            single_test_forward_backward(transfer_class, shape, dtype)


if __name__ == '__main__':
    test_transfer_class()

from __future__ import print_function
import numpy as np
from mpi4py import MPI
from mpi4py_fft.pencil import Subcomm, Pencil


def test_pencil():
    from itertools import product
    comm = MPI.COMM_WORLD
    dims = (2, 3)
    sizes = (7, 8, 9)
    types = 'fdFD' #'hilfdgFDG'

    for typecode in types:
        for dim in dims:
            for shape in product(*([sizes]*dim)):
                axes = list(range(dim))
                for axis1, axis2, axis3 in product(axes, axes, axes):

                    if axis1 == axis2: continue
                    if axis2 == axis3: continue
                    axis3 -= len(shape)
                    #if comm.rank == 0:
                    #    print(shape, axis1, axis2, axis3, typecode)

                    for pdim in [None] + list(range(1, dim-1)):

                        subcomm = Subcomm(comm, pdim)
                        pencil0 = Pencil(subcomm, shape)

                        pencilA = pencil0.pencil(axis1)
                        pencilB = pencilA.pencil(axis2)
                        pencilC = pencilB.pencil(axis3)

                        trans1 = Pencil.transfer(pencilA, pencilB, typecode)
                        trans2 = Pencil.transfer(pencilB, pencilC, typecode)

                        X = np.random.random(pencilA.subshape).astype(typecode)

                        A = np.empty(pencilA.subshape, dtype=typecode)
                        B = np.empty(pencilB.subshape, dtype=typecode)
                        C = np.empty(pencilC.subshape, dtype=typecode)

                        A[...] = X

                        B.fill(0)
                        trans1.forward(A, B)
                        C.fill(0)
                        trans2.forward(B, C)

                        B.fill(0)
                        trans2.backward(C, B)
                        A.fill(0)
                        trans1.backward(B, A)

                        assert np.allclose(A, X)

                        trans1.destroy()
                        trans2.destroy()
                        subcomm.destroy()


if __name__ == '__main__':
    test_pencil()

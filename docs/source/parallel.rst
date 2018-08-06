Parallel Fast Fourier Transforms
================================

Parallel FFTs are computed through a combination of :ref:`global redistributions <global>`
and :ref:`serial transforms <dfts>`. In mpi4py-fft the interface to performing such
parallel transforms is the :class:`.mpifft.PFFT` class. The class is highly
configurable and best explained through a few examples.

Slab decomposition
..................

With slab decompositions we use only one group of processors and distribute 
only one index of a multidimensional array at the time.

Consider the complete transform of a three-dimensional array of random numbers, 
and of shape (128, 128, 128). We can plan the transform of such an array with 
the following code snippet::

    import numpy as np
    from mpi4py import MPI
    from mpi4py_fft.mpifft import PFFT, Function
    N = np.array([128, 128, 128], dtype=int)
    fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype=np.float, slab=True)

Here the signature ``N, axes=(0, 1, 2), dtype=np.float, slab=True`` tells us 
that the created ``fft`` instance is *planned* such as to slab distribute and 
transform any 3D array of shape ``N`` and type ``np.float``. Furthermore, we 
plan to transform axis 2 first, and then 1 and 0, which is exactly the reverse 
order of ``axes=(0, 1, 2)``. Mathematically, the planned transform corresponds
to

.. math::

    \tilde{u}_{j_0/P,k_1,k_2} &= \mathcal{F}_1( \mathcal{F}_{2}(u_{j_0/P, j_1, j_2}), \\
    \tilde{u}_{j_0, k_1/P, k_2} &\xleftarrow[P]{1\rightarrow 0} \tilde{u}_{j_0/P, k_1, k_2}, \\
    \hat{u}_{k_0,k_1/P,k_2} &= \mathcal{F}_0(\tilde{u}_{j_0, k_1/P, k_2}).

Note that axis 0 is distributed on the
input array and axis 1 on the output array. In the first step above we compute
the transforms along axes 2 and 1 (in that order), but we cannot compute the
serial transform along axis 0 since the global array is distributed in that
direction. We need to perform a global redistribution, the middle step,
that realigns the global data such that it is aligned in axes 0. 
With data aligned in axis 0, we can perform the final transform 
:math:`\mathcal{F}_{0}` and be done with it.

Assume now that all the code in this section is stored to a file named 
``pfft_example.py``, and add to the above code::

    u = Function(fft, False)
    u[:] = np.random.random(u.shape).astype(u.dtype)
    u_hat = fft.forward(u)
    uj = np.zeros_like(u)
    uj = fft.backward(u_hat, uj)
    assert np.allclose(uj, u)
    print(MPI.COMM_WORLD.Get_rank(), u.shape)

Running this code with two processors (``mpirun -np 2 python pfft_example.py``) 
should raise no exception, and the output should be::

    1 (64, 128, 128)
    0 (64, 128, 128)

This shows that the first index has been shared between the two processors
equally. The array ``u`` thus corresponds to :math:`u_{j_0/P,j_1,j_2}`. Note 
that :class:`.Function` is an overloaded Numpy ndarray which simply has
a constructor using ``fft`` to determine its size and type. 
``Function(fft, False)`` here simply returns an ndarray of shape (64, 128, 128)
and type ``np.float``. The ``False`` argument indicates that the shape
and type should be of the input array type, as opposed to the output
array type (:math:`\hat{u}_{k_0,k_1/P,k_2}` that one gets with ``True``).

Note that because the input array is of real type, and not complex, the
output array will be of global shape::

    128, 128, 65 

The output array will be distributed in axis 1, so the output array
shape should be (128, 64, 65). We check this by adding the following
code and rerunning::

    u_hat = Function(fft, True)
    print(MPI.COMM_WORLD.Get_rank(), u_hat.shape)

leading to an additional print of::

    1 (128, 64, 65)
    0 (128, 64, 65)

To distribute in the first axis first is default and most efficient for
row-major C arrays. However, we can easily configure the ``fft`` instance
by modifying the axes keyword. Changing for example to::

    fft = PFFT(MPI.COMM_WORLD, N, axes=(2, 0, 1), dtype=np.float)

and axis 1 will be transformed first, such that the global output array
will be of shape (128, 65, 128). The distributed input and output arrays
will now have shape::

    0 (64, 128, 128)
    1 (64, 128, 128)
    
    0 (128, 33, 128)
    1 (128, 32, 128)

Note that the input array will still be distributed in axis 0 and the
output is axis 1. However, the size of the two output arrays are no longer
equal because 65 is an odd number.

Another way to tweak the distribution is to use the :class:`.Subcomm`
class directly::

    subcomms = Subcomm(MPI.COMM_WORLD, [1, 0, 1])
    fft = PFFT(subcomms, N, axes=(0, 1, 2), dtype=np.float)

Here the ``subcomms`` tuple will decide that axis 1 should be distributed,
because the only zero in the list ``[1, 0, 1]`` is along axis 1. The ones
determine that axes 0 and 2 should use one processor each, i.e., they should
be non-distributed.

The :class:`.PFFT` class has a few additional keyword arguments that one
should be aware of. The default behaviour of :class:`.PFFT` is to use
one transform object for each axis, and then use these sequentially. 
Setting ``collapse=True`` will attempt to minimize the number of transform
objects by combining whenever possible. Take our example, the array
:math:`u_{j_0/P,j_1,j_2}` can transform along both axes 1 and 2 simultaneously,
without any intermediate global redistributions. By setting
``collapse=True`` only one object of ``rfftn(u, axes=(1, 2))`` will be
used instead of two (like ``rfftn(rfftn(u, axes=2), axes=1)``). 
Note that a collapse can also be configured through the ``axes`` keyword, 
using::

    fft = PFFT(MPI.COMM_WORLD, N, axes=((0,), (1, 2)), dtype=np.float)

will collapse axes 1 and 2, just like one would obtain with ``collapse=True``.

If serial transforms other than :func:`.fftn`/:func:`.rfftn` and 
:func:`.ifftn`/:func:`.irfftn` are required, then this can be achieved
using the ``transforms`` keyword and a dictionary pointing from axes to
the type of transform. We can for example combine real-to-real
with real-to-complex transforms like this::

    from mpi4py_fft.fftw import rfftn, irfftn, dctn, idctn
    import functools
    dct = functools.partial(dctn, type=3)
    idct = functools.partial(idctn, type=3)
    transforms = {(0,): (rfftn, irfftn), (1, 2): (dct, idct)}
    r2c = PFFT(MPI.COMM_WORLD, N, axes=((0,), (1, 2)), transforms=transforms)
    u = Function(r2c, False)
    u[:] = np.random.random(u.shape).astype(u.dtype)
    u_hat = r2c.forward(u)
    uj = np.zeros_like(u)
    uj = r2c.backward(u_hat, uj)
    assert np.allclose(uj, u)

Pencil decomposition
....................

A pencil decomposition uses two groups of processors. Each group then is
responsible for distributing one index set each of a multidimensional array.
We can perform a pencil decomposition simply by running the first example
from the previous section, but now with 4 processors. To remind you, we
put this in ``pfft_example.py``, where now ``slab=True`` has been removed
in the PFFT calling:: 

    import numpy as np
    from mpi4py import MPI
    from mpi4py_fft.mpifft import PFFT, Function

    N = np.array([128, 128, 128], dtype=int)
    fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype=np.float)
    u = Function(fft, False)
    u[:] = np.random.random(u.shape).astype(u.dtype)
    u_hat = fft.forward(u)
    uj = np.zeros_like(u)
    uj = fft.backward(u_hat, uj)
    assert np.allclose(uj, u)
    print(MPI.COMM_WORLD.Get_rank(), u.shape)

The output of running ``mpirun -np 4 python pfft_example.py`` will then be::

    0 (64, 64, 128)
    2 (64, 64, 128)
    3 (64, 64, 128)
    1 (64, 64, 128)

Note that now both the two first index sets are shared, so we have a pencil
decomposition. The shared input array is now denoted as 
:math:`u_{j_0/P_0,j_1/P_1,j2}` and the complete forward transform performs 
the following 5 steps:

.. math::

    \tilde{u}_{j_0/P_0,j_1/P_1,k_2} &= \mathcal{F}_{2}(u_{j_0/P_0, j_1/P_1, j_2}), \\
    \tilde{u}_{j_0/P_0, j_1, k_2/P_1} &\xleftarrow[P_1]{2\rightarrow 1} \tilde{u}_{j_0/P_0, j_1/P_1, k_2}, \\
    \tilde{u}_{j_0/P_0,k_1,k_2/P_1} &= \mathcal{F}_1(\tilde{u}_{j_0/P_0, j_1, k_2/P_1}), \\
    \tilde{u}_{j_0, k_1/P_0, k_2/P_1} &\xleftarrow[P_0]{1\rightarrow 0} \tilde{u}_{j_0/P_0, k_1, k_2/P_1}, \\
    \hat{u}_{k_0,k_1/P_0,k_2/P_1} &= \mathcal{F}_0(\tilde{u}_{j_0, k_1/P_0, k_2/P_1}).


Like for the slab decomposition, the order of the different steps is 
configurable. Simply change the value of ``axes``, e.g., as::

    fft = PFFT(MPI.COMM_WORLD, N, axes=(2, 0, 1), dtype=np.float)

and the input and output arrays will be of shape::

    3 (64, 128, 64)
    2 (64, 128, 64)
    1 (64, 128, 64)
    0 (64, 128, 64)

    3 (64, 32, 128)
    2 (64, 32, 128)
    1 (64, 33, 128)
    0 (64, 33, 128)

We see that the input array is aligned in axis 1, because this is the direction
transformed first.
 

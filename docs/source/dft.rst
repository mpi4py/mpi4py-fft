.. _dfts:

Discrete Fourier Transforms
---------------------------

Consider first two one-dimensional arrays :math:`\boldsymbol{u} = \{u_j\}_{j=0}^{N-1}` and
:math:`\boldsymbol{\hat{u}} =\{\hat{u}_k\}_{k=0}^{N-1}`. We define the forward and backward 
Discrete Fourier transforms (DFT), respectively, as

.. math::
    :label: dft

    \hat{u}_k &= \frac{1}{N}\sum_{j=0}^{N-1}u_j e^{-2\pi i j k / N}, \quad \forall \, k=0, 1, \ldots, N-1, \\
    u_j &= \sum_{k=0}^{N-1}\hat{u}_k e^{2\pi i j k / N}, \quad \forall \, j=0, 1, \ldots, N-1,

where :math:`i=\sqrt{-1}`. Discrete Fourier transforms are computed efficiently
using algorithms termed Fast Fourier Transforms, known in short as FFTs. 

An more compact notation is commonly used for the DFTs, where the 1D 
forward and backward transforms are written as

.. math::

    \boldsymbol{\hat{u}} &= \mathcal{F}(\boldsymbol{u}), \\
    \boldsymbol{u} &= \mathcal{F}^{-1}(\boldsymbol{\hat{u}}).

Numpy, Scipy, and many other scientific softwares contain implementations that 
make working with Fourier series simple and straight forward. These 1D Fourier 
transforms can be implemented easily with just Numpy as, e.g.::

    import numpy as np
    N = 16
    u = np.random.random(N)
    u_hat = np.fft.fft(u)
    uc = np.fft.ifft(u_hat)
    assert np.allclose(u, uc)

However, there is a minor difference. Numpy performs by default the 
:math:`1/N` scaling with the *backward* transform (``ifft``) and not the
forward as shown in :eq:`dft`. These are merely different conventions and
not important as long as one is aware of them. We use
the scaling on the forward transform simply because this follows naturally 
when using the harmonic functions :math:`e^{2 \pi k x}` as basis functions 
when solving PDEs with the 
`spectral Galerkin method <https://github.com/spectralDNS/shenfun>`_ or
the `spectral collocation method (see chap. 3) <https://people.maths.ox.ac.uk/trefethen/spectral.html>`_.

With mpi4py-fft the same operations take just a few more steps, because instead
of executing ffts directly, like in the calls for ``np.fft.fft`` and 
``np.fft.ifft``, we need to create the objects that are to do the 
transforms first. We need to *plan* the transforms::

    from mpi4py_fft import fftw
    u = fftw.aligned(N, dtype=np.complex)
    u_hat = fftw.aligned_like(u)
    fft = fftw.fftn(u, flags=(fftw.FFTW_MEASURE,))        # plan fft
    ifft = fftw.ifftn(u_hat, flags=(fftw.FFTW_ESTIMATE,)) # plan ifft
    u[:] = np.random.random(N)
    # Now execute the transforms
    u_hat = fft(u, u_hat, normalize=True)
    uc = ifft(u_hat)
    assert np.allclose(uc, u)

The planning of transforms makes an effort to find the fastest possible transform
of the given kind. See more in :ref:`fftwmodule`.

Multidimensional transforms
...........................

It is for multidimensional arrays that it starts to become
interesting for the current software. Multidimensional arrays are a bit tedious
with notation, though, especially when the number of dimensions grow. We will
stick with the `index notation <https://en.wikipedia.org/wiki/Index_notation>`_
because it it most stright forward in comparison with implementation.

We denote the entries of a two-dimensional array as :math:`u_{j_0, j_1}`, 
which corresponds to a row-major matrix
:math:`\boldsymbol{u}=\{u_{j_0, j_1}\}_{(j_0, j_1) \in \textbf{j}_0 \times \textbf{j}_1}` of 
size :math:`N_0\cdot N_1`. Denoting also :math:`\omega_m=j_m k_m / N_m`, a 
two-dimensional forward and backward DFTs can be defined as

.. math::
    :label: 2dfourier

    \hat{u}_{k_0,k_1} &= \frac{1}{N_0}\sum_{j_0 \in \textbf{j}_0}\Big( e^{-2\pi i \omega_0} \frac{1}{N_1} \sum_{j_1\in \textbf{j}_1} \Big( e^{-2\pi i \omega_1} u_{j_0,j_1}\Big) \Big), \quad \forall \, (k_0, k_1) \in \textbf{k}_0  \times \textbf{k}_1, \\
    u_{j_0, j_1} &= \sum_{k_1\in \textbf{k}_1} \Big( e^{2\pi i \omega_1} \sum_{k_0\in\textbf{k}_0} \Big(  e^{2\pi i \omega_0} \hat{u}_{k_0, k_1} \Big) \Big), \quad \forall \, (j_0, j_1) \in \textbf{j}_0 \times \textbf{j}_1.

Note that the forward transform corresponds to taking the 1D Fourier 
transform first along axis 1, once for each of the indices in :math:`j_0`. 
Afterwords the transform is executed along axis 0. The two steps are more 
easily understood if we break things up a little bit and write the forward
transform in :eq:`2dfourier` in two steps as

.. math::
    :label: forward2

    \tilde{u}_{j_0,k_1} &= \frac{1}{N_1}\sum_{j_1 \in \textbf{j}_1} u_{j_0,j_1} e^{-2\pi i \omega_1}, \quad \forall \, k_1 \in \textbf{k}_1, \\
    \hat{u}_{k_0,k_1} &= \frac{1}{N_0}\sum_{j_0 \in \textbf{j}_0} \tilde{u}_{j_0,k_1} e^{-2\pi i \omega_0}, \quad \forall \, k_0 \in \textbf{k}_0.

The backward (inverse) transform
if performed in the opposite order, axis 0 first and then 1. The order is actually
arbitrary, but this is how is is usually computed. With mpi4py-fft the
order of the directional transforms can easily be configured.

We can write the complete transform on compact notation as

.. math::
    :label: dft_short

    \boldsymbol{\hat{u}} &= \mathcal{F}(\boldsymbol{u}), \\
    \boldsymbol{u} &= \mathcal{F}^{-1}(\boldsymbol{\hat{u}}).

But if we denote the two *partial* transforms along each axis as 
:math:`\mathcal{F}_0` and :math:`\mathcal{F}_1`, we can also write it as

.. math::
    :label: forward_2dpartial

    \boldsymbol{\hat{u}} &= \mathcal{F}_0(\mathcal{F}_1(\boldsymbol{u})), \\
    \boldsymbol{u} &= \mathcal{F}_1^{-1}(\mathcal{F}_0^{-1}(\boldsymbol{\hat{u}})).


Extension to multiple dimensions is straight forward. We denote a :math:`d`-dimensional
array as :math:`u_{j_0, j_1, \ldots, j_{d-1}}` and a partial transform of :math:`u` 
along axis :math:`i` is denoted as

.. math::
    :label: partial_dft

    \tilde{u}_{j_0, \ldots, k_i, \ldots, j_{d-1}} = \mathcal{F}_i(u_{j_0, \ldots, j_i, \ldots, j_d})

We get the complete multidimensional transforms on short form still as :eq:`dft_short`, and
with partial transforms as

.. math::
    :label: multi_dft_partial

    \boldsymbol{\hat{u}} &= \mathcal{F}_0(\mathcal{F}_1( \ldots \mathcal{F}_{d-1}(\boldsymbol{u})), \\
    \boldsymbol{u} &= \mathcal{F}_{d-1}^{-1}( \mathcal{F}_{d-2}^{-1}( \ldots \mathcal{F}_0^{-1}(\boldsymbol{\hat{u}}))).


Multidimensional transforms are straightforward to implement in Numpy

.. _numpy2d:
.. code-block:: python

    import numpy as np
    M, N = 16, 16
    u = np.random.random((M, N))
    u_hat = np.fft.rfftn(u)
    uc = np.fft.irfftn(u_hat)
    assert np.allclose(u, uc)

.. _fftwmodule:

The :mod:`.fftw` module
.......................

The :mod:`.fftw` module provides an interface to most of the  
`FFTW library <http://www.fftw.org>`_. In the :mod:`.fftw.xfftn`
submodule there are planner functions for:

    * :func:`.fftn` - complex-to-complex forward Fast Fourier Transforms 
    * :func:`.ifftn` - complex-to-complex backward Fast Fourier Transforms
    * :func:`.rfftn` - real-to-complex forward FFT
    * :func:`.irfftn` - complex-to-real backward FFT
    * :func:`.dctn` - real-to-real Discrete Cosine Transform (DCT)
    * :func:`.idctn` - real-to-real inverse DCT
    * :func:`.dstn` - real-to-real Discrete Sine Transform (DST)
    * :func:`.idstn` - real-to-real inverse DST
    * :func:`.hfftn` - complex-to-real forward FFT with Hermitian symmetry
    * :func:`.ihfftn` - real-to-complex backward FFT with Hermitian symmetry 

All these transform functions return instances of one of the classes 
:class:`.fftwf_xfftn.FFT`, :class:`.fftw_xfftn.FFT` or :class:`.fftwl_xfftn.FFT`, 
depending on the requested precision being single, double or long double, 
respectively. Except from precision, the tree classes are identical.
All transforms are non-normalized by default. Note that all these functions
are *planners*. They do not execute the transforms, they simply return an 
instance of a class that can do it. See docstrings of each function for usage.
For quick reference, the 2D transform :ref:`shown for Numpy <numpy2d>` can be 
done using :mod:`.fftw` as::

    from mpi4py_fft.fftw import rfftn as plan_rfftn, irfftn as plan_irfftn
    from mpi4py_fft.fftw import FFTW_ESTIMATE
    rfftn = plan_rfftn(u.copy(), flags=(FFTW_ESTIMATE,))
    irfftn = plan_irfftn(u_hat.copy(), flags=(FFTW_ESTIMATE,))
    u_hat = rfftn(uc, normalize=True)
    uu = irfftn(u_hat)
    assert np.allclose(uu, uc)

Note that since all the functions in the above list are planners, an extra step
is required in comparison with Numpy. Also note that we are using copies of
the ``u`` and ``u_hat`` arrays in creating the plans. This is done
because the provided arrays will be used under the hood as work arrays for
the :func:`.rfftn` and :func:`.irfftn` functions, and the work arrays may
be destroyed upon creation.

The real-to-real transforms are by FFTW defined as one of (see `definitions <http://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transform-Kinds.html#Real_002dto_002dReal-Transform-Kinds>`_ and `extended definitions <http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html#What-FFTW-Really-Computes>`_) 

    * FFTW_REDFT00
    * FFTW_REDFT01
    * FFTW_REDFT10
    * FFTW_REDFT11
    * FFTW_RODFT00
    * FFTW_RODFT01
    * FFTW_RODFT10
    * FFTW_RODFT11

Different real-to-real cosine and sine transforms may be combined into one
object using :func:`.factory.get_planned_FFT` with a list of different 
transform kinds. However, it is not possible to combine, in one single
object, real-to-real transforms with real-to-complex. For such transforms
more than one object is required.


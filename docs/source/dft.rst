Discrete Fourier Transforms
---------------------------

Consider first two one-dimensional arrays :math:`\{u_j\}_{j=0}^{N-1}` and
:math:`\{\hat{u}_k\}_{k=0}^{N-1}`. We define the forward and backward Discrete
Fourier transforms (DFT), respectively, as

.. math::
    :label: dft

    \hat{u}_k &= \frac{1}{N}\sum_{j=0}^{N-1}u_j e^{-2\pi i j k / N}, \quad \forall \, k=0, 1, \ldots, N-1, \\
    u_j &= \sum_{k=0}^{N-1}\hat{u}_k e^{2\pi i j k / N}, \quad \forall \, j=0, 1, \ldots, N-1,

where :math:`i=\sqrt{-1}`. Discrete Fourier transforms are computed efficiently
using algorithms termed Fast Fourier Transforms, known in short as FFTs. 
Numpy, Scipy, and many other scientific softwares contain implementations that 
make working with Fourier series simple and straight forward. 

With multidimensional arrays, that we will get to soon, it becomes necessary to 
use different index sets for each dimension of the multidimensional arrays, and we 
will use a bold font to represent these sets. For our 1D transforms we get 
:math:`\textbf{j}=[0, 1, \ldots, N-1]` and :math:`\textbf{k}=[0, 1, \ldots, N-1]`, 
that are used to write the above equations as 

.. math::

    \hat{u}_k &= \frac{1}{N}\sum_{j\in\textbf{j}}u_j e^{-2\pi i j k / N}, \quad \forall \, k\in\textbf{k}, \\
    u_j &= \sum_{k\in\textbf{k}}\hat{u}_k e^{2\pi i j k / N}, \quad \forall \, j\in\textbf{j},

A bold font is used to represent the entire arrays:
:math:`\boldsymbol{u}=\{u_j\}_{j\in\textbf{j}}` and 
:math:`\boldsymbol{\hat{u}}=\{\hat{u}_k\}_{k\in\textbf{k}}`. This extends also
to multidimensional arrays.

An more compact notation is commonly used for the DFTs, and the 1D 
forward and backward transforms can then alternatively be written as

.. math::

    \boldsymbol{\hat{u}} &= \mathcal{F}(\boldsymbol{u}), \\
    \boldsymbol{u} &= \mathcal{F}^{-1}(\boldsymbol{\hat{u}}).

These 1D Fourier transforms can be implemented easily with just Numpy as::

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
`spectral Galerkin method <https://github.com/spectralDNS/shenfun>`_. See also


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
of the given kind. You determine yourselves how much effort to put into this
planning, providing flags from the list: 
(FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE), where the effort
is increasing from the lowest (FFTW_ESTIMATE) to the highest (FFTW_EXHAUSTIVE).

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
two-dimensional DFT can be defined as

.. math::
    :label: 2dfourier

    \hat{u}_{k_0,k_1} &= \frac{1}{N_0}\sum_{j_0 \in \textbf{j}_0}\Big( e^{-2\pi i \omega_0} \frac{1}{N_1} \sum_{j_1\in \textbf{j}_1} \Big( e^{-2\pi i \omega_1} u_{j_0,j_1}\Big) \Big), \quad \forall \, (k_0, k_1) \in \textbf{k}_0  \times \textbf{k}_1, \\
    u_{j_0, j_1} &= \sum_{k_1\in \textbf{k}_1} \Big( e^{2\pi i \omega_1} \sum_{k_0\in\textbf{k}_0} \Big(  e^{2\pi i \omega_0} \hat{u}_{k_0, k_1} \Big) \Big), \quad \forall \, (j_0, j_1) \in \textbf{j}_0 \times \textbf{j}_1.

Note that the forward transform corresponds to taking the 1D Fourier 
transform first along axis 1, once for each of the indices in :math:`j_0`. 
Afterwords the transform is executed along axis 0. The two steps are more 
easily understood if we break things up a little bit and write the forward
transform in :eq:`2dfourier` as

.. math::

    \tilde{u}_{j_0,k_1} &= \frac{1}{N_1}\sum_{j_1 \in \textbf{j}_1} u_{j_0,j_1} e^{-2\pi i \omega_1}, \quad \forall \, k_1 \in \textbf{k}_1, \\
    \hat{u}_{k_0,k_1} &= \frac{1}{N_0}\sum_{j_0 \in \textbf{j}_0} \tilde{u}_{j_0,k_1} e^{-2\pi i \omega_0}, \quad \forall \, k_0 \in \textbf{k}_0.

The inverse transform
if performed in the other order, axis 0 first and then 1. The order is actually
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

    \boldsymbol{\hat{u}} &= \mathcal{F}_0(\mathcal{F}_1(\boldsymbol{u})), \\
    \boldsymbol{u} &= \mathcal{F}_1^{-1}(\mathcal{F}_0^{-1}(\boldsymbol{\hat{u}})).


Extension to multiple dimensions is straight forward. We denote a :math:`d`-dimensional
array as :math:`u_{j_0, j_1, \ldots, j_{d-1}}` and a partial transform of :math:`u` 
along axis :math:`i` is denoted as

.. math::

    \tilde{u}_{j_0, \ldots, k_i, \ldots, j_{d-1}} = \mathcal{F}_i(u_{j_0, \ldots, j_i, \ldots, j_d})

We get the multidimensional transforms on short form still as :eq:`dft_short`, and
with partial transforms as

.. math::

    \boldsymbol{\hat{u}} &= \mathcal{F}_0(\mathcal{F}_1( \ldots \mathcal{F}_{d-1}(\boldsymbol{u})), \\
    \boldsymbol{u} &= \mathcal{F}_{d-1}^{-1}( \mathcal{F}_{d-2}^{-1}( \ldots \mathcal{F}_0^{-1}(\boldsymbol{\hat{u}}))).

Multidimensional transforms are straightforward to implement in Numpy::

    import numpy as np
    M, N = 16, 16
    u = np.random.random((M, N))
    u_hat = np.fft.fftn(u)
    uc = np.fft.ifftn(u_hat)
    assert np.allclose(u, uc)




"""
Demo program that solves the Navier Stokes equations in a triply
periodic domain. The solution is initialized using the Taylor-Green
vortex and evolved in time with a 4'th order Runge Kutta method.

"""
from numpy import array, pi, ndarray, where, sin, cos, sum, mgrid, meshgrid, \
    fft, ndim
from mpi4py_fft.mpifft import MPI, PFFT


class DistributedArray(ndarray):
    """MPI distributed numpy array

    Parameters
    ----------

    pfft : Instance of the PFFT class
    put : ('input', 'output')
        Array of type input or output wrt PFFT
    tensor : int or tuple of ints
        To create tensors of arrays
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    buffer : object exposing buffer interface, optional
        Used to fill the array with data.
    offset : int, optional
        Offset of array data in buffer.
    strides : tuple of ints, optional
        Strides of data in memory.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.
    val : int or float
        Value used to initialize array

    For more information, see numpy.ndarray

    Examples
    --------

    from mpi4py_fft import MPI, PFFT

    FFT = PFFT(MPI.COMM_WORLD, [64, 64, 64])
    u = DistributedArray(FFT, 'input', tensor=3)
    uhat = DistributedArray(FFT, 'output', tensor=3)

    """

    # pylint: disable=too-few-public-methods,too-many-arguments

    def __new__(cls, pfft, put='input', tensor=None, buffer=None, offset=0,
                strides=None, order=None, val=0):
        local_shape = pfft.forward.input_array.shape
        dtype = pfft.forward.input_array.dtype
        if put == 'output':
            local_shape = pfft.forward.output_array.shape
            dtype = pfft.forward.output_array.dtype
        if not tensor is None:
            tensor = list(tensor) if ndim(tensor) else [tensor]
            local_shape = tensor + list(local_shape)
        obj = ndarray.__new__(cls,
                              local_shape,
                              dtype=dtype,
                              buffer=buffer,
                              offset=offset,
                              strides=strides,
                              order=order)
        obj.fill(val)
        return obj


def get_local_mesh(pfft):
    """Returns local mesh."""
    N = pfft.pencil[0].shape
    x1 = slice(pfft.forward.input_pencil.substart[0],
               pfft.forward.input_pencil.substart[0] +
               pfft.forward.input_pencil.subshape[0])

    x2 = slice(pfft.forward.input_pencil.substart[1],
               pfft.forward.input_pencil.substart[1] +
               pfft.forward.input_pencil.subshape[1])

    x = mgrid[x1, x2, :N[2]].astype(float)
    x[0] *= L[0]/N[0]
    x[1] *= L[1]/N[1]
    x[2] *= L[2]/N[2]
    return x


def get_local_wavenumbermesh(pfft):
    """Returns local wavenumber mesh."""

    x1 = slice(pfft.backward.input_pencil.substart[2],
               pfft.backward.input_pencil.substart[2] +
               pfft.backward.input_pencil.subshape[2])

    x2 = slice(pfft.backward.input_pencil.substart[1],
               pfft.backward.input_pencil.substart[1] +
               pfft.backward.input_pencil.subshape[1])

    # Set wavenumbers in grid
    kx = fft.fftfreq(N[0], 1./N[0]).astype(int)
    ky = fft.fftfreq(N[1], 1./N[1]).astype(int)
    kz = fft.rfftfreq(N[2], 1./N[2]).astype(int)
    wavemesh = array(meshgrid(kx, ky[x2], kz[x1], indexing='ij'), dtype=float)
    return wavemesh


def get_scaled_local_wavenumbermesh(pfft):
    """Returns scaled local wavenumber mesh.

    Maps physical domain to a computational cube of size (2pi)**3.

    """
    wavemesh = get_local_wavenumbermesh(pfft)
    Lp = 2*pi/L
    for j in range(3):
        wavemesh[j] *= Lp[j]
    return wavemesh


# Set viscosity, end time and time step
nu = 0.000625
T = 0.1
dt = 0.01

# Set global size of the computational box
M = 6
N = [2**M, 2**(M), 2**M]
L = array([2*pi, 4*pi, 4*pi], dtype=float) # Needs to be (2*int)*pi in all directions (periodic)

# Create instance of PFFT to perform parallel FFT
FFT = PFFT(MPI.COMM_WORLD, N)
FFT_pad = PFFT(MPI.COMM_WORLD, N, padding=True)

# Declare variables needed to solve Navier-Stokes
U = DistributedArray(FFT, 'input', tensor=3)        # Velocity
U_hat = DistributedArray(FFT, 'output', tensor=3)   # Velocity transformed
P = DistributedArray(FFT, 'input')                  # Pressure (scalar)
P_hat = DistributedArray(FFT, 'output')             # Pressure transformed
U_hat0 = DistributedArray(FFT, 'output', tensor=3)  # Runge-Kutta work array
U_hat1 = DistributedArray(FFT, 'output', tensor=3)  # Runge-Kutta work array
a = [1./6., 1./3., 1./3., 1./6.]                    # Runge-Kutta parameter
b = [0.5, 0.5, 1.]                                  # Runge-Kutta parameter
dU = DistributedArray(FFT, 'output', tensor=3)      # Right hand side of ODEs
curl = DistributedArray(FFT, 'input', tensor=3)

U_pad = DistributedArray(FFT_pad, 'input', tensor=3)
curl_pad = DistributedArray(FFT_pad, 'input', tensor=3)

X = get_local_mesh(FFT)
K = get_scaled_local_wavenumbermesh(FFT)
K2 = sum(K*K, 0, dtype=float)
K_over_K2 = K.astype(float) / where(K2 == 0, 1, K2).astype(float)


def cross(x, y, z):
    """Cross product z = x \times y"""
    z[0] = FFT_pad.forward(x[1]*y[2]-x[2]*y[1], z[0])
    z[1] = FFT_pad.forward(x[2]*y[0]-x[0]*y[2], z[1])
    z[2] = FFT_pad.forward(x[0]*y[1]-x[1]*y[0], z[2])
    return z


def compute_curl(x, z):
    z[2] = FFT_pad.backward(1j*(K[0]*x[1]-K[1]*x[0]), z[2])
    z[1] = FFT_pad.backward(1j*(K[2]*x[0]-K[0]*x[2]), z[1])
    z[0] = FFT_pad.backward(1j*(K[1]*x[2]-K[2]*x[1]), z[0])
    return z


def compute_rhs(rhs):
    for j in range(3):
        U_pad[j] = FFT_pad.backward(U_hat[j], U_pad[j])

    curl_pad[:] = compute_curl(U_hat, curl_pad)
    rhs = cross(U_pad, curl_pad, rhs)
    P_hat[:] = sum(rhs*K_over_K2, 0, out=P_hat)
    rhs -= P_hat*K
    rhs -= nu*K2*U_hat
    return rhs


# Initialize a Taylor Green vortex
U[0] = sin(X[0])*cos(X[1])*cos(X[2])
U[1] = -cos(X[0])*sin(X[1])*cos(X[2])
U[2] = 0
for i in range(3):
    U_hat[i] = FFT.forward(U[i], U_hat[i])

# Integrate using a 4th order Rung-Kutta method
t = 0.0
tstep = 0
while t < T-1e-8:
    t += dt
    tstep += 1
    U_hat1[:] = U_hat0[:] = U_hat
    for rk in range(4):
        dU = compute_rhs(dU)
        if rk < 3:
            U_hat[:] = U_hat0 + b[rk]*dt*dU
        U_hat1[:] += a[rk]*dt*dU
    U_hat[:] = U_hat1[:]

    for i in range(3):
        U[i] = FFT.backward(U_hat[i], U[i])
    k = MPI.COMM_WORLD.reduce(sum(U*U)/N[0]/N[1]/N[2]/2)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Energy = {}".format(k))

# Transform result to real physical space
for i in range(3):
    U[i] = FFT.backward(U_hat[i], U[i])

# Check energy
k = MPI.COMM_WORLD.reduce(sum(U*U)/N[0]/N[1]/N[2]/2)
if MPI.COMM_WORLD.Get_rank() == 0:
    assert round(float(k) - 0.124953117517, 7) == 0

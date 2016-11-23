"""
Demo program that solves the Navier Stokes equations in a triply
periodic domain. The solution is initialized using the Taylor-Green
vortex and evolved in time with a 4'th order Runge Kutta method.

"""
from numpy import array, pi, empty, where, sin, cos, sum, mgrid, meshgrid, fft
from mpi4py_fft.mpifft import MPI, PFFT
#from mpi4py_fft.pencil import distribution

# Set viscosity, end time and time step
nu = 0.000625
T = 0.1
dt = 0.01

# Set global size of the computational box
M = 5
N = array([2**M, 2**(M+1)+1, 2**M+1], dtype=int)
L = array([2*pi, 4*pi, 4*pi], dtype=float) # Needs to be (2*int)*pi in all directions because of periodicity

FFT = PFFT(MPI.COMM_WORLD, N)

# Some helper functions 
def real_shape(FFT):
    return FFT.forward.input_array.shape

def complex_shape(FFT):
    return FFT.forward.output_array.shape

def get_local_mesh(FFT):

    x1 = slice(FFT.forward.input_pencil.substart[0], 
               FFT.forward.input_pencil.substart[0]+FFT.forward.input_pencil.subshape[0])

    x2 = slice(FFT.forward.input_pencil.substart[1], 
               FFT.forward.input_pencil.substart[1]+FFT.forward.input_pencil.subshape[1])

    X = mgrid[x1, x2, :N[2]].astype(float)
    X[0] *= L[0]/N[0]
    X[1] *= L[1]/N[1]
    X[2] *= L[2]/N[2]
    return X

def get_local_wavenumbermesh(FFT):
    
    x1 = slice(FFT.backward.input_pencil.substart[0], 
               FFT.backward.input_pencil.substart[0]+FFT.backward.input_pencil.subshape[0])

    x2 = slice(FFT.backward.input_pencil.substart[1], 
               FFT.backward.input_pencil.substart[1]+FFT.backward.input_pencil.subshape[1])

    # Set wavenumbers in grid
    kx = fft.fftfreq(N[0], 1./N[0]).astype(int)
    ky = fft.fftfreq(N[1], 1./N[1]).astype(int)
    kz = fft.rfftfreq(N[2], 1./N[2]).astype(int)
    K = array(meshgrid(kx, ky[x2], kz[x1], indexing='ij'), dtype=float)
    return K

def get_scaled_local_wavenumbermesh(FFT):
    K = get_local_wavenumbermesh(FFT)
    # Scale with physical mesh size. This takes care of mapping the physical domain to a computational cube of size (2pi)**3
    Lp = 2*pi/L
    for i in range(3):
        K[i] *= Lp[i]
    return K

# Declare variables needed to solve Navier-Stokes
U = empty((3,) + real_shape(FFT))  # Velocity
U_hat = empty((3,) + complex_shape(FFT), dtype=complex) # Velocity transformed
P = empty(real_shape(FFT)) # Pressure
P_hat = empty(complex_shape(FFT), dtype=complex) # Pressure transformed
U_hat0 = empty((3,) + complex_shape(FFT), dtype=complex)  # For Runge-Kutta
U_hat1 = empty((3,) + complex_shape(FFT), dtype=complex)  # For Runge-Kutta
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.]
rhs = empty((3,) + complex_shape(FFT), dtype=complex)     # Right hand side of ODEs
curl = empty((3,) + real_shape(FFT))

X = get_local_mesh(FFT)
K = get_scaled_local_wavenumbermesh(FFT)
K2 = sum(K*K, 0, dtype=float)
K_over_K2 = K.astype(float) / where(K2 == 0, 1, K2).astype(float)

def cross(x, y, z):
    """Cross product z = x \times y"""
    z[0] = FFT.forward(x[1]*y[2]-x[2]*y[1])
    z[1] = FFT.forward(x[2]*y[0]-x[0]*y[2])
    z[2] = FFT.forward(x[0]*y[1]-x[1]*y[0])
    return z

def compute_curl(x, z):
    z[2] = FFT.backward(1j*(K[0]*x[1]-K[1]*x[0]))
    z[1] = FFT.backward(1j*(K[2]*x[0]-K[0]*x[2]))
    z[0] = FFT.backward(1j*(K[1]*x[2]-K[2]*x[1]))
    return z

def compute_rhs(rhs):
    for i in range(3):
        U[i] = FFT.backward(U_hat[i])

    curl[:] = compute_curl(U_hat, curl)
    rhs = cross(U, curl, rhs)
    P_hat[:] = sum(rhs*K_over_K2, 0, out=P_hat)
    rhs -= P_hat*K
    rhs -= nu*K2*U_hat
    return rhs

# Initialize a Taylor Green vortex
U[0] = sin(X[0])*cos(X[1])*cos(X[2])
U[1] = -cos(X[0])*sin(X[1])*cos(X[2])
U[2] = 0
for i in range(3):
    U_hat[i] = FFT.forward(U[i])

# Integrate using a 4th order Rung-Kutta method
t = 0.0
tstep = 0
while t < T-1e-8:
    t += dt
    tstep += 1
    U_hat1[:] = U_hat0[:] = U_hat
    for rk in range(4):
        rhs = compute_rhs(rhs)
        if rk < 3:
            U_hat[:] = U_hat0 + b[rk]*dt*rhs
        U_hat1[:] += a[rk]*dt*rhs
    U_hat[:] = U_hat1[:]

# Transform result to real physical space
for i in range(3):
    U[i] = FFT.backward(U_hat[i])

# Check energy
k = MPI.COMM_WORLD.reduce(sum(U*U)/N[0]/N[1]/N[2]/2)
if MPI.COMM_WORLD.Get_rank() == 0:
    assert round(k - 0.124953117517, 7) == 0

"""
Demo program that solves the Navier Stokes equations in a triply
periodic domain. The solution is initialized using the Taylor-Green
vortex and evolved in time with a 4'th order Runge Kutta method.

"""
from numpy import array, pi, where, sin, cos, sum, fft
from mpi4py_fft.mpifft import MPI, PFFT, Function
from time import time


# Set viscosity, end time and time step
nu = 0.000625
T = 0.1
dt = 0.01

# Set global size of the computational box
M = 6
N = [2**M, 2**M, 2**M]
L = array([2*pi, 4*pi, 4*pi], dtype=float) # Needs to be (2*int)*pi in all directions (periodic) because of initialization

# Create instance of PFFT to perform parallel FFT + an instance to do FFT with padding (3/2-rule)
FFT = PFFT(MPI.COMM_WORLD, N, collapse=False)
#FFT_pad = PFFT(MPI.COMM_WORLD, N, padding=[1.5, 1.5, 1.5])
FFT_pad = FFT

# Declare variables needed to solve Navier-Stokes
U = Function(FFT, False, tensor=3)       # Velocity
U_hat = Function(FFT, tensor=3)          # Velocity transformed
P = Function(FFT, False)                 # Pressure (scalar)
P_hat = Function(FFT)                    # Pressure transformed
U_hat0 = Function(FFT, tensor=3)         # Runge-Kutta work array
U_hat1 = Function(FFT, tensor=3)         # Runge-Kutta work array
a = [1./6., 1./3., 1./3., 1./6.]         # Runge-Kutta parameter
b = [0.5, 0.5, 1.]                       # Runge-Kutta parameter
dU = Function(FFT, tensor=3)             # Right hand side of ODEs
curl = Function(FFT, False, tensor=3)

U_pad = Function(FFT_pad, False, tensor=3)
curl_pad = Function(FFT_pad, False, tensor=3)

X = FFT.get_local_mesh(L)
K = FFT.get_local_wavenumbermesh(L)
K = array(K).astype(float)
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
t0 = time()
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
    #k = MPI.COMM_WORLD.reduce(sum(U*U)/N[0]/N[1]/N[2]/2)
    #if MPI.COMM_WORLD.Get_rank() == 0:
        #print("Energy = {}".format(k))

## Transform result to real physical space
#for i in range(3):
    #U[i] = FFT.backward(U_hat[i], U[i])

# Check energy
k = MPI.COMM_WORLD.reduce(sum(U*U)/N[0]/N[1]/N[2]/2)
if MPI.COMM_WORLD.Get_rank() == 0:
    print('Time = {}'.format(time()-t0))
    assert round(float(k) - 0.124953117517, 7) == 0

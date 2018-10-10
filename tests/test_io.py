from mpi4py import MPI
import numpy as np
import pytest
from mpi4py_fft import PFFT, HDF5Writer, HDF5Reader, Function, generate_xdmf

N = (12, 13, 14, 15)
comm = MPI.COMM_WORLD

skip = False
try:
    import h5py
except ImportError:
    skip = True

ex = {True: 'c', False: 'r'}

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
def test_regular_2D(forward_output):
    T = PFFT(comm, (N[0], N[1]))
    for i, domain in enumerate([((0, np.pi), (0, 2*np.pi)),
                                (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
                                 np.arange(N[1], dtype=np.float)*2*np.pi/N[1])]):
        filename = 'h5test_{}{}.h5'.format(ex[i == 0], ex[forward_output])
        hfile = HDF5Writer(filename, 'u', T, domain=domain)
        u = Function(T, forward_output=forward_output, val=1)
        hfile.write_step(0, u, forward_output)
        hfile.write_step(1, u, forward_output)
        hfile.close()
        if not forward_output:
            generate_xdmf(filename)

        u0 = Function(T, forward_output=forward_output)
        reader = HDF5Reader(filename, T)
        reader.read(u0, '/u/2D/0', forward_output)
        assert np.allclose(u0, u)
        reader.close()

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
def test_regular_3D(forward_output):
    T = PFFT(comm, (N[0], N[1], N[2]))
    d0 = ((0, np.pi), (0, 2*np.pi), (0, 3*np.pi))
    d1 = (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
          np.arange(N[1], dtype=np.float)*2*np.pi/N[1],
          np.arange(N[2], dtype=np.float)*2*np.pi/N[2])
    for i, domain in enumerate([d0, d1]):
        filename = 'h5test_{}{}.h5'.format(ex[i == 0], ex[forward_output])
        h0file = HDF5Writer('uv'+filename, ['u', 'v'], T, domain)
        h1file = HDF5Writer('v'+filename, ['u'], T, domain)
        u = Function(T, forward_output=forward_output)
        v = Function(T, forward_output=forward_output)
        u[:] = np.random.random(u.shape)
        v[:] = 2
        for k in range(3):
            h0file.write_step(k, [u, v], forward_output)
            h1file.write_step(k, v, forward_output)
            h0file.write_slice_step(k, [slice(None), 4, slice(None)], [u, v], forward_output)
            h0file.write_slice_step(k, [slice(None), 4, 4], [u, v], forward_output)
            h1file.write_slice_step(k, [slice(None), 4, slice(None)], v, forward_output)
            h1file.write_slice_step(k, [slice(None), 4, 4], v, forward_output)
        h0file.close()
        h1file.close()
        if not forward_output:
            generate_xdmf('uv'+filename)
            generate_xdmf('v'+filename)

        u0 = Function(T, forward_output=forward_output)
        reader = HDF5Reader('uv'+filename, T)
        reader.read(u0, '/u/3D/0', forward_output)
        assert np.allclose(u0, u)
        reader.read(u0, '/v/3D/0', forward_output)
        assert np.allclose(u0, v)
        reader.close()


if __name__ == '__main__':
    test_regular_3D(False)
    #test_regular_2D(False)

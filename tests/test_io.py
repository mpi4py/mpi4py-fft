from mpi4py import MPI
import numpy as np
from mpi4py_fft import PFFT, HDF5Writer, HDF5Reader, Function, generate_xdmf, \
    NCWriter, NCReader

N = (12, 13, 14, 15)
comm = MPI.COMM_WORLD

ex = {True: 'c', False: 'r'}

def test_h5py_2D(forward_output):
    T = PFFT(comm, (N[0], N[1]))
    for i, domain in enumerate([((0, np.pi), (0, 2*np.pi)),
                                (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
                                 np.arange(N[1], dtype=np.float)*2*np.pi/N[1])]):
        filename = 'h5test_{}{}.h5'.format(ex[i == 0], ex[forward_output])
        hfile = HDF5Writer(filename, T, domain=domain)
        u = Function(T, forward_output=forward_output, val=1)
        hfile.write(0, {'u': [u]}, forward_output)
        hfile.write(1, {'u': [u]}, forward_output)
        hfile.close()
        if not forward_output:
            generate_xdmf(filename)

        u0 = Function(T, forward_output=forward_output)
        reader = HDF5Reader(filename, T)
        reader.read(u0, '/u/2D/0', forward_output)
        assert np.allclose(u0, u)
        reader.close()

def test_h5py_3D(forward_output):
    T = PFFT(comm, (N[0], N[1], N[2]))
    d0 = ((0, np.pi), (0, 2*np.pi), (0, 3*np.pi))
    d1 = (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
          np.arange(N[1], dtype=np.float)*2*np.pi/N[1],
          np.arange(N[2], dtype=np.float)*2*np.pi/N[2])
    for i, domain in enumerate([d0, d1]):
        filename = 'h5test_{}{}.h5'.format(ex[i == 0], ex[forward_output])
        h0file = HDF5Writer('uv'+filename, T, domain)
        h1file = HDF5Writer('v'+filename, T, domain)
        u = Function(T, forward_output=forward_output)
        v = Function(T, forward_output=forward_output)
        u[:] = np.random.random(u.shape)
        v[:] = 2
        for k in range(3):
            h0file.write(k, {'u': [u,
                                   (u, [slice(None), 4, slice(None)]),
                                   (u, [slice(None), slice(None), 5])],
                             'v': [v, (v, [slice(None), slice(None), 5])]},
                         forward_output)
            h1file.write(k, {'v': [v,
                                   (v, [slice(None), slice(None), 2]),
                                   (v, [4, 4, slice(None)])]},
                         forward_output)
        h0file.close()
        h1file.close()
        if not forward_output:
            generate_xdmf('uv'+filename)
            generate_xdmf('v'+filename, periodic=False)
            generate_xdmf('v'+filename, periodic=(True, True, True))

        u0 = Function(T, forward_output=forward_output)
        reader = HDF5Reader('uv'+filename, T)
        reader.read(u0, '/u/3D/0', forward_output)
        assert np.allclose(u0, u)
        reader.read(u0, '/v/3D/0', forward_output)
        assert np.allclose(u0, v)
        reader.close()

def test_h5py_4D(forward_output):
    T = PFFT(comm, (N[0], N[1], N[2], N[3]))
    d0 = ((0, np.pi), (0, 2*np.pi), (0, 3*np.pi), (0, 4*np.pi))
    d1 = (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
          np.arange(N[1], dtype=np.float)*2*np.pi/N[1],
          np.arange(N[2], dtype=np.float)*2*np.pi/N[2],
          np.arange(N[3], dtype=np.float)*2*np.pi/N[3]
          )
    for i, domain in enumerate([d0, d1]):
        filename = 'h5test4_{}{}.h5'.format(ex[i == 0], ex[forward_output])
        h0file = HDF5Writer('uv'+filename, T, domain)
        u = Function(T, forward_output=forward_output)
        v = Function(T, forward_output=forward_output)
        u[:] = np.random.random(u.shape)
        v[:] = 2
        for k in range(3):
            h0file.write(k, {'u': [u, (u, [slice(None), 4, slice(None), slice(None)])],
                             'v': [v, (v, [slice(None), slice(None), 5, 6])]},
                         forward_output)
        h0file.close()
        if not forward_output:
            generate_xdmf('uv'+filename)

        u0 = Function(T, forward_output=forward_output)
        reader = HDF5Reader('uv'+filename, T)
        reader.read(u0, '/u/4D/0', forward_output)
        assert np.allclose(u0, u)
        reader.read(u0, '/v/4D/0', forward_output)
        assert np.allclose(u0, v)
        reader.close()

def test_netcdf_2D():
    T = PFFT(comm, (N[0], N[1]))
    for i, domain in enumerate([((0, np.pi), (0, 2*np.pi)),
                                (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
                                 np.arange(N[1], dtype=np.float)*2*np.pi/N[1])]):
        filename = 'nctest_{}.nc'.format(ex[i == 0])
        hfile = NCWriter(filename, T, domain=domain)
        u = Function(T, forward_output=False, val=1)
        hfile.write(0, {'u': [u]})
        hfile.write(1, {'u': [u]})
        hfile.close()

        u0 = Function(T, forward_output=False)
        reader = NCReader(filename, T)
        reader.read(u0, 'u', 0)
        assert np.allclose(u0, u)
        reader.close()

def test_netcdf_3D():
    T = PFFT(comm, (N[0], N[1], N[2]))
    d0 = ((0, np.pi), (0, 2*np.pi), (0, 3*np.pi))
    d1 = (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
          np.arange(N[1], dtype=np.float)*2*np.pi/N[1],
          np.arange(N[2], dtype=np.float)*2*np.pi/N[2])
    for i, domain in enumerate([d0, d1]):
        filename = 'nctest3_{}.nc'.format(ex[i == 0])
        h0file = NCWriter('uv'+filename, T, domain)
        h1file = NCWriter('v'+filename, T, domain)
        u = Function(T, forward_output=False)
        v = Function(T, forward_output=False)
        u[:] = np.random.random(u.shape)
        v[:] = 2
        for k in range(3):
            h0file.write(k, {'u': [u,
                                   (u, np.s_[:, :, 4]),
                                   (u, np.s_[:, 4, :])],
                             'v': [v, (v, np.s_[:, 4, :])]})
            h1file.write(k, {'v': [v, (v, np.s_[:, :, 3])]})
        h0file.close()
        h1file.close()

        u0 = Function(T, forward_output=False)
        reader = NCReader('uv'+filename, T)
        reader.read(u0, 'u', 0)
        assert np.allclose(u0, u)
        reader.read(u0, 'v', 0)
        assert np.allclose(u0, v)
        reader.close()

if __name__ == '__main__':
    skip = False
    try:
        import h5py
    except ImportError:
        skip = True

    if not skip:
        test_h5py_3D(False)
        test_h5py_3D(True)
        test_h5py_2D(False)
        test_h5py_2D(True)
        test_h5py_4D(False)
        test_netcdf_2D()
        if comm.Get_size() <= 2:
            test_netcdf_3D()

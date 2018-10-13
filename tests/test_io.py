from mpi4py import MPI
import numpy as np
from mpi4py_fft import PFFT, HDF5Writer, HDF5Reader, Function, generate_xdmf, \
    NCWriter, NCReader

N = (12, 13, 14, 15)
comm = MPI.COMM_WORLD

ex = {True: 'c', False: 'r'}

writer = {'hdf5': HDF5Writer,
          'netcdf': NCWriter}
reader = {'hdf5': HDF5Reader,
          'netcdf': NCReader}
ending = {'hdf5': '.h5', 'netcdf': '.nc'}

def test_2D(backend, forward_output):
    if backend == 'netcdf':
        assert forward_output is False
    T = PFFT(comm, (N[0], N[1]))
    for i, domain in enumerate([((0, np.pi), (0, 2*np.pi)),
                                (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
                                 np.arange(N[1], dtype=np.float)*2*np.pi/N[1])]):
        filename = "".join(('test2D_{}{}'.format(ex[i == 0], ex[forward_output]),
                            ending[backend]))
        hfile = writer[backend](filename, T, domain=domain)
        u = Function(T, forward_output=forward_output, val=1)
        hfile.write(0, {'u': [u]}, forward_output=forward_output)
        hfile.write(1, {'u': [u]}, forward_output=forward_output)
        hfile.close()
        if not forward_output and backend == 'hdf5':
            generate_xdmf(filename)

        u0 = Function(T, forward_output=forward_output)
        read = reader[backend](filename, T)
        fl = {'hdf5': '/u/2D/0', 'netcdf': 'u'}
        read.read(u0, fl[backend], step=0, forward_output=forward_output)
        assert np.allclose(u0, u)
        read.close()

def test_3D(backend, forward_output):
    if backend == 'netcdf':
        assert forward_output is False
    T = PFFT(comm, (N[0], N[1], N[2]))
    d0 = ((0, np.pi), (0, 2*np.pi), (0, 3*np.pi))
    d1 = (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
          np.arange(N[1], dtype=np.float)*2*np.pi/N[1],
          np.arange(N[2], dtype=np.float)*2*np.pi/N[2])
    for i, domain in enumerate([d0, d1]):
        filename = ''.join(('test_{}{}'.format(ex[i == 0], ex[forward_output]),
                            ending[backend]))
        h0file = writer[backend]('uv'+filename, T, domain)
        h1file = writer[backend]('v'+filename, T, domain)
        u = Function(T, forward_output=forward_output)
        v = Function(T, forward_output=forward_output)
        u[:] = np.random.random(u.shape)
        v[:] = 2
        for k in range(3):
            h0file.write(k, {'u': [u,
                                   (u, [slice(None), 4, slice(None)]),
                                   (u, [slice(None), slice(None), 5])],
                             'v': [v, (v, [slice(None), slice(None), 5])]},
                         forward_output=forward_output)
            h1file.write(k, {'v': [v,
                                   (v, [slice(None), slice(None), 2]),
                                   (v, [4, 4, slice(None)])]},
                         forward_output=forward_output)
        h0file.close()
        h1file.close()
        if not forward_output and backend == 'hdf5':
            generate_xdmf('uv'+filename)
            generate_xdmf('v'+filename, periodic=False)
            generate_xdmf('v'+filename, periodic=(True, True, True))

        u0 = Function(T, forward_output=forward_output)
        read = reader[backend]('uv'+filename, T)
        fl = {'hdf5': '/u/3D/0', 'netcdf': 'u'}
        read.read(u0, fl[backend], forward_output=forward_output, step=0)
        assert np.allclose(u0, u)
        fl = {'hdf5': '/v/3D/0', 'netcdf': 'v'}
        read.read(u0, fl[backend], forward_output=forward_output, step=0)
        assert np.allclose(u0, v)
        read.close()

def test_4D(backend, forward_output):
    if backend == 'netcdf':
        assert forward_output is False
    T = PFFT(comm, (N[0], N[1], N[2], N[3]))
    d0 = ((0, np.pi), (0, 2*np.pi), (0, 3*np.pi), (0, 4*np.pi))
    d1 = (np.arange(N[0], dtype=np.float)*2*np.pi/N[0],
          np.arange(N[1], dtype=np.float)*2*np.pi/N[1],
          np.arange(N[2], dtype=np.float)*2*np.pi/N[2],
          np.arange(N[3], dtype=np.float)*2*np.pi/N[3]
          )
    for i, domain in enumerate([d0, d1]):
        filename = "".join(('h5test4_{}{}'.format(ex[i == 0], ex[forward_output]),
                            ending[backend]))
        h0file = writer[backend]('uv'+filename, T, domain)
        u = Function(T, forward_output=forward_output)
        v = Function(T, forward_output=forward_output)
        u[:] = np.random.random(u.shape)
        v[:] = 2
        for k in range(3):
            h0file.write(k, {'u': [u, (u, [slice(None), 4, slice(None), slice(None)])],
                             'v': [v, (v, [slice(None), slice(None), 5, 6])]},
                         forward_output=forward_output)
        h0file.close()
        if not forward_output and backend == 'hdf5':
            generate_xdmf('uv'+filename)

        u0 = Function(T, forward_output=forward_output)
        read = reader[backend]('uv'+filename, T)
        fl = {'hdf5': '/u/4D/0', 'netcdf': 'u'}
        read.read(u0, fl[backend], forward_output=forward_output, step=0)
        assert np.allclose(u0, u)
        fl = {'hdf5': '/v/4D/0', 'netcdf': 'v'}
        read.read(u0, fl[backend], forward_output=forward_output, step=0)
        assert np.allclose(u0, v)
        read.close()

if __name__ == '__main__':
    skip = False
    try:
        import h5py
    except ImportError:
        skip = True

    if not skip:
        for bnd in ('hdf5', 'netcdf'):
            forw_output = [False]
            if bnd == 'hdf5':
                forw_output.append(True)
            for kind in forw_output:
                if not (bnd == 'netcdf' and comm.Get_size() > 2):
                    test_3D(bnd, kind)
                test_2D(bnd, kind)
                if not bnd is 'netcdf':
                    test_4D(bnd, kind)

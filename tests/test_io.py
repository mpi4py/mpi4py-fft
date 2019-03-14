import functools
import os
from mpi4py import MPI
import numpy as np
from mpi4py_fft import PFFT, HDF5File, NCFile, newDistArray, generate_xdmf

N = (12, 13, 14, 15)
comm = MPI.COMM_WORLD

ex = {True: 'c', False: 'r'}

writer = {'hdf5': functools.partial(HDF5File, mode='w'),
          'netcdf4': functools.partial(NCFile, mode='w')}
reader = {'hdf5': functools.partial(HDF5File, mode='r'),
          'netcdf4': functools.partial(NCFile, mode='r')}
ending = {'hdf5': '.h5', 'netcdf4': '.nc'}

def remove_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def test_2D(backend, forward_output):
    if backend == 'netcdf4':
        assert forward_output is False
    T = PFFT(comm, (N[0], N[1]))
    for i, domain in enumerate([None, ((0, np.pi), (0, 2*np.pi)),
                                (np.arange(N[0], dtype=np.float)*1*np.pi/N[0],
                                 np.arange(N[1], dtype=np.float)*2*np.pi/N[1])]):
        for rank in range(3):
            filename = "".join(('test2D_{}{}{}'.format(ex[i == 0], ex[forward_output], rank),
                                ending[backend]))
            if backend == 'netcdf4':
                remove_if_exists(filename)
            u = newDistArray(T, forward_output=forward_output, val=1, rank=rank)
            hfile = writer[backend](filename, domain=domain)
            assert hfile.backend() == backend
            hfile.write(0, {'u': [u]})
            hfile.write(1, {'u': [u]})
            u.write(hfile, 'u', 2)
            if rank > 0:
                hfile.write(0, {'u': [u]}, as_scalar=True)
                hfile.write(1, {'u': [u]}, as_scalar=True)
                u.write(hfile, 'u', 2, as_scalar=True)
            u.write('t'+filename, 'u', 0)
            u.write('t'+filename, 'u', 0, [slice(None), 3])

            if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
                generate_xdmf(filename)
                generate_xdmf(filename, order='visit')

            u0 = newDistArray(T, forward_output=forward_output, rank=rank)
            read = reader[backend](filename, u=u0)
            read.read(u0, 'u', step=0)
            u0.read(filename, 'u', 2)
            u0.read(read, 'u', 2)
            assert np.allclose(u0, u)
    T.destroy()

def test_3D(backend, forward_output):
    if backend == 'netcdf4':
        assert forward_output is False
    T = PFFT(comm, (N[0], N[1], N[2]))
    d0 = ((0, np.pi), (0, 2*np.pi), (0, 3*np.pi))
    d1 = (np.arange(N[0], dtype=np.float)*1*np.pi/N[0],
          np.arange(N[1], dtype=np.float)*2*np.pi/N[1],
          np.arange(N[2], dtype=np.float)*3*np.pi/N[2])
    for i, domain in enumerate([None, d0, d1]):
        for rank in range(3):
            filename = ''.join(('test_{}{}{}'.format(ex[i == 0], ex[forward_output], rank),
                                ending[backend]))
            if backend == 'netcdf4':
                remove_if_exists('uv'+filename)
                remove_if_exists('v'+filename)

            u = newDistArray(T, forward_output=forward_output, rank=rank)
            v = newDistArray(T, forward_output=forward_output, rank=rank)
            h0file = writer[backend]('uv'+filename, domain=domain)
            h1file = writer[backend]('v'+filename, domain=domain)
            u[:] = np.random.random(u.shape)
            v[:] = 2
            for k in range(3):
                h0file.write(k, {'u': [u,
                                       (u, [slice(None), slice(None), 4]),
                                       (u, [5, 5, slice(None)])],
                                 'v': [v,
                                       (v, [slice(None), 6, slice(None)])]})
                h1file.write(k, {'v': [v,
                                       (v, [slice(None), 6, slice(None)]),
                                       (v, [6, 6, slice(None)])]})
            # One more time with same k
            h0file.write(k, {'u': [u,
                                   (u, [slice(None), slice(None), 4]),
                                   (u, [5, 5, slice(None)])],
                             'v': [v,
                                   (v, [slice(None), 6, slice(None)])]})
            h1file.write(k, {'v': [v,
                                   (v, [slice(None), 6, slice(None)]),
                                   (v, [6, 6, slice(None)])]})

            if rank > 0:
                for k in range(3):
                    u.write('uv'+filename, 'u', k, as_scalar=True)
                    u.write('uv'+filename, 'u', k, [slice(None), slice(None), 4], as_scalar=True)
                    u.write('uv'+filename, 'u', k, [5, 5, slice(None)], as_scalar=True)
                    v.write('uv'+filename, 'v', k, as_scalar=True)
                    v.write('uv'+filename, 'v', k, [slice(None), 6, slice(None)], as_scalar=True)

            if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
                generate_xdmf('uv'+filename)
                generate_xdmf('v'+filename, periodic=False)
                generate_xdmf('v'+filename, periodic=(True, True, True))
                generate_xdmf('v'+filename, order='visit')

            u0 = newDistArray(T, forward_output=forward_output, rank=rank)
            read = reader[backend]('uv'+filename, u=u0)
            read.read(u0, 'u', step=0)
            assert np.allclose(u0, u)
            read.read(u0, 'v', step=0)
            assert np.allclose(u0, v)
    T.destroy()

def test_4D(backend, forward_output):
    if backend == 'netcdf4':
        assert forward_output is False
    T = PFFT(comm, (N[0], N[1], N[2], N[3]))
    d0 = ((0, np.pi), (0, 2*np.pi), (0, 3*np.pi), (0, 4*np.pi))
    d1 = (np.arange(N[0], dtype=np.float)*1*np.pi/N[0],
          np.arange(N[1], dtype=np.float)*2*np.pi/N[1],
          np.arange(N[2], dtype=np.float)*3*np.pi/N[2],
          np.arange(N[3], dtype=np.float)*4*np.pi/N[3]
          )
    for i, domain in enumerate([None, d0, d1]):
        for rank in range(3):
            filename = "".join(('h5test4_{}{}{}'.format(ex[i == 0], ex[forward_output], rank),
                                ending[backend]))
            if backend == 'netcdf4':
                remove_if_exists('uv'+filename)
            u = newDistArray(T, forward_output=forward_output, rank=rank)
            v = newDistArray(T, forward_output=forward_output, rank=rank)
            h0file = writer[backend]('uv'+filename, domain=domain)
            u[:] = np.random.random(u.shape)
            v[:] = 2
            for k in range(3):
                h0file.write(k, {'u': [u, (u, [slice(None), 4, slice(None), slice(None)])],
                                 'v': [v, (v, [slice(None), slice(None), 5, 6])]})

            if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
                generate_xdmf('uv'+filename)

            u0 = newDistArray(T, forward_output=forward_output, rank=rank)
            read = reader[backend]('uv'+filename, u=u0)
            read.read(u0, 'u', step=0)
            assert np.allclose(u0, u)
            read.read(u0, 'v', step=0)
            assert np.allclose(u0, v)
    T.destroy()

if __name__ == '__main__':
    #pylint: disable=unused-import
    skip = {'hdf5': False, 'netcdf4': False}
    try:
        import h5py
    except ImportError:
        skip['hdf5'] = True
    try:
        import netCDF4
    except ImportError:
        skip['netcdf4'] = True
    skip['hdf5'] = True
    for bnd in ('hdf5', 'netcdf4'):
        if not skip[bnd]:
            forw_output = [False]
            if bnd == 'hdf5':
                forw_output.append(True)
            for kind in forw_output:
                test_3D(bnd, kind)
                test_2D(bnd, kind)
                if bnd == 'hdf5':
                    test_4D(bnd, kind)

import warnings
import numpy as np
from mpi4py import MPI

# https://github.com/Unidata/netcdf4-python/blob/master/examples/mpi_example.py

try:
    from netCDF4 import Dataset
except ImportError:
    warnings.warn('netcdf not installed')

__all__ = ('NCWriter', 'NCReader')

comm = MPI.COMM_WORLD

class NCWriter(object):
    """Class for writing data to netcdf format

    Parameters
    ----------
        ncname : str
            Name of netcdf file to be created
        names : list of strings
            Names of fields to be stored
        T : PFFT
            Instance of a :class:`.PFFT` class. Must be the same as the space
            used for storing with 'write_step' and 'write_slice_step'
        domain : dim-sequence of 2-tuples or arrays of coordinates
            Use dim-sequence of 2-tuples to give the size of the domain as
            origin and length, e.g., (0, 2*pi).
            Use dim-sequence of arrays if using a non-uniform mesh where the
            grid points must be specified. One array per direction.
        clobber : bool, optional
    """
    def __init__(self, ncname, names, T, domain, clobber=True, **kw):
        self.f = Dataset(ncname, mode="w", clobber=clobber, parallel=True, comm=comm, **kw)
        self.T = T
        self.N = N = T.shape(False)
        self.names = names
        dtype = self.T.dtype(False)
        assert dtype.char in 'fdg'
        self._dtype = dtype

        self.f.createDimension('time', None)
        self.dims = ['time']
        self.nc_t = self.f.createVariable('time', self._dtype, ('time'))
        self.nc_t.set_collective(True)
        self.slice_step = dict()

        d = list(domain)
        if not isinstance(domain[0], np.ndarray):
            assert len(domain[0]) == 2
            for i in range(len(domain)):
                d[i] = np.arange(N[i], dtype=np.float)*2*np.pi/N[i]

        for i in range(len(d)):
            xyz = {0:'x', 1:'y', 2:'z'}[i]
            self.f.createDimension(xyz, N[i])
            self.dims.append(xyz)
            nc_xyz = self.f.createVariable(xyz, self._dtype, (xyz))
            nc_xyz[:] = d[i]

        self.handles = dict()
        for i, name in enumerate(names):
            self.handles[name] = self.f.createVariable(name, self._dtype, self.dims)
            self.handles[name].set_collective(True)

        self.f.sync()

    def close(self):
        self.f.close()

    def write_step(self, step, fields):
        """Write field u to netcdf format at a given index step

        Parameters
        ----------
            step : int
                Index of field stored
            fields : list of arrays
                The fields to be stored
        """
        it = self.nc_t.size
        self.nc_t[it] = step

        if isinstance(fields, np.ndarray):
            fields = [fields]
        for name, field in zip(self.names, fields):
            self._write_group(name, field, it)

    def _write_group(self, name, u, it):
        s = self.T.local_slice(False)
        if self.T.ndim() == 3:
            self.handles[name][it, s[0], s[1], s[2]] = u
        elif self.T.ndim() == 2:
            self.handles[name][it, s[0], s[1]] = u
        else:
            raise NotImplementedError

        self.f.sync()

    def write_slice_step(self, step, sl, fields):
        """Write slice of ``fields`` to NetCDF4 format

        Parameters
        ----------
            step : int
                Index of field stored
            sl : list of slices
                The slice to be stored
            fields : list of arrays
                The fields to be stored
        """
        if isinstance(fields, np.ndarray):
            fields = [fields]
        ndims = sl.count(slice(None))
        sl = list(sl)
        sp = []
        for i, j in enumerate(sl):
            if isinstance(j, slice):
                sp.append(i)
        slname = ''
        for ss in sl:
            if isinstance(ss, slice):
                slname += 'slice_'
            else:
                slname += str(ss)+'_'
        slname = slname[:-1]
        s = self.T.local_slice(False)

        # Check if slice is on this processor and make sl local
        sf = []
        for i, j in enumerate(sl):
            if isinstance(j, slice):
                sf.append(s[i])
            else:
                if j >= s[i].start and j < s[i].stop:
                    sl[i] -= s[i].start
        assert len(self.names) == len(fields)
        for name, field in zip(self.names, fields):
            self._write_slice_group(name, slname, ndims, sp, field, sl, sf, step)

    def _write_slice_group(self, name, slname, ndims, sp, u, sl, sf, step):
        sl = tuple(sl)
        sf = tuple(sf)
        group = "_".join((name, "{}D".format(ndims), slname))
        t = "time_" + group
        if group not in self.slice_step:
            self.f.createDimension(t, None)
            ns_t = self.f.createVariable(t, self._dtype, (t))
            ns_t.set_collective(True)
            self.slice_step[group] = ns_t
        sdims = [t] + list(np.take(self.dims, np.array(sp)+1))
        it = self.slice_step[group].size
        self.slice_step[group][it] = step

        if group not in self.handles:
            self.handles[group] = self.f.createVariable(group, self._dtype, sdims)
            self.handles[group].set_collective(True)

        if len(sf) == 3:
            self.handles[group][it, sf[0], sf[1], sf[2]] = u[sl] #pragma: no cover
        elif len(sf) == 2:
            self.handles[group][it, sf[0], sf[1]] = u[sl]
        elif len(sf) == 1:
            self.handles[group][it, sf[0]] = u[sl]

        self.f.sync()

class NCReader(object):
    """Class for reading data from NetCDF4 format

    Parameters
    ----------
        h5name : str
            Name of hdf5 file to read from
        T : PFFT
            Instance of a :class:`PFFT` class. Must be the same as the space
            used for storing with 'write_step' and 'write_slice_step'
        domain : dim-sequence of 2-tuples or arrays of coordinates
            Use dim-sequence of 2-tuples to give the size of the domain as
            origin and length, e.g., (0, 2*pi).
            Use dim-sequence of arrays if using a non-uniform mesh where the
            grid points must be specified. One array per direction.
    """
    def __init__(self, ncname, T):
        self.f = Dataset(ncname, "r", parallel=True, comm=comm)
        self.T = T

    def read(self, u, dset, step):
        """Read into array ``u``

        Parameters
        ----------
        u : numpy array
        dset : str
            Name of array to be read
        step : int
            Index of field to be read
        """
        s = self.T.local_slice(False)
        s = [step] + s
        u[:] = self.f[dset][tuple(s)]

    def close(self):
        self.f.close()

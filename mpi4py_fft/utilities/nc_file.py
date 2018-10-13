import warnings
import six
import numpy as np
from mpi4py import MPI

# https://github.com/Unidata/netcdf4-python/blob/master/examples/mpi_example.py

try:
    from netCDF4 import Dataset
except ImportError: #pragma: no cover
    warnings.warn('netcdf not installed')

__all__ = ('NCWriter', 'NCReader')

comm = MPI.COMM_WORLD

class NCWriter(object):
    """Class for writing data to netcdf format

    Parameters
    ----------
        ncname : str
            Name of netcdf file to be created
        T : PFFT
            Instance of a :class:`.PFFT` class. Must be the same as the space
            used for storing with 'write_step' and 'write_slice_step'
        domain : dim-sequence of 2-tuples or arrays of coordinates
            Use dim-sequence of 2-tuples to give the size of the domain as
            origin and length, e.g., (0, 2*pi).
            Use dim-sequence of arrays if using a non-uniform mesh where the
            grid points must be specified. One array per direction.
        clobber : bool, optional

    Note
    ----
    Each class instance creates one unique NetCDF4-file, with one step-counter.
    It is possible to store multiple fields in each file, but all snapshots of
    the fields must be taken at the same time. If you want one field stored
    every 10th timestep and another every 20th timestep, then use two different
    class instances and as such two NetCDF4-files.
    """
    def __init__(self, ncname, T, domain, clobber=True, **kw):
        self.f = Dataset(ncname, mode="w", clobber=clobber, parallel=True, comm=comm, **kw)
        self.T = T
        self.N = N = T.shape(False)
        dtype = self.T.dtype(False)
        assert dtype.char in 'fdg'
        self._dtype = dtype

        self.f.createDimension('time', None)
        self.dims = ['time']
        self.nc_t = self.f.createVariable('time', self._dtype, ('time'))
        self.nc_t.set_collective(True)

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
        self.f.sync()

    def close(self):
        self.f.close()

    def write(self, step, fields):
        """Write snapshot step of ``fields`` to NetCDF4 file

        Parameters
        ----------
        step : int
            Index of snapshot
        fields : dict
            The fields to be dumped to file. key, values pairs are variable
            name and either arrays or 2-tuples, respectively.

        FIXME: NetCDF4 hangs in parallel for slices if some of the
        processors do not contain the slice.

        """
        it = self.nc_t.size
        self.nc_t[it] = step
        for name, list_of_fields in six.iteritems(fields):
            assert isinstance(list_of_fields, (tuple, list))
            assert isinstance(name, str)
            for field in list_of_fields:
                if isinstance(field, np.ndarray):
                    # Complete array
                    self._write_group(name, field, it)
                else:
                    # A slice of another array
                    assert len(field) == 2
                    u, slices = field
                    self._write_slice_group(name, u, it, slices)

    def _write_group(self, name, u, it):
        s = self.T.local_slice(False)
        if name not in self.handles:
            self.handles[name] = self.f.createVariable(name, self._dtype, self.dims)
            self.handles[name].set_collective(True)

        if self.T.ndim() == 3:
            self.handles[name][it, s[0], s[1], s[2]] = u
        elif self.T.ndim() == 2:
            self.handles[name][it, s[0], s[1]] = u
        else:
            raise NotImplementedError
        self.f.sync()

    def _write_slice_group(self, name, u, it, slices):
        sl = list(slices)
        slname = ''
        for ss in sl:
            if isinstance(ss, slice):
                slname += 'slice_'
            else:
                slname += str(ss)+'_'
        s = self.T.local_slice(False)

        # Check if slice is on this processor and make sl local
        inside = 1
        sf = []
        sp = []
        for i, j in enumerate(sl):
            if isinstance(j, slice):
                sf.append(s[i])
                sp.append(i)
            else:
                if j >= s[i].start and j < s[i].stop:
                    inside *= 1
                    sl[i] -= s[i].start
                else:
                    inside *= 0

        sdims = ['time'] + list(np.take(self.dims, np.array(sp)+1))
        fname = "_".join((name, slname[:-1]))

        if fname not in self.handles:
            self.handles[fname] = self.f.createVariable(fname, self._dtype, sdims)
            self.handles[fname].set_collective(True)
            self.handles[fname].setncattr_string('slices', str(slices))

        sl = tuple(sl)
        if inside:
            if len(sf) == 3: #pragma: no cover
                self.handles[fname][it, sf[0], sf[1], sf[2]] = u[sl]
            elif len(sf) == 2:
                self.handles[fname][it, sf[0], sf[1]] = u[sl]
            elif len(sf) == 1:
                self.handles[fname][it, sf[0]] = u[sl]

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
    """
    def __init__(self, ncname, T):
        self.f = Dataset(ncname, "r", parallel=True, comm=comm)
        self.T = T

    def read(self, u, dset, step):
        """Read into array ``u``

        Parameters
        ----------
        u : array
            The array to read into
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

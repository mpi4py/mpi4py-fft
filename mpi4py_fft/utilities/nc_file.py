import numpy as np
from mpi4py import MPI
from .file_base import FileBase

# https://github.com/Unidata/netcdf4-python/blob/master/examples/mpi_example.py
# Note. Not using groups because Visit does not understand it

__all__ = ('NCFile',)

comm = MPI.COMM_WORLD

class NCFile(FileBase):
    """Class for writing data to NetCDF4 format

    Parameters
    ----------
    ncname : str
        Name of netcdf file to be created
    T : PFFT
        Instance of a :class:`.PFFT` class. Must be the same as the space
        used for storing with :class:`NCWriter.write`
    domain : Sequence
        The spatial domain. Sequence of either

            - 2-tuples, where each 2-tuple contains the (origin, length)
              of each dimension, e.g., (0, 2*pi).
            - Arrays of coordinates, e.g., np.linspace(0, 2*pi, N). One
              array per dimension.
    clobber : bool, optional
    mode : str
        ``r`` or ``w`` for read or write. Default is 'r'.
    Note
    ----
    Each class instance creates one unique NetCDF4-file, with one step-counter.
    It is possible to store multiple fields in each file, but all snapshots of
    the fields must be taken at the same time. If you want one field stored
    every 10th timestep and another every 20th timestep, then use two different
    class instances and as such two NetCDF4-files.
    """
    def __init__(self, ncname, T, domain=None, clobber=True, mode='r', **kw):
        FileBase.__init__(self, T, domain=domain, **kw)
        from netCDF4 import Dataset
        self.filename = ncname
        self.f = f = Dataset(ncname, mode=mode, clobber=clobber, parallel=True, comm=comm, **kw)
        self.N = N = T.shape(False)[-T.dimensions():]
        dtype = self.T.dtype(False)
        assert dtype.char in 'fdg'
        self._dtype = dtype

        if mode == 'w':
            f.createDimension('time', None)
            f.createVariable('time', self._dtype, ('time'))
            d = list(self.domain)
            if not isinstance(self.domain[0], np.ndarray):
                assert len(self.domain[0]) == 2
                for i in range(T.dimensions()):
                    d[i] = np.arange(N[i], dtype=np.float)*2*np.pi/N[i]
            else:
                for i in range(T.dimensions()):
                    d[i] = np.squeeze(d[i])
            self.domain = d
            self.dims = ['time']
            for i in range(T.dimensions()):
                xyz = 'xyzrst'[i]
                f.createDimension(xyz, N[i])
                self.dims.append(xyz)
                nc_xyz = f.createVariable(xyz, self._dtype, (xyz))
                nc_xyz[:] = d[i]
            f.setncatts({"ndim": T.dimensions(), "shape": T.shape(False)})
            f.sync()
        self.close()

    @staticmethod
    def backend():
        return 'netcdf4'

    def open(self):
        from netCDF4 import Dataset
        self.f = Dataset(self.filename, mode='r+', parallel=True, comm=comm)

    def write(self, step, fields, **kw):
        """Write snapshot step of ``fields`` to NetCDF4 file

        Parameters
        ----------
        step : int
            Index of snapshot
        fields : dict
            The fields to be dumped to file. (key, value) pairs are group name
            and either arrays or 2-tuples, respectively. The arrays are complete
            arrays to be stored, whereas 2-tuples are arrays with associated
            *global* slices.

        """
        self.open()
        nc_t = self.f.variables.get('time')
        nc_t.set_collective(True)
        it = nc_t.size
        nc_t[it] = step
        FileBase.write(self, it, fields)
        self.close()

    def read(self, u, name, **kw):
        """Read into array ``u``

        Parameters
        ----------
        u : array
            The array to read into.
        name : str
            Name of array to be read.
        step : int, optional
            Index of field to be read. Default is 0.
        """
        step = kw.get('step', 0)
        self.open()
        s = self.T.local_slice(False)
        s = (step,) + s
        u[:] = self.f[name][s]
        self.close()

    def _write_slice_step(self, name, step, slices, field, **kw):
        slices = list(slices)
        slname = self._get_slice_name(slices)
        s = self.T.local_slice(False)

        slices, inside = self._get_local_slices(slices, s)
        sp = np.nonzero([isinstance(x, slice) for x in slices])[0]
        sf = np.take(s, sp)
        sdims = ['time'] + list(np.take(self.dims, np.array(sp)+1))
        fname = "_".join((name, slname))
        if fname not in self.f.variables:
            h = self.f.createVariable(fname, self._dtype, sdims)
        else:
            h = self.f.variables[fname]
        h.set_collective(True)

        h[step] = 0 # collectively create dataset
        h.set_collective(False)
        sf = tuple([step] + list(sf))
        sl = tuple(slices)
        if inside:
            h[sf] = field[sl]
        h.set_collective(True)
        self.f.sync()

    def _write_group(self, name, u, step, **kw):
        s = self.T.local_slice(False)
        if name not in self.f.variables:
            h = self.f.createVariable(name, self._dtype, self.dims)
        else:
            h = self.f.variables[name]
        h.set_collective(True)
        s = (step,) + s
        h[s] = u
        self.f.sync()

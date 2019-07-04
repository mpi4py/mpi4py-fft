import os
import copy
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
    domain : Sequence, optional
        An optional spatial mesh or domain to go with the data.
        Sequence of either

            - 2-tuples, where each 2-tuple contains the (origin, length)
              of each dimension, e.g., (0, 2*pi).
            - Arrays of coordinates, e.g., np.linspace(0, 2*pi, N). One
              array per dimension.
    mode : str
        ``r``, ``w`` or ``a`` for read, write or append. Default is ``a``.
    clobber : bool, optional
        If True (default), opening a file with mode='w' will clobber an
        existing file with the same name. If False, an exception will be
        raised if a file with the same name already exists.
    kw : dict, optional
        Optional additional keyword arguments used when creating the file
        used to store data.

    Note
    ----
    Each class instance creates one unique NetCDF4-file, with one step-counter.
    It is possible to store multiple fields in each file, but all snapshots of
    the fields must be taken at the same time. If you want one field stored
    every 10th timestep and another every 20th timestep, then use two different
    class instances with two different filenames ``ncname``.
    """
    def __init__(self, ncname, domain=None, mode='a', clobber=True, **kw):
        FileBase.__init__(self, ncname, domain=domain)
        from netCDF4 import Dataset
        # netCDF4 does not seem to handle 'a' if the file does not already exist
        if mode == 'a' and not os.path.exists(ncname):
            mode = 'w'
        self.f = Dataset(ncname, mode=mode, clobber=clobber, parallel=True,
                         comm=comm, **kw)
        self.dims = None
        if 'time' not in self.f.variables:
            self.f.createDimension('time', None)
            self.f.createVariable('time', np.float, ('time'))
        self.close()

    def _check_domain(self, group, field):
        N = field.global_shape[field.rank:]
        if self.domain is None:
            self.domain = []
            for i in range(field.dimensions):
                self.domain.append(np.linspace(0, 2*np.pi, N[i]))

        assert len(self.domain) == field.dimensions
        if len(self.domain[0]) == 2:
            d = self.domain
            self.domain = []
            for i in range(field.dimensions):
                self.domain.append(np.linspace(d[i][0], d[i][1], N[i]))

        self.dims = ['time']
        for i in range(field.rank):
            ind = 'ijk'[i]
            self.dims.append(ind)
            if not ind in self.f.variables:
                self.f.createDimension(ind, field.dimensions)
                n = self.f.createVariable(ind, np.float, (ind))
                n[:] = np.arange(field.dimensions)

        for i in range(field.dimensions):
            xyz = 'xyzrst'[i]
            self.dims.append(xyz)
            if not xyz in self.f.variables:
                self.f.createDimension(xyz, N[i])
                nc_xyz = self.f.createVariable(xyz, np.float, (xyz))
                nc_xyz[:] = self.domain[i]

        self.f.sync()

    @staticmethod
    def backend():
        return 'netcdf4'

    def open(self, mode='r+'):
        from netCDF4 import Dataset
        self.f = Dataset(self.filename, mode=mode, parallel=True, comm=comm)

    def write(self, step, fields, **kw):
        """Write snapshot ``step`` of ``fields`` to NetCDF4 file

        Parameters
        ----------
        step : int
            Index of snapshot.
        fields : dict
            The fields to be dumped to file. (key, value) pairs are group name
            and either arrays or 2-tuples, respectively. The arrays are complete
            arrays to be stored, whereas 2-tuples are arrays with associated
            *global* slices.
        as_scalar : boolean, optional
            Whether to store rank > 0 arrays as scalars. Default is False.

        Example
        -------
        >>> from mpi4py import MPI
        >>> from mpi4py_fft import PFFT, NCFile, newDistArray
        >>> comm = MPI.COMM_WORLD
        >>> T = PFFT(comm, (15, 16, 17))
        >>> u = newDistArray(T, forward_output=False, val=1)
        >>> v = newDistArray(T, forward_output=False, val=2)
        >>> f = NCFile('ncfilename.nc', mode='w')
        >>> f.write(0, {'u': [u, (u, [slice(None), 4, slice(None)])],
        ...             'v': [v, (v, [slice(None), 5, 5])]})
        >>> f.write(1, {'u': [u, (u, [slice(None), 4, slice(None)])],
        ...             'v': [v, (v, [slice(None), 5, 5])]})

        This stores the following datasets to the file ``ncfilename.nc``.
        Using in a terminal 'ncdump -h ncfilename.nc', one gets::

            netcdf ncfilename {
            dimensions:
                    time = UNLIMITED ; // (2 currently)
                    x = 15 ;
                    y = 16 ;
                    z = 17 ;
            variables:
                    double time(time) ;
                    double x(x) ;
                    double y(y) ;
                    double z(z) ;
                    double u(time, x, y, z) ;
                    double u_slice_4_slice(time, x, z) ;
                    double v(time, x, y, z) ;
                    double v_slice_5_5(time, x) ;
            }

        """
        self.open()
        nc_t = self.f.variables.get('time')
        nc_t.set_collective(True)
        it = nc_t.size
        if step in nc_t.__array__(): # If already stored at this step previously
            it = np.argwhere(nc_t.__array__() == step)[0][0]
        else:
            nc_t[it] = step
        FileBase.write(self, it, fields, **kw)
        self.close()

    def read(self, u, name, **kw):
        step = kw.get('step', 0)
        self.open()
        s = u.local_slice()
        s = (step,) + s
        u[:] = self.f[name][s]
        self.close()

    def _write_slice_step(self, name, step, slices, field, **kw):
        assert name not in self.dims # Crashes if user tries to name fields x, y, z, .
        rank = field.rank
        slices = list((slice(None),)*rank + tuple(slices))
        slname = self._get_slice_name(slices[rank:])
        s = field.local_slice()
        slices, inside = self._get_local_slices(slices, s)
        sp = np.nonzero([isinstance(x, slice) for x in slices])[0]
        sf = np.take(s, sp)
        sdims = ['time'] + list(np.take(self.dims, np.array(sp)+1))
        fname = "_".join((name, slname))
        if fname not in self.f.variables:
            h = self.f.createVariable(fname, field.dtype, sdims)
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
        assert name not in self.dims # Crashes if user tries to name fields x, y, z, .
        s = u.local_slice()
        if name not in self.f.variables:
            h = self.f.createVariable(name, u.dtype, self.dims)
        else:
            h = self.f.variables[name]
        h.set_collective(True)
        s = (step,) + s
        h[s] = u
        self.f.sync()

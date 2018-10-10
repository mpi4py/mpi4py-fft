import warnings
import numpy as np
from mpi4py import MPI

try:
    import h5py
except ImportError: #pragma: no cover
    warnings.warn('h5py not installed')

__all__ = ('HDF5Writer', 'HDF5Reader')

comm = MPI.COMM_WORLD

class HDF5Writer(object):
    """Class for writing data to HDF5 format

    Parameters
    ----------
        h5name : str
            Name of hdf5 file to be created
        name : list of strings
            Names of fields to be stored
        T : PFFT
            Instance of a :class:`PFFT` class. Must be the same as the space
            used for storing with 'write_step' and 'write_slice_step'
        domain : dim-sequence of 2-tuples or arrays of coordinates
            Use dim-sequence of 2-tuples to give the size of the domain as
            origin and length, e.g., (0, 2*pi).
            Use dim-sequence of arrays if using a non-uniform mesh where the
            grid points must be specified. One array per direction.
    """
    def __init__(self, h5name, names, T, domain=None):
        self.f = h5py.File(h5name, "w", driver="mpio", comm=comm)
        self.T = T
        self.names = names
        domain = domain if domain is not None else ((0, 2*np.pi),)*3
        if isinstance(domain[0], np.ndarray):
            self.f.create_group("mesh")
        else:
            self.f.create_group("domain")
        for i in range(T.ndim()):
            d = domain[i]
            if isinstance(d, np.ndarray):
                self.f["mesh"].create_dataset("x{}".format(i), data=d)
            else:
                self.f["domain"].create_dataset("x{}".format(i), data=np.array([d[0], d[1]]))
        for name in names:
            self.f.create_group(name)

    def write_step(self, step, fields, forward_output=False):
        """Write ``fields`` to HDF5 format

        Parameters
        ----------
            step : int
                Index of field stored
            fields : list of arrays
                The fields to be stored
            forward_output : bool, optional
                If False, then u is an array from real physical space,
                If True, then u is an array from spectral space.

        Note
        ----
        Fields with name 'name' will be stored under

            - name/{2,3,4}D/step

        """
        if isinstance(fields, np.ndarray):
            fields = [fields]
        for name, field in zip(self.names, fields):
            self._write_group(name, field, step, forward_output)

    def write_slice_step(self, step, sl, fields, forward_output=False):
        """Write slice of ``fields`` to HDF5 format

        Parameters
        ----------
            step : int
                Index of field stored
            sl : list of slices
                The slice to be stored
            fields : list of arrays
                The fields to be stored
            forward_output : bool, optional
                If False, then fields are arrays from real physical space,
                If True, then fields are arrays from spectral space.

        Note
        ----
        Slices of fields with name 'name' will be stored for, e.g.,
        sl = [slice(None), 16, slice(None)], as

            name/2D/slice_16_slice/step

        whereas sl = [8, slice(None), 12] will be stored as

            name/1D/8_slice_12/step

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
        s = self.T.local_slice(forward_output)

        # Check if slice is on this processor and make sl local
        inside = 1
        sf = []
        for i, j in enumerate(sl):
            if isinstance(j, slice):
                sf.append(s[i])
            else:
                if j >= s[i].start and j < s[i].stop:
                    inside *= 1
                    sl[i] -= s[i].start
                else:
                    inside *= 0
        assert len(self.names) == len(fields)
        for name, field in zip(self.names, fields):
            self._write_slice_group(name, slname, ndims, sp, field, sl, sf,
                                    inside, step, forward_output)

    def close(self):
        self.f.close()

    def _write_group(self, name, u, step, forward_output):
        s = tuple(self.T.local_slice(forward_output))
        group = "/".join((name, "{}D".format(len(u.shape))))
        if group not in self.f:
            self.f.create_group(group)
        self.f[group].create_dataset(str(step), shape=self.T.shape(forward_output), dtype=u.dtype)
        if self.T.ndim() == 5:
            self.f["/".join((group, str(step)))][s[0], s[1], s[2], s[3], s[4]] = u #pragma: no cover
        elif self.T.ndim() == 4:
            self.f["/".join((group, str(step)))][s[0], s[1], s[2], s[3]] = u #pragma: no cover
        elif self.T.ndim() == 3:
            self.f["/".join((group, str(step)))][s[0], s[1], s[2]] = u
        elif self.T.ndim() == 2: #pragma: no cover
            self.f["/".join((group, str(step)))][s[0], s[1]] = u
        else:
            raise NotImplementedError

    def _write_slice_group(self, name, slname, ndims, sp, u, sl, sf, inside,
                           step, forward_output):
        sl = tuple(sl)
        sf = tuple(sf)
        group = "/".join((name, "{}D".format(ndims), slname))
        if group not in self.f:
            self.f.create_group(group)
        N = self.T.shape(forward_output)
        self.f[group].create_dataset(str(step), shape=np.take(N, sp), dtype=u.dtype)
        if inside == 1:
            if len(sf) == 3:
                self.f["/".join((group, str(step)))][sf[0], sf[1], sf[2]] = u[sl] #pragma: no cover
            elif len(sf) == 2:
                self.f["/".join((group, str(step)))][sf[0], sf[1]] = u[sl]
            elif len(sf) == 1:
                self.f["/".join((group, str(step)))][sf[0]] = u[sl]

class HDF5Reader(object):
    """Class for reading data from HDF5 format

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
    def __init__(self, h5name, T):
        self.f = h5py.File(h5name, driver="mpio", comm=comm)
        self.T = T

    def read(self, u, dset, forward_output=False):
        """Read into array ``u``

        Parameters
        ----------
        u : numpy array
        dset : str
            Name of array to be read
        forward_output : bool, optional
            The array to be read is the output of a forward transform

        """
        s = self.T.local_slice(forward_output)
        u[:] = self.f[dset][tuple(s)]

    def close(self):
        self.f.close()

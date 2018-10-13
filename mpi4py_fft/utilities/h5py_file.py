import warnings
import six
import numpy as np
from mpi4py import MPI

try:
    import h5py
except ImportError: #pragma: no cover
    warnings.warn('h5py not installed')

__all__ = ('HDF5Writer', 'HDF5Reader')

comm = MPI.COMM_WORLD

class HDF5Writer(h5py.File):
    """Class for writing data to HDF5 format

    Parameters
    ----------
        h5name : str
            Name of hdf5 file to be created
        T : PFFT
            Instance of a :class:`PFFT` class. Must be the same as the space
            used for storing with 'write'
        domain : dim-sequence of 2-tuples or arrays of coordinates
            Use dim-sequence of 2-tuples to give the size of the domain as
            origin and length, e.g., (0, 2*pi).
            Use dim-sequence of arrays if using a non-uniform mesh where the
            grid points must be specified. One array per direction.
    """
    def __init__(self, h5name, T, domain=None):
        h5py.File.__init__(self, h5name, "w", driver="mpio", comm=comm)
        self.T = T
        domain = domain if domain is not None else ((0, 2*np.pi),)*len(T)
        if isinstance(domain[0], np.ndarray):
            self.create_group("mesh")
        else:
            self.create_group("domain")
        for i in range(T.ndim()):
            d = domain[i]
            if isinstance(d, np.ndarray):
                self["mesh"].create_dataset("x{}".format(i), data=d)
            else:
                self["domain"].create_dataset("x{}".format(i), data=np.array([d[0], d[1]]))

    def write(self, step, fields, forward_output=False):
        """Write snapshot ``step`` of ``fields`` to HDF5 file

        Parameters
        ----------
        step : int
            Index of snapshot
        fields : dict
            The fields to be dumped to file. (key, value) pairs are group name
            and either arrays or 2-tuples, respectively.

        Example
        -------
        >>> from mpi4py import MPI
        >>> from mpi4py_fft import PFFT, HDF5Writer, Function
        >>> comm = MPI.COMM_WORLD
        >>> T = PFFT(comm, (15, 16, 17))
        >>> u = Function(T, forward_output=False, val=1)
        >>> v = Function(T, forward_output=False, val=2)
        >>> f = HDF5Writer(comm, T)
        >>> f.write(0, {'u': [u, (u, [slice(None), 4, slice(None)])]
        ...             'v': [v, (v, [slice(None), 5, 5]])})
        >>> f.write(1, {'u': [u, (u, [slice(None), 4, slice(None)])]
        ...             'v': [v, (v, [slice(None), 5, 5]])})

        This stores data within two main groups ``u`` and ``v``. The HDF5 file
        will in the end contain groups::

            /u/3D/{0, 1}
            /u/2D/slice_4_slice/{0, 1}
            /v/3D/{0, 1}
            /v/1D/slice_5_5/{0, 1}

        Note
        ----
        The list of slices used in storing only parts of the arrays are views
        of the *global* arrays.

        """
        for group, list_of_fields in six.iteritems(fields):
            assert isinstance(list_of_fields, (tuple, list))
            assert isinstance(group, str)

            for field in list_of_fields:
                if isinstance(field, np.ndarray):
                    self._write_group(group, field, step, forward_output)
                else:
                    assert len(field) == 2
                    u, sl = field
                    self._write_slice_step(group, step, sl, u, forward_output)

    def _write_slice_step(self, name, step, sl, field, forward_output=False):
        """Write slice of ``field`` to HDF5 format

        Parameters
        ----------
            group : str
                Name of the main group in the HDF5 file
            step : int
                Index of field to be stored
            sl : list of slices
                The slice to be stored
            field : array
                The base field to be stored from
            forward_output : bool, optional
                If False, then fields are arrays from real physical space,
                If True, then fields are arrays from spectral space.
        """
        ndims = sl.count(slice(None))
        sl = list(sl)
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

        sl = tuple(sl)
        group = "/".join((name, "{}D".format(ndims), slname))
        if group not in self:
            self.create_group(group)
        N = self.T.shape(forward_output)
        self[group].create_dataset(str(step), shape=np.take(N, sp), dtype=field.dtype)
        if inside == 1:
            if len(sf) == 3:
                self["/".join((group, str(step)))][sf[0], sf[1], sf[2]] = field[sl] #pragma: no cover
            elif len(sf) == 2:
                self["/".join((group, str(step)))][sf[0], sf[1]] = field[sl]
            elif len(sf) == 1:
                self["/".join((group, str(step)))][sf[0]] = field[sl]

    def _write_group(self, name, u, step, forward_output):
        s = tuple(self.T.local_slice(forward_output))
        group = "/".join((name, "{}D".format(len(u.shape))))
        if group not in self:
            self.create_group(group)
        self[group].create_dataset(str(step), shape=self.T.shape(forward_output), dtype=u.dtype)
        if self.T.ndim() == 5:
            self["/".join((group, str(step)))][s[0], s[1], s[2], s[3], s[4]] = u #pragma: no cover
        elif self.T.ndim() == 4:
            self["/".join((group, str(step)))][s[0], s[1], s[2], s[3]] = u #pragma: no cover
        elif self.T.ndim() == 3:
            self["/".join((group, str(step)))][s[0], s[1], s[2]] = u
        elif self.T.ndim() == 2: #pragma: no cover
            self["/".join((group, str(step)))][s[0], s[1]] = u
        else:
            raise NotImplementedError


class HDF5Reader(h5py.File):
    """Class for reading data from HDF5 format

    Parameters
    ----------
        h5name : str
            Name of hdf5 file to read from
        T : PFFT
            Instance of a :class:`PFFT` class. Must be the same as the space
            used for storing with 'write_step' and 'write_slice_step'
    """
    def __init__(self, h5name, T):
        h5py.File.__init__(self, h5name, driver="mpio", comm=comm)
        self.T = T

    def read(self, u, dset, forward_output=False):
        """Read into array ``u``

        Parameters
        ----------
        u : array
            The array to read into
        dset : str
            Name of array to be read
        forward_output : bool, optional
            The array to be read is the output of a forward transform

        """
        s = self.T.local_slice(forward_output)
        u[:] = self[dset][tuple(s)]

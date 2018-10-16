import warnings
import six
import numpy as np
from mpi4py import MPI
from .file_base import FileBase

try:
    import h5py
except ImportError: #pragma: no cover
    warnings.warn('h5py not installed')

__all__ = ('HDF5File',)

comm = MPI.COMM_WORLD

class HDF5File(FileBase):
    """Class for reading/writing data to HDF5 format

    Parameters
    ----------
        h5name : str
            Name of hdf5 file to be created
        T : PFFT
            Instance of a :class:`PFFT` class. Must be the same as the space
            used for storing with :class:`HDF5Writer.write`
        domain : Sequence
            The spatial domain. Sequence of either

                - 2-tuples, where each 2-tuple contains the (origin, length)
                  of each dimension, e.g., (0, 2*pi).
                - Arrays of coordinates, e.g., np.linspace(0, 2*pi, N). One
                  array per dimension.
        mode : str
            'r' or 'w' for read or write
        kw : dict
            Additional keywords
    """
    def __init__(self, h5name, T, domain=None, mode='w', **kw):
        FileBase.__init__(self, T, domain=domain, **kw)
        self.f = h5py.File(h5name, mode, driver="mpio", comm=comm)
        if mode == 'w':
            if isinstance(self.domain[0], np.ndarray):
                self.f.create_group("mesh")
            else:
                self.f.create_group("domain")
            for i in range(T.ndim()):
                d = self.domain[i]
                if isinstance(d, np.ndarray):
                    self.f["mesh"].create_dataset("x{}".format(i), data=d)
                else:
                    self.f["domain"].create_dataset("x{}".format(i), data=np.array([d[0], d[1]]))

    def write(self, step, fields, forward_output=False):
        """Write snapshot ``step`` of ``fields`` to HDF5 file

        Parameters
        ----------
        step : int
            Index of snapshot
        fields : dict
            The fields to be dumped to file. (key, value) pairs are group name
            and either arrays or 2-tuples, respectively. The arrays are complete
            arrays to be stored, whereas 2-tuples are arrays with associated
            *global* slices.
        forward_output : bool, optional
            Whether fields to be stored are shaped as the output of a forward
            transform or not
        kw : dict
            Additional keywords

        Example
        -------
        >>> from mpi4py import MPI
        >>> from mpi4py_fft import PFFT, HDF5Writer, Function
        >>> comm = MPI.COMM_WORLD
        >>> T = PFFT(comm, (15, 16, 17))
        >>> u = Function(T, forward_output=False, val=1)
        >>> v = Function(T, forward_output=False, val=2)
        >>> f = HDF5Writer('h5filename.h5', T)
        >>> f.write(0, {'u': [u, (u, [slice(None), 4, slice(None)])],
        ...             'v': [v, (v, [slice(None), 5, 5])]})
        >>> f.write(1, {'u': [u, (u, [slice(None), 4, slice(None)])],
        ...             'v': [v, (v, [slice(None), 5, 5])]})

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
        FileBase.write(self, step, fields, forward_output=forward_output)

    def read(self, u, dset, forward_output=False, **kw):
        """Read into array ``u``

        Parameters
        ----------
        u : array
            The array to read into
        dset : str
            Name of array to be read
        forward_output : bool, optional
            Whether the array to be read is the output of a forward transform
            or not
        """
        s = self.T.local_slice(forward_output)
        u[:] = self.f[dset][tuple(s)]

    def _write_slice_step(self, name, step, slices, field, forward_output=False):
        """Write slice of ``field`` to HDF5 format

        Parameters
        ----------
            group : str
                Name of the main group in the HDF5 file
            step : int
                Index of field to be stored
            slices : list of slices
                The slice to be stored
            field : array
                The base field to be stored from
            forward_output : bool, optional
                If False, then fields are arrays from real physical space,
                If True, then fields are arrays from spectral space.
        """
        slices = list(slices)
        ndims = slices.count(slice(None))
        slname = self._get_slice_name(slices)
        s = self.T.local_slice(forward_output)

        slices, inside = self._get_local_slices(slices, s)
        sp = np.nonzero([isinstance(x, slice) for x in slices])[0]
        sf = np.take(s, sp)
        sl = tuple(slices)
        group = "/".join((name, "{}D".format(ndims), slname))
        if group not in self.f:
            self.f.create_group(group)
        N = self.T.shape(forward_output)
        self.f[group].create_dataset(str(step), shape=np.take(N, sp), dtype=field.dtype)
        if inside == 1:
            if len(sf) == 3:
                self.f["/".join((group, str(step)))][sf[0], sf[1], sf[2]] = field[sl] #pragma: no cover
            elif len(sf) == 2:
                self.f["/".join((group, str(step)))][sf[0], sf[1]] = field[sl]
            elif len(sf) == 1:
                self.f["/".join((group, str(step)))][sf[0]] = field[sl]

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

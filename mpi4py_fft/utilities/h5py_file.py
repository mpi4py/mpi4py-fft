import numpy as np
from mpi4py import MPI
from .file_base import FileBase

__all__ = ('HDF5File',)

comm = MPI.COMM_WORLD

class HDF5File(FileBase):
    """Class for reading/writing data to HDF5 format

    Parameters
    ----------
    h5name : str
        Name of hdf5 file to be created.
    T : PFFT
        Instance of a :class:`PFFT` class. Must be the same as the space
        used for storing with :class:`HDF5Writer.write`.
    domain : Sequence, optional
        The spatial domain. Sequence of either

            - 2-tuples, where each 2-tuple contains the (origin, length)
              of each dimension, e.g., (0, 2*pi).
            - Arrays of coordinates, e.g., np.linspace(0, 2*pi, N). One
              array per dimension.
    mode : str, optional
        ``r`` or ``w`` for read or write. Default is ``r``.
    """
    def __init__(self, h5name, T, domain=None, mode='r', **kw):
        FileBase.__init__(self, T, domain=domain, **kw)
        import h5py
        self.filename = h5name
        self.f = f = h5py.File(h5name, mode, driver="mpio", comm=comm)
        if mode == 'w':
            if isinstance(self.domain[0], np.ndarray):
                f.create_group("mesh")
            else:
                f.create_group("domain")
            for i in range(T.dimensions()):
                d = self.domain[i]
                if isinstance(d, np.ndarray):
                    f["mesh"].create_dataset("x{}".format(i), data=np.squeeze(d))
                else:
                    f["domain"].create_dataset("x{}".format(i), data=np.array([d[0], d[1]]))
            f.attrs.create("ndim", T.dimensions())
            f.attrs.create("shape", T.shape(False))
        self.close()

    @staticmethod
    def backend():
        return 'hdf5'

    def open(self):
        import h5py
        self.f = h5py.File(self.filename, 'r+', driver="mpio", comm=comm)

    def write(self, step, fields, **kw):
        """Write snapshot ``step`` of ``fields`` to HDF5 file

        Parameters
        ----------
        step : int
            Index of snapshot.
        fields : dict
            The fields to be dumped to file. (key, value) pairs are group name
            and either arrays or 2-tuples, respectively. The arrays are complete
            arrays to be stored, whereas 2-tuples are arrays with associated
            *global* slices.
        forward_output : bool, optional
            Whether fields to be stored are shaped as the output of a
            forward transform or not. Default is False.

        Example
        -------
        >>> from mpi4py import MPI
        >>> from mpi4py_fft import PFFT, HDF5File, Function
        >>> comm = MPI.COMM_WORLD
        >>> T = PFFT(comm, (15, 16, 17))
        >>> u = Function(T, forward_output=False, val=1)
        >>> v = Function(T, forward_output=False, val=2)
        >>> f = HDF5File('h5filename.h5', T)
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
        self.open()
        FileBase.write(self, step, fields, **kw)
        self.close()

    def read(self, u, name, **kw):
        """Read into array ``u``

        Parameters
        ----------
        u : array
            The array to read into.
        name : str
            Name of array to be read.
        forward_output : bool, optional
            Whether the array to be read is the output of a forward transform
            or not. Default is False.
        step : int, optional
            Index of field to be read. Default is 0.
        """
        forward_output = kw.get('forward_output', False)
        step = kw.get('step', 0)
        self.open()
        s = self.T.local_slice(forward_output)
        dset = "/".join((name, "{}D".format(u.ndim), str(step)))
        u[:] = self.f[dset][s]
        self.close()

    def _write_slice_step(self, name, step, slices, field, **kw):
        forward_output = kw.get('forward_output', False)
        slices = list(slices)
        ndims = slices.count(slice(None))
        slname = self._get_slice_name(slices)
        s = self.T.local_slice(forward_output)
        slices, inside = self._get_local_slices(slices, s)
        sp = np.nonzero([isinstance(x, slice) for x in slices])[0]
        sf = tuple(np.take(s, sp))
        sl = tuple(slices)
        group = "/".join((name, "{}D".format(ndims), slname))
        if group not in self.f:
            self.f.create_group(group)
        N = self.T.shape(forward_output)
        self.f[group].require_dataset(str(step), shape=tuple(np.take(N, sp)), dtype=field.dtype)
        if inside == 1:
            self.f["/".join((group, str(step)))][sf] = field[sl]

    def _write_group(self, name, u, step, **kw):
        forward_output = kw.get('forward_output', False)
        s = self.T.local_slice(forward_output)
        group = "/".join((name, "{}D".format(self.T.dimensions())))
        if group not in self.f:
            self.f.create_group(group)
        self.f[group].require_dataset(str(step), shape=self.T.shape(forward_output), dtype=u.dtype)
        self.f["/".join((group, str(step)))][s] = u

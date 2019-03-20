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
    domain : sequence, optional
        An optional spatial mesh or domain to go with the data.
        Sequence of either

            - 2-tuples, where each 2-tuple contains the (origin, length)
              of each dimension, e.g., (0, 2*pi).
            - Arrays of coordinates, e.g., np.linspace(0, 2*pi, N). One
              array per dimension.
    mode : str, optional
        ``r``, ``w`` or ``a`` for read, write or append. Default is ``a``.
    kw : dict, optional
        Optional additional keyword arguments used when creating the file
        used to store data.
    """
    def __init__(self, h5name, domain=None, mode='a', **kw):
        FileBase.__init__(self, h5name, domain=domain)
        import h5py
        self.f = h5py.File(h5name, mode, driver="mpio", comm=comm, **kw)
        self.close()

    def _check_domain(self, group, field):
        if self.domain is None:
            self.domain = ((0, 2*np.pi),)*field.dimensions
        assert len(self.domain) == field.dimensions
        self.f.require_group(group)
        if not "shape" in self.f[group].attrs:
            self.f[group].attrs.create("shape", field._p0.shape)
        if not "rank" in self.f[group].attrs:
            self.f[group].attrs.create("rank", field.rank)
        assert field.rank == self.f[group].attrs["rank"]
        assert np.all(field._p0.shape == self.f[group].attrs["shape"])
        if isinstance(self.domain[0], np.ndarray):
            self.f[group].require_group("mesh")
        else:
            self.f[group].require_group("domain")
        for i in range(field.dimensions):
            d = self.domain[i]
            if isinstance(d, np.ndarray):
                d0 = np.squeeze(d)
                self.f[group]["mesh"].require_dataset("x{}".format(i),
                                                      shape=d0.shape,
                                                      dtype=d0.dtype,
                                                      data=d0)
            else:
                d0 = np.array([d[0], d[1]])
                self.f[group]["domain"].require_dataset("x{}".format(i),
                                                        shape=d0.shape,
                                                        dtype=d0.dtype,
                                                        data=d0)

    @staticmethod
    def backend():
        return 'hdf5'

    def open(self, mode='r+'):
        import h5py
        self.f = h5py.File(self.filename, mode, driver="mpio", comm=comm)

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
        as_scalar : boolean, optional
            Whether to store rank > 0 arrays as scalars. Default is False.

        Example
        -------
        >>> from mpi4py import MPI
        >>> from mpi4py_fft import PFFT, HDF5File, newDistArray
        >>> comm = MPI.COMM_WORLD
        >>> T = PFFT(comm, (15, 16, 17))
        >>> u = newDistArray(T, forward_output=False, val=1)
        >>> v = newDistArray(T, forward_output=False, val=2)
        >>> f = HDF5File('h5filename.h5', mode='w')
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
        step = kw.get('step', 0)
        self.open()
        s = u.local_slice()
        dset = "/".join((name, "{}D".format(u.dimensions), str(step)))
        u[:] = self.f[dset][s]
        self.close()

    def _write_slice_step(self, name, step, slices, field, **kw):
        rank = field.rank
        slices = (slice(None),)*rank + tuple(slices)
        slices = list(slices)
        ndims = slices[rank:].count(slice(None))
        slname = self._get_slice_name(slices[rank:])
        s = field.local_slice()
        slices, inside = self._get_local_slices(slices, s)
        sp = np.nonzero([isinstance(x, slice) for x in slices])[0]
        sf = tuple(np.take(s, sp))
        sl = tuple(slices)
        group = "/".join((name, "{}D".format(ndims), slname))
        self.f.require_group(group)
        N = field.global_shape
        self.f[group].require_dataset(str(step), shape=tuple(np.take(N, sp)), dtype=field.dtype)
        if inside == 1:
            self.f["/".join((group, str(step)))][sf] = field[sl]

    def _write_group(self, name, u, step, **kw):
        s = u.local_slice()
        group = "/".join((name, "{}D".format(u.dimensions)))
        self.f.require_group(group)
        self.f[group].require_dataset(str(step), shape=u.global_shape, dtype=u.dtype)
        self.f["/".join((group, str(step)))][s] = u

from mpi4py import MPI
import numpy as np

__all__ = ('FileBase',)

comm = MPI.COMM_WORLD

class FileBase(object):
    """Base class for reading/writing distributed arrays

    Parameters
    ----------
    filename : str, optional
        Name of backend file used to store data
    domain : sequence, optional
        An optional spatial mesh or domain to go with the data.
        Sequence of either

            - 2-tuples, where each 2-tuple contains the (origin, length)
              of each dimension, e.g., (0, 2*pi).
            - Arrays of coordinates, e.g., np.linspace(0, 2*pi, N). One
              array per dimension.

    """
    def __init__(self, filename=None, domain=None):
        self.f = None
        self.filename = filename
        self.domain = domain

    def _check_domain(self, group, field):
        """Check dimensions and store (if missing) self.domain"""
        raise NotImplementedError

    def write(self, step, fields, **kw):
        """Write snapshot ``step`` of ``fields`` to file

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
        """
        as_scalar = kw.get("as_scalar", False)

        def _write(group, u, sl, step, kw, k=None):
            if sl is None:
                self._write_group(group, u, step, **kw)
            else:
                self._write_slice_step(group, step, sl, u, **kw)

        for group, list_of_fields in fields.items():
            assert isinstance(list_of_fields, (tuple, list))
            assert isinstance(group, str)

            for field in list_of_fields:
                u = field[0] if isinstance(field, (tuple, list)) else field
                sl = field[1] if isinstance(field, (tuple, list)) else None
                if as_scalar is False or u.rank == 0:
                    self._check_domain(group, u)
                    _write(group, u, sl, step, kw)
                else: # as_scalar is True and u.rank > 0
                    if u.rank == 1:
                        for k in range(u.shape[0]):
                            g = group + str(k)
                            self._check_domain(g, u[k])
                            _write(g, u[k], sl, step, kw)
                    elif u.rank == 2:
                        for k in range(u.shape[0]):
                            for l in range(u.shape[1]):
                                g = group + str(k) + str(l)
                                self._check_domain(g, u[k, l])
                                _write(g, u[k, l], sl, step, kw)

    def read(self, u, name, **kw):
        """Read field ``name`` into distributed array ``u``

        Parameters
        ----------
        u : array
            The :class:`.DistArray` to read into.
        name : str
            Name of field to be read.
        step : int, optional
            Index of field to be read. Default is 0.
        """
        raise NotImplementedError

    def close(self):
        """Close the self.filename file"""
        self.f.close()

    def open(self, mode='r+'):
        """Open the self.filename file for reading or writing

        Parameters
        ----------
        mode : str
           Open file in this mode. Default is 'r+'.
        """
        raise NotImplementedError

    @staticmethod
    def backend():
        """Return which backend is used to store data"""
        raise NotImplementedError

    def _write_slice_step(self, name, step, slices, field, **kwargs):
        raise NotImplementedError

    def _write_group(self, name, u, step, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_slice_name(slices):
        sl = list(slices)
        slname = ''
        for ss in sl:
            if isinstance(ss, slice):
                slname += 'slice_'
            else:
                slname += str(ss)+'_'
        return slname[:-1]

    @staticmethod
    def _get_local_slices(slices, s):
        # Check if data is on this processor and make slices local
        inside = 1
        si = np.nonzero([isinstance(x, int) and not z == slice(None) for x, z in zip(slices, s)])[0]
        for i in si:
            if slices[i] >= s[i].start and slices[i] < s[i].stop:
                slices[i] -= s[i].start
            else:
                inside = 0
        return slices, inside

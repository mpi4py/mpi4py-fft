from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

class FileBase(object):
    """Base class for reading/writing structured arrays

    Parameters
    ----------
    T : PFFT
        Instance of a :class:`.PFFT` class.
    domain : sequence, optional
        The spatial domain. Sequence of either

            - 2-tuples, where each 2-tuple contains the (origin, length)
              of each dimension, e.g., (0, 2*pi).
            - Arrays of coordinates, e.g., np.linspace(0, 2*pi, N). One
              array per dimension.
    """
    def __init__(self, T, domain=None, **kw):
        self.f = None
        self.filename = None
        self.T = T
        self.domain = domain if domain is not None else ((0, 2*np.pi),)*T.dimensions()

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
        """
        for group, list_of_fields in fields.items():
            assert isinstance(list_of_fields, (tuple, list))
            assert isinstance(group, str)

            for field in list_of_fields:
                if isinstance(field, np.ndarray):
                    self._write_group(group, field, step, **kw)
                else:
                    assert len(field) == 2
                    u, sl = field
                    self._write_slice_step(group, step, sl, u, **kw)

    def read(self, u, name, **kw):
        """Read into array ``u``

        Parameters
        ----------
        u : array
            The array to read into.
        name : str
            Name of array to be read.
        """
        raise NotImplementedError

    def close(self):
        self.f.close()

    def open(self):
        raise NotImplementedError

    @staticmethod
    def backend():
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

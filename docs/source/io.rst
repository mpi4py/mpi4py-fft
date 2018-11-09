Storing datafiles
=================

mpi4py-fft works with regular Numpy arrays. However, since arrays in parallel
can become very large, and the arrays live on multiple processors, we require
parallel IO capabilities that goes beyond Numpys regular methods.
In the :mod:`.utilities` module there are two helper classes for dumping
dataarrays to either `HDF5 <https://www.hdf5.org>`_ or
`NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ format:

    * :class:`.HDF5File`
    * :class:`.NCFile`

Both classes have one ``write`` and one ``read`` method that stores or
reads data in parallel. A simple example of usage is::

    from mpi4py import MPI
    import numpy as np
    from mpi4py_fft import PFFT, HDF5File, NCFile, Function

    N = (128, 256, 512)
    T = PFFT(MPI.COMM_WORLD, N)
    u = Function(T, forward_output=False)
    v = Function(T, forward_output=False, val=2)
    u[:] = np.random.random(N)

    fields = {'u': [u], 'v': [v]}
    f0 = HDF5File('h5test.h5', T)
    f1 = NCFile('nctest.nc', T)
    f0.write(0, fields)
    f1.write(0, fields)
    v[:] = 3
    f0.write(1, fields)
    f1.write(1, fields)

Note that we are creating two datafiles ``h5test.h5`` and ``nctest.nc``,
for storing in HDF5 or NetCDF4 formats respectively. Normally, one would be
satisfied using only one format, so this is only for illustration. We store
the fields ``u`` and ``v`` using method ``write`` on two different occasions,
so the datafiles will contain two snapshots of each field ``u`` and ``v``.

The stored dataarrays can be retrieved later on::

    f0 = HDF5File('h5test.h5', T, mode='r')
    f1 = NCFile('nctest.nc', T, mode='r')
    u0 = Function(T, forward_output=False)
    u1 = Function(T, forward_output=False)
    f0.read(u0, 'u', 0)
    f0.read(u1, 'u', 1)
    f1.read(u0, 'u', 0)
    f1.read(u1, 'u', 1)

Note that one does not have to use the same number of processors when
retrieving the data as when they were stored.

It is also possible to store only parts of the, potentially large, arrays.
Any chosen slice may be stored, using a *global* view of the arrays::

    f2 = HDF5File('variousfields.h5', T, mode='w')
    fields = {'u': [u,
                    (u, [slice(None), slice(None), 4]),
                    (u, [5, 5, slice(None)])],
              'v': [v,
                    (v, [slice(None), 6, slice(None)])]}
    f2.write(0, fields)
    f2.write(1, fields)
    f2.write(2, fields)

This will lead to an hdf5-file with groups::

    variousfields.h5/
    ├─ u/
    |  ├─ 1D/
    |  |  └─ 5_5_slice/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 3
    |  ├─ 2D/
    |  |  └─ slice_slice_4/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 2
    |  └─ 3D/
    |     ├─ 0
    |     ├─ 1
    |     └─ 2
    ├─ v/
    |  ├─ 2D/
    |  |  └─ slice_6_slice/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 2
    |  └─ 3D/
    |     ├─ 0
    |     ├─ 1
    |     └─ 2
    └─ mesh/
       ├─ x0
       ├─ x1
       └─ x2

Note that a mesh is stored along with all the data. This mesh can be given in
two different ways when creating the datafiles:

    1) A sequence of 2-tuples, where each 2-tuple contains the (origin, length)
       of the domain along its dimension. For example, a uniform mesh that
       originates from the origin, with lengths :math:`\pi, 2\pi, 3\pi`, can be
       given as::

        f0 = HDF5File('filename.h5', T, domain=((0, pi), (0, 2*np.pi), (0, 3*np.pi)))

    2) One array giving the coordinates for each dimension. For example::

        d = (np.arange(N[0], dtype=np.float)*1*np.pi/N[0],
             np.arange(N[1], dtype=np.float)*2*np.pi/N[1],
             np.arange(N[2], dtype=np.float)*2*np.pi/N[2])
        f0 = HDF5File('filename.h5', T, domain=d)

With NetCDF4 the layout is somewhat different. For ``variousfields`` above,
if we were using :class:`.NCFile` instead of :class:`.HDF5File`,
we would get a datafile that with ``ncdump -h variousfields.nc`` would look like::

    netcdf variousfields {
    dimensions:
            time = UNLIMITED ; // (3 currently)
            x = 128 ;
            y = 256 ;
            z = 512 ;
    variables:
            double time(time) ;
            double x(x) ;
            double y(y) ;
            double z(z) ;
            double u(time, x, y, z) ;
            double u_slice_slice_4(time, x, y) ;
            double u_5_5_slice(time, z) ;
            double v(time, x, y, z) ;
            double v_slice_6_slice(time, x, z) ;
    }

Postprocessing
--------------

Dataarrays stored to HDF5 files can be visualized using both `Paraview <https://www.paraview.org>`_
and `Visit <https://www.visitusers.org>`_, whereas NetCDF4 files can at the time of writing only be
opened with `Visit <https://www.visitusers.org>`_.

To view the HDF5-files we first need to generate some light-weight *xdmf*-files that can
be understood by both Paraview or Visit. To generate such files, simply throw the
module :mod:`.utilities.generate_xdmf` on the HDF5-files::

    from mpi4py_fft.utilities import generate_xdmf
    generate_xdmf('variousfields.h5')

This will create a number of xdmf-files, one for each group that contains 2D
or 3D data::

    variousfields.xdmf
    variousfields_slice_slice_4.xdmf
    variousfields_slice_6_slice.xdmf

These files can be opened directly in Paraview. However, note that for Visit, one has to
generate the files using::

    generate_xdmf('variousfields.h5', order='visit')

because for some reason Paraview and Visit require the mesh in the xdmf-files
to be stored in opposite order.
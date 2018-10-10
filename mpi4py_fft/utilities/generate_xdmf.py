# pylint: disable=line-too-long
import copy
import six
from numpy import dtype, array, invert
try:
    import h5py
except ImportError:
    import warnings
    warnings.warn('h5py not installed')


__all__ = ('generate_xdmf',)

xdmffile = """<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
  <Domain>
    <Grid Name="Structured Grid" GridType="Collection" CollectionType="Temporal">
      <Time TimeType="List"><DataItem Format="XML" Dimensions="{1}"> {0} </DataItem></Time>
       {2}
    </Grid>
  </Domain>
</Xdmf>
"""

def get_grid(geometry, topology, attrs):
    return """<Grid GridType="Uniform">
          {0}
          {1}
          {2}
        </Grid>
        """.format(geometry, topology, attrs)

def get_geometry(kind=0, dim=2):
    if dim == 2:
        if kind == 0:
            return """<Geometry Type="ORIGIN_DXDY">
          <DataItem Format="XML" NumberType="Float" Dimensions="2">
            {0} {1}
          </DataItem>
          <DataItem Format="XML" NumberType="Float" Dimensions="2">
            {2} {3}
          </DataItem>
          </Geometry>"""
        if kind == 1:
            return """<Geometry Type="VXVY">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
            {3}:/mesh/{4}
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
            {3}:/mesh/{5}
          </DataItem>
          </Geometry>"""

    if dim == 3:
        if kind == 0:
            return """<Geometry Type="ORIGIN_DXDYDZ">
          <DataItem Format="XML" NumberType="Float" Dimensions="3">
            {0} {1} {2}
          </DataItem>
          <DataItem Format="XML" NumberType="Float" Dimensions="3">
            {3} {4} {5}
          </DataItem>
          </Geometry>"""
        if kind == 1:
            return """<Geometry Type="VXVYVZ">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{3}">
            {4}:/mesh/{5}
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
            {4}:/mesh/{6}
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
            {4}:/mesh/{7}
          </DataItem>
          </Geometry>"""
    return ""

def get_topology(dims, kind=0):
    assert len(dims) in (2, 3)
    co = 'Co' if kind == 0 else ''
    if len(dims) == 2:
        return """<Topology Dimensions="{0} {1}" Type="2D{2}RectMesh"/>""".format(*dims, co)
    if len(dims) == 3:
        return """<Topology Dimensions="{0} {1} {2}" Type="3D{3}RectMesh"/>""".format(*dims, co)

def get_attribute(attr, h5filename, dims, prec):
    name = attr.split("/")[0]
    if len(dims) == 2:
        return """<Attribute Name="{0}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{5}" Dimensions="{1} {2}">
            {3}:/{4}
          </DataItem>
          </Attribute>""".format(name, dims[0], dims[1], h5filename, attr, prec)

    if len(dims) == 3:
        return """<Attribute Name="{0}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{6}" Dimensions="{1} {2} {3}">
            {4}:/{5}
          </DataItem>
          </Attribute>""".format(name, dims[0], dims[1], dims[2], h5filename, attr, prec)
    return ""

def generate_xdmf(h5filename, periodic=True):
    """Generate XDMF-files

    Parameters
    ----------
        h5filename : str
            Name of hdf5-file that we want to decorate with xdmf
        periodic : bool or dim-sequence of bools, optional
            If true along axis i, assume data is periodic.
            Only affects the calculation of the domain size

    """
    f = h5py.File(h5filename)
    keys = []
    f.visit(keys.append)
    assert 'mesh' in keys or 'domain' in keys

    # Find unique groups
    datasets = {2:{}, 3:{}}  # 2D and 3D datasets
    for key in keys:
        if isinstance(f[key], h5py.Dataset):
            if not ('mesh' in key or 'domain' in key):
                tstep = int(key.split("/")[-1])
                ndim = len(f[key].shape)
                if ndim in (2, 3):
                    ds = datasets[ndim]
                    if tstep in ds:
                        ds[tstep] += [key]
                    else:
                        ds[tstep] = [key]

    if periodic is True:
        periodic = (0,)*3
    elif periodic is False:
        periodic = (1,)*3
    else:
        assert isinstance(periodic, (tuple, list))
        periodic = tuple(array(invert(periodic), int))

    coor = {0:'x0', 1:'x1', 2:'x2'}
    for ndim, dsets in six.iteritems(datasets):
        timesteps = list(dsets.keys())
        if not timesteps:
            continue

        timesteps.sort(key=int)
        tt = ""
        for i in timesteps:
            tt += "%s " %i

        datatype = f[dsets[timesteps[0]][0]].dtype
        assert datatype.char not in 'FDG', "Cannot use generate_xdmf to visualize complex data."
        prec = 4 if datatype is dtype('float32') else 8
        if ndim == 2:
            xff = {}
            geometry = {}
            topology = {}
            grid = {}
            NN = {}
            for name in dsets[timesteps[0]]:
                slices = name.split("/")[2]
                if not slices in xff:
                    xff[slices] = copy.copy(xdmffile)
                    NN[slices] = N = f[name].shape
                    if 'slice' in slices:
                        ss = slices.split("_")
                        cc = [coor[i] for i, sx in enumerate(ss) if 'slice' in sx]
                    else:
                        cc = ['x0', 'x1']
                    if 'domain' in keys:
                        geo = get_geometry(kind=0, dim=2)
                        geometry[slices] = geo.format(f['domain/{}'.format(cc[0])][0],
                                                      f['domain/{}'.format(cc[1])][0],
                                                      f['domain/{}'.format(cc[0])][1]/(N[0]-periodic[0]),
                                                      f['domain/{}'.format(cc[1])][1]/(N[1]-periodic[1]))
                        topology[slices] = get_topology(N, kind=0)
                    elif 'mesh' in keys:
                        geo = get_geometry(kind=1, dim=2)
                        geometry[slices] = geo.format(prec, N[0], N[1], h5filename, cc[0], cc[1])
                        topology[slices] = get_topology(N, kind=1)
                    grid[slices] = ''

            # if slice of 3D data, need to know xy, xz or yz plane
            # Since there may be many different slices, we need to create
            # one xdmf-file for each composition of slices
            for tstep in timesteps:
                d = dsets[tstep]
                attrs = ''
                for i, x in enumerate(d):
                    slices = x.split("/")[2]
                    if not 'slice' in slices:
                        slices = dsets[timesteps[0]][0].split("/")[2]
                    N = NN[slices]
                    attrs += get_attribute(x, h5filename, N, prec)
                grid[slices] += get_grid(geometry[slices], topology[slices], attrs)

            for slices, ff in six.iteritems(xff):
                fname = h5filename[:-3]+"_"+slices+".xdmf" if 'slice' in slices else h5filename[:-3]+".xdmf"
                xfl = open(fname, "w")
                h = ff.format(tt, len(timesteps), grid[slices].rstrip())
                xfl.write(h)
                xfl.close()

        elif ndim == 3:
            grid = ''
            for tstep in timesteps:
                d = dsets[tstep]
                N = f[d[0]].shape
                if 'domain' in keys:
                    geo = get_geometry(kind=0, dim=2)
                    geometry = geo.format(f['domain/x0'][0],
                                          f['domain/x1'][0],
                                          f['domain/x2'][0],
                                          f['domain/x0'][1]/(N[0]-periodic[0]),
                                          f['domain/x1'][1]/(N[1]-periodic[1]),
                                          f['domain/x2'][1]/(N[2]-periodic[2]))
                elif 'mesh' in keys:
                    geo = get_geometry(kind=1, dim=3)
                    geometry[slices] = geo.format(prec, N[0], N[1], N[2], h5filename, 'x2', 'x1', 'x0')
                topology = get_topology(N)
                attrs = ""
                for x in d:
                    attrs += get_attribute(x, h5filename, N, prec)
                grid += get_grid(geometry, topology, attrs)
            xfl = open(h5filename[:-3]+".xdmf", "w")
            ff = copy.copy(xdmffile)
            ff = ff.format(tt, len(timesteps), grid.rstrip())
            xfl.write(ff)
            xfl.close()

if __name__ == "__main__":
    import sys
    generate_xdmf(sys.argv[-1])

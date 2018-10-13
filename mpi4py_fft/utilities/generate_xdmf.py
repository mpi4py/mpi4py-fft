# pylint: disable=line-too-long
import copy
import six
import pprint
from numpy import dtype, array, invert, take
try:
    import h5py
except ImportError: #pragma: no cover
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
    assert kind in (0, 1)
    assert dim in (2, 3)
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

def get_topology(dims, kind=0):
    assert len(dims) in (2, 3)
    co = 'Co' if kind == 0 else ''
    if len(dims) == 2:
        return """<Topology Dimensions="{0} {1}" Type="2D{2}RectMesh"/>""".format(dims[0], dims[1], co)
    if len(dims) == 3:
        return """<Topology Dimensions="{0} {1} {2}" Type="3D{3}RectMesh"/>""".format(dims[0], dims[1], dims[2], co)

def get_attribute(attr, h5filename, dims, prec):
    name = attr.split("/")[0]
    assert len(dims) in (2, 3)
    if len(dims) == 2:
        return """<Attribute Name="{0}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{5}" Dimensions="{1} {2}">
            {3}:/{4}
          </DataItem>
          </Attribute>""".format(name, dims[0], dims[1], h5filename, attr, prec)

    return """<Attribute Name="{0}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{6}" Dimensions="{1} {2} {3}">
            {4}:/{5}
          </DataItem>
          </Attribute>""".format(name, dims[0], dims[1], dims[2], h5filename, attr, prec)

def generate_xdmf(h5filename, periodic=True):
    """Generate XDMF-files

    Parameters
    ----------
        h5filename : str
            Name of hdf5-file that we want to decorate with xdmf
        periodic : bool or dim-sequence of bools, optional
            If true along axis i, assume data is periodic.
            Only affects the calculation of the domain size and only if the
            domain is given as 2-tuple of origin+dx.
    """
    f = h5py.File(h5filename)
    keys = []
    f.visit(keys.append)
    assert 'mesh' in keys or 'domain' in keys

    # Find unique groups of 2D and 3D datasets
    datasets = {2:{}, 3:{}}
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

    coor = ['x0', 'x1', 'x2', 'x3', 'x4']
    for ndim, dsets in six.iteritems(datasets):
        timesteps = list(dsets.keys())
        if not timesteps:
            continue
        if periodic is True:
            periodic = (0,)*5
        elif periodic is False:
            periodic = (1,)*5
        else:
            assert isinstance(periodic, (tuple, list))
            periodic = tuple(array(invert(periodic), int))

        timesteps.sort(key=int)
        tt = ""
        for i in timesteps:
            tt += "%s " %i
        datatype = f[dsets[timesteps[0]][0]].dtype
        assert datatype.char not in 'FDG', "Cannot use generate_xdmf to visualize complex data."
        prec = 4 if datatype is dtype('float32') else 8
        xff = {}
        geometry = {}
        topology = {}
        attrs = {}
        grid = {}
        NN = {}
        for name in dsets[timesteps[0]]:
            if 'slice' in name:
                slices = name.split("/")[2]
            else:
                slices = 'whole'
            cc = copy.copy(coor)
            if not slices in xff:
                xff[slices] = copy.copy(xdmffile)
                NN[slices] = N = f[name].shape
                if 'slice' in slices:
                    ss = slices.split("_")
                    ii = []
                    for i, sx in enumerate(ss):
                        if 'slice' in sx:
                            ii.append(i)
                    cc = take(coor, ii)
                else:
                    ii = list(range(ndim))
                if 'domain' in keys:
                    geo = get_geometry(kind=0, dim=ndim)
                    if ndim == 2:
                        assert len(ii) == 2
                        i, j = ii
                        geometry[slices] = geo.format(f['domain/{}'.format(coor[i])][0],
                                                      f['domain/{}'.format(coor[j])][0],
                                                      f['domain/{}'.format(coor[i])][1]/(N[0]-periodic[i]),
                                                      f['domain/{}'.format(coor[j])][1]/(N[1]-periodic[j]))
                    else:
                        assert len(ii) == 3
                        i, j, k = ii
                        geometry[slices] = geo.format(f['domain/{}'.format(coor[i])][0],
                                                      f['domain/{}'.format(coor[j])][0],
                                                      f['domain/{}'.format(coor[k])][0],
                                                      f['domain/{}'.format(coor[i])][1]/(N[0]-periodic[i]),
                                                      f['domain/{}'.format(coor[j])][1]/(N[1]-periodic[j]),
                                                      f['domain/{}'.format(coor[k])][1]/(N[2]-periodic[k]))
                    topology[slices] = get_topology(N, kind=0)
                elif 'mesh' in keys:
                    geo = get_geometry(kind=1, dim=ndim)
                    if ndim == 2:
                        sig = (prec, N[0], N[1], h5filename, cc[0], cc[1])
                    else:
                        sig = (prec, N[0], N[1], N[2], h5filename, cc[2], cc[1], cc[0])
                    geometry[slices] = geo.format(*sig)
                    topology[slices] = get_topology(N, kind=1)
                grid[slices] = ''

        # if slice of data, need to know along which axes
        # Since there may be many different slices, we need to create
        # one xdmf-file for each composition of slices
        pp = pprint.PrettyPrinter()
        attrs = {}
        for tstep in timesteps:
            d = dsets[tstep]
            slx = set()
            for i, x in enumerate(d):
                slices = x.split("/")[2]
                if not 'slice' in slices:
                    slices = 'whole'
                N = NN[slices]
                if slices not in attrs:
                    attrs[slices] = ''
                attrs[slices] += get_attribute(x, h5filename, N, prec)
                slx.add(slices)
            for slices in slx:
                grid[slices] += get_grid(geometry[slices], topology[slices], attrs[slices])
                attrs[slices] = ''

        for slices, ff in six.iteritems(xff):
            fname = h5filename[:-3]+"_"+slices+".xdmf" if 'slice' in slices else h5filename[:-3]+".xdmf"
            xfl = open(fname, "w")
            h = ff.format(tt, len(timesteps), grid[slices].rstrip())
            xfl.write(h)
            xfl.close()

if __name__ == "__main__":
    import sys
    generate_xdmf(sys.argv[-1])

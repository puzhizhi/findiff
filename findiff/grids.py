import numpy as np

from findiff.arithmetic import Node


class Grid(object):
    pass


class UniformGrid(Grid):

    def __init__(self, shape, spac, center=None):

        if not hasattr(shape, '__len__'):
            self.shape = shape,
            self.ndims = 1
        else:
            self.shape = shape
            self.ndims = len(shape)

        if not hasattr(spac, '__len__'):
            self.spac = spac,
        else:
            self.spac = spac

        if center is None:
            self.center = np.zeros(self.ndims)
        else:
            assert len(center) == self.ndims
            self.center = np.array(center)

    def spacing(self, axis):
        return self.spac[axis]


class Coordinate(Node):

    def __init__(self, axis):
        assert axis >= 0 and axis == int(axis)
        self.name = 'x_{%d}' % axis
        self.axis = axis

    def __eq__(self, other):
        return self.axis == other.axis

    def apply(self, f, grid, *args, **kwargs):
        return grid.meshed_coords[self.axis] * f


class EquidistantGrid:

    def __init__(self, *args):
        self.ndims = len(args)
        self.coords = [np.linspace(*arg) for arg in args]
        self.meshed_coords = np.meshgrid(*self.coords, indexing='ij')
        self.spacings = np.array(
            [self.coords[axis][1] - self.coords[axis][0] for axis in range(len(self.coords))]
        )

    def spacing(self, axis):
        return self.spacings[axis]

    @property
    def shape(self):
        return tuple(len(c) for c in self.coords)

    @classmethod
    def from_spacings(cls, ndims, spacings):
        args = []
        for axis in range(ndims):
            if axis in spacings:
                h = spacings[axis]
                args.append((0, h * 20, 21))
            else:
                args.append((0, 1, 11))
        return EquidistantGrid(*args)

import numpy as np

from findiff.arithmetic import Combinable


class Coordinate(Combinable):

    def __init__(self, axis):
        assert axis >= 0 and axis == int(axis)
        super(Coordinate, self).__init__()
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
                args.append((0, 10, 11))
        return EquidistantGrid(*args)

    @classmethod
    def from_shape_and_spacings(cls, shape, spacings):
        args = []
        for axis in range(len(shape)):
            if axis in spacings:
                h = spacings[axis]
                args.append((0, h * (shape[axis]-1), shape[axis]))
            else:
                args.append((0, shape[axis]-1, shape[axis]))
        return EquidistantGrid(*args)
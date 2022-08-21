import numpy as np

from findiff.arithmetic import Node
from findiff.deriv import PartialDerivative
from findiff.grids import EquidistantGrid


class FinDiff(PartialDerivative):

    def __init__(self, *args, **kwargs):

        self.acc = 2

        if 'acc' in kwargs:
            self.acc = kwargs['acc']

        degrees, spacings = self._parse_args(args)
        super(FinDiff, self).__init__(degrees)

        self.grid = EquidistantGrid.from_spacings(max(degrees.keys()) + 1, spacings)

    def __call__(self, f, acc=None):
        return super().apply(f, self.grid, self.acc)

    def apply(self, f):
        return super().apply(f, self.grid, self.acc)

    def _parse_args(self, args):
        assert len(args) > 0
        canonic_args = []

        def parse_tuple(tpl):
            if len(tpl) == 2:
                canonic_args.append(tpl + (1,))
            elif len(tpl) == 3:
                canonic_args.append(tpl)
            else:
                raise ValueError('Invalid input format for FinDiff.')

        if not hasattr(args[0], '__len__'):  # we expect a pure derivative
            parse_tuple(args)
        else:  # we have a mixed partial derivative
            for arg in args:
                parse_tuple(arg)

        degrees = {}
        spacings = {}
        for axis, spacing, degree in canonic_args:
            if axis in degrees:
                raise ValueError('FinDiff: Same axis specified twice.')
            degrees[axis] = degree
            spacings[axis] = spacing

        return degrees, spacings


class Diff:

    def __init__(self, *args):
        if len(args) == 1 and type(args[0]) == dict:
            pass
        elif len(args) == 2:
            axis, degree = args
        elif len(args) == 1:
            axis, degree = args[0], 1

        self.partial = PartialDerivative(({axis: degree}))

    def __call__(self, f, **kwargs):
        if 'spacing' in kwargs:
            spacing = kwargs['spacing']
            spacing = np.array([spacing.get(axis, 1) for axis in range(f.ndim)])
            ends = spacing * (np.array(f.shape) - 1)
            args = [(0, end, num_points) for end, num_points in zip(ends, f.shape)]
            grid = EquidistantGrid(*args)
        if 'acc' in kwargs:
            acc = kwargs['acc']
        else:
            acc = 2

        return self.partial.apply(f, grid, acc)

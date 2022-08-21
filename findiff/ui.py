import numpy as np

from findiff.continuous import PartialDerivative
from findiff.discrete import EquidistantGrid, discretized


class Diff:

    def __init__(self, *args):
        if len(args) == 1 and type(args[0]) == dict:
            pass
        elif len(args) == 2:
            axis, degree = args
        elif len(args) == 1:
            axis, degree = args[0], 1

        self.exact = PartialDerivative(({axis: degree}))
        self.discrete = None

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

        self.discrete = discretized(self.exact, grid, acc)
        return self.discrete.apply(f)

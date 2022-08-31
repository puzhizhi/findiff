import numpy as np

from findiff.arithmetic import Combinable, Numberlike, Add, Mul, Operation
from findiff.deriv import PartialDerivative, matrix_repr
from findiff.grids import EquidistantGrid
from findiff.stencils import StencilSet


class FinDiff(Combinable):

    def __init__(self, *args, **kwargs):

        super(FinDiff, self).__init__()
        self.add_handler = DirtyAdd
        self.mul_handler = DirtyMul
        self.acc = 2

        if 'acc' in kwargs:
            self.acc = kwargs['acc']

        degrees, spacings = self._parse_args(args)
        self.partial = PartialDerivative(degrees)

        # The old FinDiff API does not fully specify the grid.
        # So use a dummy-grid for all non-specified values:
        self.grid = EquidistantGrid.from_spacings(max(degrees.keys()) + 1, spacings)
        self._user_specified_spacings = spacings

    def __call__(self, f, acc=None):
        self.acc = acc or self.acc
        return self.apply(f)

    def apply(self, f):
        return self.partial.apply(f, self.grid, self.acc)

    def matrix(self, shape, acc=None):
        acc = acc or self.acc
        if shape != self.grid.shape:
            # The old FinDiff API does not fully specify the grid.
            # The constructor tentatively constructed a dummy grid. Now
            # update this information. In particular, we now know the exact
            # number of space dimensions:
            self.grid = EquidistantGrid.from_shape_and_spacings(shape, self._user_specified_spacings)
        return self.partial.matrix_repr(self.grid, acc)

    def stencil(self, shape):
        return StencilSet(self, shape)

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



class DirtyMixin:

    def __init__(self):
        self.add_handler = DirtyAdd
        self.mul_handler = DirtyMul

    def matrix(self, shape):
        if isinstance(self, Operation):
            left = self.left.matrix(shape)
            right = self.right.matrix(shape)
            return self.operation(left, right)
        elif not isinstance(self, FinDiff):
            grid = EquidistantGrid.from_shape_and_spacings(shape, {})
            return matrix_repr(self, grid, 2)
        return self.matrix(shape)

    def stencil(self, shape):
        return StencilSet(self, shape)


class DirtyNumberlike(DirtyMixin, Numberlike):

    def __init__(self, value):
        Numberlike.__init__(self, value)
        DirtyMixin.__init__(self)

class Coef(DirtyNumberlike):
    def __init__(self, value):
        super(Coef, self).__init__(value)

class Identity(DirtyNumberlike):

    def __init__(self):
        super(Identity, self).__init__(1)

    def __call__(self, f, *args, **kwargs):
        return f


class DirtyAdd(DirtyMixin, Add):
    wrapper_class = DirtyNumberlike

    def __init__(self, *args, **kwargs):
        Add.__init__(self, *args, **kwargs)
        DirtyMixin.__init__(self)


class DirtyMul(DirtyMixin, Mul):
    wrapper_class = DirtyNumberlike

    def __init__(self, *args, **kwargs):
        Mul.__init__(self, *args, **kwargs)
        DirtyMixin.__init__(self)

import math
from copy import deepcopy

import numpy as np

from findiff.continuous import PartialDerivative
from findiff.arithmetic import Operation, Node


class Stencil1D:

    def __init__(self, deriv, offsets, dx):
        self.offsets = offsets
        self.deriv = deriv
        self.dx = dx

        A = self._build_matrix()
        rhs = self._build_rhs()
        self.coefs = np.linalg.solve(A, rhs) * (self.dx ** (-self.deriv))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.data())

    @property
    def data(self):
        return {off: coef for off, coef in zip(self.offsets, self.coefs)}

    def _build_matrix(self):
        """Constructs the equation system matrix for the finite difference coefficients"""
        A = [([1 for _ in self.offsets])]
        for i in range(1, len(self.offsets)):
            A.append([j ** i for j in self.offsets])
        return np.array(A, dtype='float')

    def _build_rhs(self):
        """The right hand side of the equation system matrix"""
        b = [0 for _ in self.offsets]
        b[self.deriv] = math.factorial(self.deriv)
        return np.array(b, dtype='float')


class SymmetricStencil1D(Stencil1D):

    def __init__(self, deriv, dx, acc=2):
        assert acc % 2 == 0
        num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
        p = num_central // 2
        offsets = np.array(list(range(-p, p + 1)))
        super(SymmetricStencil1D, self).__init__(deriv, offsets, dx)


class ForwardStencil1D(Stencil1D):
    def __init__(self, deriv, dx, acc=2):
        assert acc % 2 == 0
        num_coefs = 2 * math.floor((deriv + 1) / 2) - 1 + acc
        if deriv % 2 == 0:
            num_coefs = num_coefs + 1
        offsets = np.array(list(range(0, num_coefs)))
        super(ForwardStencil1D, self).__init__(deriv, offsets, dx)


class BackwardStencil1D(Stencil1D):
    def __init__(self, deriv, dx, acc=2):
        assert acc % 2 == 0
        num_coefs = 2 * math.floor((deriv + 1) / 2) - 1 + acc
        if deriv % 2 == 0:
            num_coefs = num_coefs + 1
        offsets = -np.array(list(range(0, num_coefs)))
        super(BackwardStencil1D, self).__init__(deriv, offsets[::-1], dx)


class StencilSet1D:
    SCHEME_CENTRAL = 'central'
    SCHEME_FORWARD = 'forward'
    SCHEME_BACKWARD = 'backward'

    def __init__(self, deriv, dx, acc=2):
        self.stencils = {
            self.SCHEME_CENTRAL: SymmetricStencil1D(deriv, dx, acc),
            self.SCHEME_FORWARD: ForwardStencil1D(deriv, dx, acc),
            self.SCHEME_BACKWARD: BackwardStencil1D(deriv, dx, acc)
        }

    def get_stencil_data(self, scheme):
        return self.stencils[scheme].data

    def get_boundary_size(self):
        return max(self.stencils[self.SCHEME_CENTRAL].offsets)

    def get_num_points_side(self, scheme):
        offs = self.stencils[scheme].offsets
        return abs(min(offs)), max(offs)


class DiscretizedPartialDerivative(Node):

    def __init__(self, partial, grid, acc=2):
        self.partial = partial
        self.grid = grid
        self.acc = acc

        self.stencil_sets = dict()
        for axis in partial.axes:
            deriv = partial.degree(axis)
            h = grid.spacing(axis)
            self.stencil_sets[deriv] = StencilSet1D(deriv, h, acc)

    def apply(self, arr):

        for axis in self.partial.axes:
            res = np.zeros_like(arr)
            deriv = self.partial.degree(axis)
            stencil_set = self.stencil_sets[deriv]
            left, right = stencil_set.get_num_points_side(StencilSet1D.SCHEME_CENTRAL)
            right = arr.shape[axis] - right
            res = self._apply_axis(res, arr, axis,
                                   stencil_set.get_stencil_data(StencilSet1D.SCHEME_CENTRAL),
                                   left, right)

            res = self._apply_axis(res, arr, axis,
                                   stencil_set.get_stencil_data(StencilSet1D.SCHEME_FORWARD),
                                   0, stencil_set.get_boundary_size())

            res = self._apply_axis(res, arr, axis,
                                   stencil_set.get_stencil_data(StencilSet1D.SCHEME_BACKWARD),
                                   arr.shape[axis] - stencil_set.get_boundary_size(), arr.shape[axis])
            arr = res

        return res

    def _apply_axis(self, res, arr, axis, stencil_data, left, right):
        base_sl = slice(left, right)
        multi_base_sl = [slice(None, None)] * arr.ndim
        multi_base_sl[axis] = base_sl
        for off, coef in stencil_data.items():
            off_sl = slice(left + off, right + off)
            multi_off_sl = [slice(None, None)] * arr.ndim
            multi_off_sl[axis] = off_sl
            res[tuple(multi_base_sl)] += coef * arr[tuple(multi_off_sl)]
        return res


class EquidistantGrid:

    def __init__(self, *args):
        self.ndims = len(args)
        self.coords = [np.linspace(*arg) for arg in args]
        self.spacings = np.array(
            [self.coords[axis][1] - self.coords[axis][0] for axis in range(len(self.coords))]
        )

    def spacing(self, axis):
        return self.spacings[axis]


def discretized(expr, grid, acc):
    if isinstance(expr, Operation):
        expr = deepcopy(expr)
        expr.replace(
            lambda p: type(p) == PartialDerivative,
            lambda p: DiscretizedPartialDerivative(p, grid, acc)
        )
        return expr
    elif isinstance(expr, PartialDerivative):
        return DiscretizedPartialDerivative(expr, grid, acc)

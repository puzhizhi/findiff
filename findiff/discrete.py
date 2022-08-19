import math

import numpy as np


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

    def __init__(self, deriv, dx, acc=2):
        self.forward = ForwardStencil1D(deriv, dx, acc)
        self.backward = BackwardStencil1D(deriv, dx, acc)
        self.central = SymmetricStencil1D(deriv, dx, acc)


class DiscretizedPartialDerivative:

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
        res = np.zeros_like(arr)

        for axis in self.partial.axes:
            deriv = self.partial.degree(axis)
            stencil = self.stencil_sets[deriv].central
            left = abs(min(stencil.offsets))
            right = arr.shape[axis] - max(stencil.offsets)
            res = self._apply_axis(res, arr, axis, stencil, left, right)
            bdnry_size = left

            stencil = self.stencil_sets[deriv].forward
            res = self._apply_axis(res, arr, axis, stencil, 0, bdnry_size)

            stencil = self.stencil_sets[deriv].backward
            res = self._apply_axis(res, arr, axis, stencil, arr.shape[axis] - bdnry_size, arr.shape[axis])

        return res

    def _apply_axis(self, res, arr, axis, stencil, left, right):
        base_sl = slice(left, right)
        multi_base_sl = [slice(None, None)] * arr.ndim
        multi_base_sl[axis] = base_sl
        for off, coef in stencil.data.items():
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

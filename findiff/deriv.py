import numpy as np

from findiff.arithmetic import Node, Mul, Add
from findiff.stencils import StencilStore, SymmetricStencil1D, ForwardStencil1D, BackwardStencil1D


class PartialDerivative(Node):

    def __init__(self, degrees):
        """ Representation of a (possibly mixed) partial derivative.

        Instances of this class are meant to be immutable.

        Parameters
        ----------
        degrees:    dict
            Dictionary describing the partial derivative. key: value <==> axis: degree
        """
        self._validate_degrees(degrees)
        self.degrees = degrees

    def degree(self, axis):
        return self.degrees.get(axis, 0)

    @property
    def axes(self):
        return sorted(self.degrees.keys())

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        if type(other) != PartialDerivative:
            return Mul(self, other)
        new_degrees = dict(self.degrees)
        for axis, degree in other.degrees.items():
            if axis in new_degrees:
                new_degrees[axis] += degree
            else:
                new_degrees[axis] = degree
        return PartialDerivative(new_degrees)

    def __rmul__(self, other):
        assert type(other) != PartialDerivative
        return Mul(other, self)

    def __repr__(self):
        return str(self.degrees)

    def __str__(self):
        return str(self.degrees)

    def __pow__(self, power):
        assert int(power) == power and power > 0
        return PartialDerivative({axis: degree * power for axis, degree in self.degrees.items()})

    def __eq__(self, other):
        return self.degrees == other.degrees

    def __hash__(self):
        return hash(
            tuple(
                [(k, self.degrees[k]) for k in sorted(self.degrees.keys())]
            )
        )

    def _validate_degrees(self, degrees):
        assert isinstance(degrees, dict)
        for axis, degree in degrees.items():
            if not isinstance(axis, int) or axis < 0:
                raise ValueError('Axis must be positive integer')
            if not isinstance(degree, int) or degree <= 0:
                raise ValueError('Degree must be positive integer')

    def apply(self, arr, grid, acc):

        for axis in self.axes:
            res = np.zeros_like(arr)
            deriv = self.degree(axis)
            spacing = grid.spacing(axis)
            stencil = StencilStore.get_stencil(SymmetricStencil1D, deriv=deriv, acc=acc, spacing=spacing)
            left, right = stencil.get_num_points_side()
            right = arr.shape[axis] - right
            res = self._apply_axis(res, arr, axis,
                                   stencil.data,
                                   left, right)
            bndry_size = stencil.get_boundary_size()

            stencil = StencilStore.get_stencil(ForwardStencil1D, deriv=deriv, acc=acc, spacing=spacing)
            res = self._apply_axis(res, arr, axis,
                                   stencil.data,
                                   0, bndry_size)

            stencil = StencilStore.get_stencil(BackwardStencil1D, deriv=deriv, acc=acc, spacing=spacing)
            res = self._apply_axis(res, arr, axis,
                                   stencil.data,
                                   arr.shape[axis] - bndry_size, arr.shape[axis])
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



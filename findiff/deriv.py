import numpy as np
import scipy

from findiff.arithmetic import Node, Mul, Add, Numberlike, Operation
from findiff.grids import Coordinate
from findiff.stencils import StencilStore, SymmetricStencil1D, ForwardStencil1D, BackwardStencil1D
from findiff.utils import to_long_index, long_indices_as_ndarray


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

    def __neg__(self):
        return Mul(-1, self)

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

            # Apply symmetric stencil in the interior of the grid,
            # wherever possible:
            stencil = StencilStore.get_stencil(SymmetricStencil1D, deriv=deriv, acc=acc, spacing=spacing)
            left, right = stencil.get_num_points_side()
            right = arr.shape[axis] - right
            res = self._apply_axis(res, arr, axis,
                                   stencil.data,
                                   left, right)

            # In the boundary of the symmetric stencil, apply
            # one-sided stencils instead (forward/backward):
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

    def matrix_repr(self, grid, acc):
        """Returns the matrix representation of the partial derivative on a given grid."""

        long_indices_nd = long_indices_as_ndarray(grid.shape)

        mats = []
        for axis in self.axes:
            deriv = self.degree(axis)
            if deriv == 0:
                continue
            siz = np.prod(grid.shape)
            mat = scipy.sparse.lil_matrix((siz, siz))

            center, forward, backward = [StencilStore.get_stencil(
                stype, deriv=deriv, acc=acc,
                spacing=grid.spacing(axis))
                for stype in (SymmetricStencil1D, ForwardStencil1D, BackwardStencil1D)]

            for stencil in (center, forward, backward):

                # translate offsets of given scheme to long format
                offsets_long = []
                for o_1d in stencil.offsets:
                    o_nd = np.zeros(grid.ndims)
                    o_nd[axis] = o_1d
                    o_long = to_long_index(o_nd, grid.shape)
                    offsets_long.append(o_long)

                if type(stencil) == SymmetricStencil1D:
                    nside = stencil.get_boundary_size()
                    multi_slice = [slice(None, None)] * grid.ndims
                    multi_slice[axis] = slice(nside, -nside)
                    Is = long_indices_nd[tuple(multi_slice)].reshape(-1)
                elif type(stencil) == ForwardStencil1D:
                    multi_slice = [slice(None, None)] * grid.ndims
                    multi_slice[axis] = slice(0, nside)
                    Is = long_indices_nd[tuple(multi_slice)].reshape(-1)
                else:
                    multi_slice = [slice(None, None)] * grid.ndims
                    multi_slice[axis] = slice(-nside, None)
                    Is = long_indices_nd[tuple(multi_slice)].reshape(-1)

                for o, c in zip(offsets_long, stencil.coefficients):
                    mat[Is, Is + o] = c

            # done with the axis, convert to csr_matrix for faster arithmetic
            mats.append(scipy.sparse.csr_matrix(mat))

        # return the matrix product
        mat = mats[0]
        for i in range(1, len(mats)):
            mat = mat.dot(mats[i])
        return mat

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


def matrix_repr(expr, grid, acc):
    """Returns the matrix representation of a given differential operator an a grid."""

    if isinstance(expr, PartialDerivative):
        return expr.matrix_repr(grid, acc)
    elif isinstance(expr, Numberlike):
        siz = np.prod(grid.shape)
        if isinstance(expr.value, np.ndarray):
            fill_value = expr.value.reshape(-1)
        else:
            fill_value = expr.value
        return scipy.sparse.diags(np.full((siz,), fill_value=fill_value))
    elif isinstance(expr, Coordinate):
        siz = np.prod(grid.shape)
        value = grid.meshed_coords[expr.axis].reshape(-1)
        return scipy.sparse.diags(np.full((siz,), fill_value=value))
    elif isinstance(expr, Operation):
        left_result = matrix_repr(expr.left, grid, acc)
        right_result = matrix_repr(expr.right, grid, acc)
        return expr.operation(left_result, right_result)
    else:
        raise ValueError('Cannot calculate matrix representation of type %s' % type(expr).__name__)

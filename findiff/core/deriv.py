"""This module contains the backend classes for operating with derivative
operators. These classes are not intended for direct use from outside of
the package. Users shall use the classes in the api module, which delegate
to the classes in this module as required. The classes in this module expect
valid input data and perform no checking on their own. Thorough input validation
is performed in the frontend classes of the api module which are exposed
externally.
"""

import numbers

import numpy as np
import scipy

from findiff.core.algebraic import Algebraic, Mul, Numberlike, Operation
from findiff.core.stencils import StencilStore, SymmetricStencil1D, ForwardStencil1D, BackwardStencil1D
from findiff.utils import long_indices_as_ndarray, to_long_index


class PartialDerivative(Algebraic):

    def __init__(self, degrees):
        """ Representation of a (possibly mixed) partial derivative.

        Instances of this class are meant to be immutable.

        Parameters
        ----------
        degrees:    dict
            Dictionary describing the partial derivative. key: value <==> axis: degree
        """
        super(PartialDerivative, self).__init__()
        self._validate_degrees(degrees)
        self.degrees = degrees

    def degree(self, axis):
        """Returns the derivative degree along a given axis."""
        return self.degrees.get(axis, 0)

    @property
    def axes(self):
        """Returns a sorted list of all axis along which to take derivatives."""
        return sorted(self.degrees.keys())

    def __mul__(self, other):
        """Multiply PartialDerivative instance with some other object.

        Overrides the method from the Arithmetic class to allow for merging
        two PartialDerivative objects into one.
        """
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
            if not isinstance(axis, numbers.Integral) or axis < 0:
                raise ValueError('Axis must be positive integer')
            if not isinstance(degree, numbers.Integral) or degree <= 0:
                raise ValueError('Degree must be positive integer')

    def apply(self, arr, grid, acc):

        if not isinstance(arr, np.ndarray):
            raise TypeError('Can only apply derivative to NumPy arrays. Instead got %s.' % (arr.__class__.__name__))

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


def matrix_repr(expr, acc, grid):
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
        left_result = matrix_repr(expr.left, acc, grid)
        right_result = matrix_repr(expr.right, acc, grid)
        return expr.operation(left_result, right_result)
    else:
        raise TypeError('Cannot calculate matrix representation of type %s' % type(expr).__name__)


class Coordinate(Algebraic):

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
    """Utility class for representing an equidistant grid of any dimension."""

    def __init__(self, *args):
        """Creates an EquidistandGrid object.

        Parameters
        ----------
        args : variable number of tuples
            Tuples for the form (from, to, num_points) specifying the domain along
            all axes.
        """
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
        """Factory method to create a (dummy) Equidistant grid from the total number of dimensinos and spacings

        Parameters
        ----------
        ndims : int > 0
            The total number of space dimensions.
        spacings : dict
            The grid spacings along all required axes. (key = axis, value = spacing along axis)

        Returns
        -------
        out : EquidistantGrid
            The generated grid.
        """
        args = []
        cls._validate_spacings(spacings)

        for axis in range(ndims):
            if axis in spacings:
                h = spacings[axis]
                args.append((0, h * 20, 21))
            else:
                args.append((0, 10, 11))
        return EquidistantGrid(*args)

    @classmethod
    def _validate_spacings(cls, spacings):
        for axis, spacing in spacings.items():
            if axis < 0 or not isinstance(axis, numbers.Integral):
                raise InvalidGrid('axis must be non-negative integer.')
            if spacing <= 0:
                raise InvalidGrid('spacing must be positive number.')

    @classmethod
    def from_shape_and_spacings(cls, shape, spacings):
        args = []
        for axis in range(len(shape)):
            if axis in spacings:
                h = spacings[axis]
                args.append((0, h * (shape[axis] - 1), shape[axis]))
            else:
                args.append((0, shape[axis] - 1, shape[axis]))
        return EquidistantGrid(*args)


class FinDiffException(Exception):
    pass


class InvalidGrid(FinDiffException):
    pass


class InvalidArraySize(FinDiffException):
    pass

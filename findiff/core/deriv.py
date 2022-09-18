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

from findiff.core.algebraic import Algebraic, Mul
from findiff.core.grids import Spacing, EquidistantGrid
from findiff.core.stencils import StencilStore, ForwardStencil, SymmetricStencil, BackwardStencil, StencilSet
from findiff.utils import long_indices_as_ndarray, to_long_index


class PartialDerivative(Algebraic):
    """Representation of a (possibly mixed) partial derivative.

        Instances of this class are meant to be immutable.
    """

    def __init__(self, degrees):
        """Constructor

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

    def apply(self, arr, grid_or_spacing, acc):
        """Applies the partial derivative to an array.

        Parameters
        ----------
        arr : array-like
            The array that shall be differentiated.
        grid : Spacing
            The spacing(s) of the numerical grid.
        acc : positive even int
            The accuracy order.

        Returns
        -------
        out : array-like
            The differentiated array. Same shape as input array.
        """

        if not isinstance(arr, np.ndarray):
            raise TypeError('Can only apply derivative to NumPy arrays. Instead got %s.' % (arr.__class__.__name__))

        if isinstance(grid_or_spacing, EquidistantGrid):
            grid = grid_or_spacing
            spacing = Spacing({axis: grid.spacing(axis) for axis in self.axes})
        else:
            spacing = grid_or_spacing
        stencil_set = StencilSet(self, spacing, arr.ndim, acc)
        return stencil_set.apply(arr)

    def matrix_repr(self, grid, acc):
        """Returns the matrix representation of the partial derivative on a given grid."""

        long_indices_nd = long_indices_as_ndarray(grid.shape)

        mats = []
        for axis in self.axes:
            deriv = self.degree(axis)
            siz = np.prod(grid.shape)
            mat = scipy.sparse.lil_matrix((siz, siz))

            center, forward, backward = [StencilStore.get_stencil(
                stype, deriv, grid.spacing(axis), acc)
                for stype in (SymmetricStencil, ForwardStencil, BackwardStencil)]

            for stencil in (center, forward, backward):

                # translate offsets of given scheme to long format
                offsets_long = []
                offsets = np.array([off[0] for off in stencil.offsets])
                for o_1d in offsets:
                    o_nd = np.zeros(grid.ndims)
                    o_nd[axis] = o_1d
                    o_long = to_long_index(o_nd, grid.shape)
                    offsets_long.append(o_long)

                if type(stencil) == SymmetricStencil:
                    nside = np.max(np.abs(offsets))
                    multi_slice = [slice(None, None)] * grid.ndims
                    multi_slice[axis] = slice(nside, -nside)
                    Is = long_indices_nd[tuple(multi_slice)].reshape(-1)
                elif type(stencil) == ForwardStencil:
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

    def _validate_degrees(self, degrees):
        assert isinstance(degrees, dict)
        for axis, degree in degrees.items():
            if not isinstance(axis, numbers.Integral) or axis < 0:
                raise ValueError('Axis must be positive integer')
            if not isinstance(degree, numbers.Integral) or degree <= 0:
                raise ValueError('Degree must be positive integer')

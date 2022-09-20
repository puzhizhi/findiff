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

from findiff.core import DEFAULT_ACCURACY
from findiff.core.algebraic import Algebraic, Mul
from findiff.core.grids import Spacing, EquidistantGrid
from findiff.core.stencils import StandardStencilSet
from findiff.utils import require_exactly_one_parameter, parse_spacing


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

        Overrides the method from the Algebraic class to allow for merging of
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

    def apply(self, arr, **kwargs):
        """Applies the partial derivative to an array.

        Parameters
        ----------
        arr : array-like
            The array that shall be differentiated.
        grid_or_spacing : Spacing or EquidistantGrid
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

        found_para = require_exactly_one_parameter(
            ['spacing', 'grid'], kwargs, 'PartialDerivative.apply')

        if found_para == 'grid':
            grid = kwargs['grid']
            spacing = Spacing({axis: grid.spacing(axis) for axis in self.axes})
        else:
            spacing = parse_spacing(kwargs['spacing'])

        acc = kwargs.get('acc', DEFAULT_ACCURACY)

        stencil_set = StandardStencilSet(self, spacing, arr.ndim, acc)
        return stencil_set.apply(arr)

    def _validate_degrees(self, degrees):
        assert isinstance(degrees, dict)
        for axis, degree in degrees.items():
            if not isinstance(axis, numbers.Integral) or axis < 0:
                raise ValueError('Axis must be positive integer')
            if not isinstance(degree, numbers.Integral) or degree <= 0:
                raise ValueError('Degree must be positive integer')

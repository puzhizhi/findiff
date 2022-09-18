import numbers

import numpy as np
import sympy
from sympy import Symbol

from findiff.core.exceptions import InvalidGrid


class Spacing:

    def __init__(self, spacing_dict):
        if isinstance(spacing_dict, dict):
            self.isotrop = False
            for axis, value in spacing_dict.items():
                if isinstance(value, str):
                    spacing_dict[axis] = sympy.Symbol(value)
                elif isinstance(value, Symbol):
                    pass # leave entry as it is
                elif value <= 0:
                    raise InvalidGrid('Spacing value must be positive.')
            self._data = spacing_dict
        else:
            self.isotrop = True
            if isinstance(spacing_dict, str):
                spacing_dict = Symbol(spacing_dict)
            elif isinstance(spacing_dict, Symbol):
                pass # leave spacing_dict as it is
            elif spacing_dict <= 0:
                raise InvalidGrid('Spacing value must be positive.')
            self._data = spacing_dict

    def for_axis(self, axis):
        if self.isotrop:
            return self._data
        if axis not in self._data:
            raise InvalidGrid('Axis %d is not defined.' % axis)
        return self._data[axis]

    def __getitem__(self, axis):
        if self.isotrop:
            return self._data
        return self._data[axis]

    def __setitem__(self, axis, value):
        if self.isotrop:
            raise ValueError('Cannot set single axis spacing for isotropic object.')
        self._data[axis] = value

    def keys(self):
        if self.isotrop:
            raise ValueError('Cannot infer number of dimensions for isotropic spacing.')
        return self._data.keys()

    def values(self):
        if self.isotrop:
            raise ValueError('Cannot infer number of dimensions for isotropic spacing.')
        return self._data.values()

    def items(self):
        if self.isotrop:
            raise ValueError('Cannot infer number of dimensions for isotropic spacing.')
        return self._data.items()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self._data)


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

    def to_spacing(self):
        return Spacing({axis: self.spacings[axis] for axis in range(self.ndims)})

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

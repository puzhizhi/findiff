""" This module contains classes that are intended for user interaction.

    The most important class is `Diff` which allows for a convenient way
    to define partial derivatives and compose general differential operators.
"""
import numbers

from findiff.arithmetic import Arithmetic, Numberlike
from findiff.deriv import PartialDerivative, EquidistantGrid, InvalidGrid

__all__ = ['Diff', 'Coef']

class Diff(Arithmetic):

    def __init__(self, *args):
        """Defines a (possibly mixed) partial derivative operator.

        Note the difference between defining the derivative operator and applying it.
        For applying the derivative operator, call it, once it is defined.

        Parameters
        ----------
        args:   Variable list of arguments specifying the kind of partial derivative.

            If exactly one integer argument is given, it means 'axis', where 'axis' is
            a positive integer, denoting the axis along which to take the (first, degree=1)
            derivative.

            If exactly one dictionary argument is given, it specifies a general, possibly
            mixed partial derivative. Each key denotes an axis along which to take a partial
            derivative, and the corresponding value denotes the degree of the derivative.

            If two integer arguments are given, the first denotes the axis along which
            to take the derivative, the second denotes the degree of the derivative.

        """
        super(Diff, self).__init__()
        if len(args) == 1 and type(args[0]) == dict:
            degrees = args[0]
        elif len(args) == 2:
            axis, degree = args
            degrees = {axis: degree}
        elif len(args) == 1:
            axis, degree = args[0], 1
            degrees = {axis: degree}
        else:
            raise ValueError('Diff constructor has received invalid argument(s): ' + str(args))

        self._validate_degrees_dict(degrees)

        self.partial = PartialDerivative(degrees)

    def __call__(self, f, **kwargs):
        return self.apply(f, **kwargs)

    def apply(self, f, **kwargs):
        if 'spacing' in kwargs:
            spacing = kwargs['spacing']

            if not isinstance(spacing, dict):
                raise InvalidGrid('spacing keyword argument must be a dict.')

            # Assert that spacings along all axes are defined, where derivatives need:
            for axis in self.partial.axes:
                if axis not in spacing:
                    raise InvalidGrid('No spacing along axis %d defined.' % axis)

            ndims_effective = max(spacing.keys()) + 1
            grid = EquidistantGrid.from_spacings(ndims_effective, spacing)
        else:
            raise InvalidGrid('No spacing defined when applying Diff.')

        if 'acc' in kwargs:
            acc = kwargs['acc']
        else:
            acc = 2

        return self.partial.apply(f, grid, acc)

    def _validate_degrees_dict(self, degrees):
        assert isinstance(degrees, dict)
        for axis, degree in degrees.items():
            if not isinstance(axis, numbers.Integral) or axis < 0:
                raise ValueError('Axis must be positive integer')
            if not isinstance(degree, numbers.Integral) or degree <= 0:
                raise ValueError('Degree must be positive integer')


class Coef(Numberlike):
    def __init__(self, value):
        super(Coef, self).__init__(value)


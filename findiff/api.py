""" This module contains classes that are intended for user interaction.

    The most important class is `Diff` which allows for a convenient way
    to define partial derivatives and compose general differential operators.
"""
import numbers

import findiff.legacy
from findiff.core.algebraic import Algebraic, Numberlike
from findiff.core.deriv import PartialDerivative, EquidistantGrid, InvalidGrid, InvalidArraySize

__all__ = ['Diff', 'Coef']


class Diff(Algebraic):

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
        """Applies the partial derivative operator to an array.

        The function delegates to self.apply(f, **kwargs).

        Parameters
        ----------
        f : numpy.ndarray
            The array on which to apply the derivative operator

        kwargs : required keyword arguments

            Keywords:

                spacing : dict
                    Dictionary specifying the grid spacing (key=axis, value=spacing).

                acc :  even int > 0, optional, default: 2
                    The desired accuracy order.

        Returns
        -------
        out : numpy.ndarray
            The array with the evaluated derivative. Same shape as f.

        Examples
        --------
        >> x = y = np.linspace(0, 1, 100)
        >> dx = dy = x[1] - x[0]
        >> X, Y = np.meshgrid(x, y, indexing='ij')
        >> f = X**2 * Y**2
        >> d2_dxdy = Diff({0: 1, 1: 1})    # or: Diff(0) * Diff(1)
        >> d2f_dxdy = d2_dxdy(f, spacing={0: dx, 1: dy})
        """
        return self.apply(f, **kwargs)

    def apply(self, f, **kwargs):
        """Applies the partial derivative operator to an array.

        For details, see help on __call__.
        """

        # make sure the array shape is big enough
        max_axis = max(self.partial.axes)
        if max_axis >= f.ndim:
            raise InvalidArraySize('Array has not enough dimensions for given derivative operator.'
                                   'Has %d but needs at least %d' % (f.ndim, max_axis))

        # require valid spacing
        if 'spacing' in kwargs:
            spacing = kwargs['spacing']

            if not isinstance(spacing, dict):
                is_positive_number = isinstance(spacing, numbers.Real) and spacing > 0
                if is_positive_number:
                    spacing = {axis: spacing for axis in range(f.ndim)}
                else:
                    raise InvalidGrid('spacing keyword argument must be a dict or single number.')

            # Assert that spacings along all axes are defined, where derivatives need:
            for axis in self.partial.axes:
                if axis not in spacing:
                    raise InvalidGrid('No spacing along axis %d defined.' % axis)

            ndims_effective = max(spacing.keys()) + 1
            grid = EquidistantGrid.from_spacings(ndims_effective, spacing)
        else:
            raise InvalidGrid('No spacing defined when applying Diff.')

        # accuracy is optional
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
    """Wrapper class for numbers or numpy arrays."""
    def __init__(self, value):
        super(Coef, self).__init__(value)


class Identity(Coef):
    def __init__(self):
        super(Identity, self).__init__(1)


def coefficients(deriv, acc=None, offsets=None, symbolic=False):
    both_set = acc and offsets
    none_set = not (acc or offsets)
    if both_set or none_set:
        raise ValueError('Must specify either acc or offsets parameter.')
    if acc:
        return findiff.legacy.coefficients(deriv, acc=acc, symbolic=symbolic)
    return findiff.legacy.coefficients(deriv, offsets=offsets, symbolic=True)


def matrix_repr(expr, shape=None, spacing=None):
    pass
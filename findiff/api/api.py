""" This module contains classes and functions that are intended for usage by
    the findiff package user.

    The most important class is `Diff` which allows for a convenient way
    to define partial derivatives and compose general differential operators.

    When adding classes or functions, validate the input carefully, as the
    classes in findiff.core mostly assume valid input.
"""
import numbers

import findiff.legacy
from findiff.core.algebraic import Numberlike
from findiff.core.exceptions import InvalidGrid

__all__ = ['Coef', 'Identity', 'coefficients']


class Coef(Numberlike):
    """Wrapper class for numbers or numpy arrays."""

    def __init__(self, value):
        super(Coef, self).__init__(value)


class Identity(Coef):
    """Represents the identity operator."""
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


def _validate_shape(shape):
    if not isinstance(shape, tuple):
        raise InvalidGrid('shape must be a tuple. Received: %s' % type(shape).__name__)
    for axis in shape:
        if axis <= 0:
            raise InvalidGrid('Number of grid points must be positive. Received: %s' % shape)


def _validate_and_convert_spacings(spacings, valid_shape):
    """ Validates if the specified spacings are compatible with the shape.
        Returns a dict of spacings.
    """

    ndims = len(valid_shape)

    # Spacings may be a single number, which means same spacing along all axes.
    if isinstance(spacings, numbers.Real):
        if spacings <= 0:
            raise InvalidGrid('Grid spacing must be > 0. Received: %f' % spacings)
        spacings = {axis: spacings for axis in range(ndims)}
    # Or it can be a dict. Then all axes must be provided:
    else:
        if set(spacings.keys()) != set(list(range(ndims))):
            raise InvalidGrid('Not all spacings specified in dict.')
        for spac in spacings.values():
            if spac <= 0:
                raise InvalidGrid('Grid spacing must be > 0. Received: %f' % spac)

    return spacings


def _validate_acc(acc):
    if acc <= 0 or acc % 2:
        raise ValueError('acc must be positive even integer. Received: %s' % str(acc))

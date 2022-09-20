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

"""
This module determines finite difference coefficients in one dimension
for any desired even accuracy order. It is mostly obsolete by now, as the 'stencils'
module basically is a generalization. Mainly kept for compatibility reasons.

Most important function:

coefficients(deriv, acc=None, offsets=None, symbolic=False)

to calculate the finite difference coefficients for a given derivative
order and given accuracy order to given offsets.
"""

from .stencils import Stencil, SymmetricStencil1D, ForwardStencil1D, BackwardStencil1D


def coefficients(deriv, acc=None, offsets=None, symbolic=False):
    """
    Calculates the finite difference coefficients for given derivative order and accuracy order.

    If acc is given, the coefficients are calculated for central, forward and backward
    schemes resulting in the specified accuracy order.

    If offsets are given, the coefficients are calculated for the offsets as specified
    and the resulting accuracy order is computed.

    *Either* acc *or* offsets must be given.

    Assumes that the underlying grid is uniform. This function is available at the `findiff`
    package level.

    :param deriv: The derivative order.
    :type deriv: int > 0

    :param acc: The accuracy order.
    :type acc:  even int > 0:

    :param offsets: The offsets for which to calculate the coefficients.
    :type offsets: list of ints

    :raises ValueError: if invalid arguments are given

    :return: dict with the finite difference coefficients and corresponding offsets
    """

    _validate_deriv(deriv)

    if acc and offsets:
        raise ValueError('acc and offsets cannot both be given')

    if offsets:
        return calc_coefs(deriv, offsets, symbolic)

    _validate_acc(acc)
    return {
        scheme: calc_coefs_standard(deriv, acc, scheme, symbolic)
        for scheme in ('center', 'forward', 'backward')
    }


def calc_coefs(deriv, offsets, symbolic=False):
    stencil = Stencil(offsets, {(deriv,): 1}, symbolic=symbolic)
    return {
        "coefficients": [stencil.coefficient(o) for o in offsets],
        "offsets": stencil.offsets,
        "accuracy": stencil.accuracy
    }


def calc_coefs_standard(deriv, acc, scheme, symbolic=False):
    if scheme == 'center':
        stencil = SymmetricStencil1D(deriv, 1, acc, symbolic)
    elif scheme == 'forward':
        stencil = ForwardStencil1D(deriv, 1, acc, symbolic)
    elif scheme == 'backward':
        stencil = BackwardStencil1D(deriv, 1, acc, symbolic)
    return {
        "coefficients": stencil.coefficients,
        "offsets": stencil.offsets,
        "accuracy": acc
    }


def _validate_acc(acc):
    if acc % 2 == 1 or acc <= 0:
        raise ValueError('Accuracy order acc must be positive EVEN integer')


def _validate_deriv(deriv):
    if deriv < 0:
        raise ValueError('Derive degree must be positive integer')

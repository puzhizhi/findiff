import numpy as np
import scipy

from findiff.core.algebraic import Numberlike, Operation
from findiff.core.deriv import PartialDerivative
from findiff.core.stencils import StencilStore, SymmetricStencil, ForwardStencil, BackwardStencil, StandardStencilSet, \
    TrivialStencilSet
from findiff.utils import long_indices_as_ndarray, to_long_index, require_parameter, require_exactly_one_parameter
from test.core import DEFAULT_ACCURACY


def matrix_repr(expr, **kwargs):
    """Returns the matrix representation of a given differential operator an a grid."""
    # shape, spacing=None, grid=None, acc=2

    spacing, shape, acc = _parse_matrix_repr_kwargs(**kwargs)

    if isinstance(expr, PartialDerivative):
        return _matrix_expr_of_partial_derivative(expr, spacing, shape, acc)
    elif isinstance(expr, Numberlike):
        siz = np.prod(shape)
        if isinstance(expr.value, np.ndarray):
            fill_value = expr.value.reshape(-1)
        else:
            fill_value = expr.value
        return scipy.sparse.diags(np.full((siz,), fill_value=fill_value))
    elif isinstance(expr, Operation):
        left_result = matrix_repr(expr.left, shape=shape, spacing=spacing, acc=acc)
        right_result = matrix_repr(expr.right, shape=shape, spacing=spacing, acc=acc)
        return expr.operation(left_result, right_result)
    else:
        raise TypeError('Cannot calculate matrix representation of type %s' % type(expr).__name__)


def stencils_repr(expr, **kwargs):

    spacing, ndims, acc = _parse_stencils_repr_kwargs(**kwargs)

    if isinstance(expr, PartialDerivative):
        return StandardStencilSet(expr, spacing, ndims, acc)
    elif isinstance(expr, Numberlike):
        return TrivialStencilSet(expr.value, ndims)
    elif isinstance(expr, Operation):
        left_result = stencils_repr(expr.left, ndims=ndims, spacing=spacing, acc=acc)
        right_result = stencils_repr(expr.right, ndims=ndims, spacing=spacing, acc=acc)
        return expr.operation(left_result, right_result)
    else:
        raise TypeError('Cannot calculate matrix representation of type %s' % type(expr).__name__)


def _matrix_expr_of_partial_derivative(partial, spacing, shape, acc):
    ndims = len(shape)
    long_indices_nd = long_indices_as_ndarray(shape)
    mats = []
    for axis in partial.axes:
        deriv = partial.degree(axis)
        siz = np.prod(shape)
        mat = scipy.sparse.lil_matrix((siz, siz))

        center, forward, backward = [StencilStore.get_stencil(
            stype, deriv, spacing.for_axis(axis), acc)
            for stype in (SymmetricStencil, ForwardStencil, BackwardStencil)]

        for stencil in (center, forward, backward):

            # Stencils store offsets as tuples, even in the 1D case. Unwrap them here:
            offsets = [off[0] for off in stencil.offsets]

            # translate offsets of given scheme to long format
            offsets_long = []
            for o_1d in offsets:
                o_nd = np.zeros(ndims)
                o_nd[axis] = o_1d
                o_long = to_long_index(o_nd, shape)
                offsets_long.append(o_long)

            if type(stencil) == SymmetricStencil:
                nside = np.max(np.abs(offsets))
                multi_slice = [slice(None, None)] * ndims
                multi_slice[axis] = slice(nside, -nside)
                Is = long_indices_nd[tuple(multi_slice)].reshape(-1)
            elif type(stencil) == ForwardStencil:
                multi_slice = [slice(None, None)] * ndims
                multi_slice[axis] = slice(0, nside)
                Is = long_indices_nd[tuple(multi_slice)].reshape(-1)
            else:
                multi_slice = [slice(None, None)] * ndims
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


def _parse_matrix_repr_kwargs(**kwargs):
    found_para = require_exactly_one_parameter(['spacing', 'grid'], kwargs, 'matrix_repr')

    if found_para == 'spacing':
        spacing = kwargs[found_para]
        shape = require_parameter('shape', kwargs, 'matrix_repr')
    else: # grid given
        grid = kwargs[found_para]
        shape = grid.shape
        spacing = grid.to_spacing()
    acc = kwargs.get('acc', DEFAULT_ACCURACY)
    return spacing, shape, acc


def _parse_stencils_repr_kwargs(**kwargs):
    ## spacing, ndims, acc
    found_para = require_exactly_one_parameter(['spacing', 'grid'], kwargs, 'stencil_repr')
    if found_para == 'spacing':
        spacing = kwargs[found_para]
        ndims = require_parameter('ndims', kwargs, 'stencil_repr')
    else: # found grid
        grid = kwargs[found_para]
        spacing = grid.to_spacing()
        ndims = grid.ndims
    acc = kwargs.get('acc', DEFAULT_ACCURACY)
    return spacing, ndims, acc
import numpy as np

from findiff.core import InvalidGrid, PartialDerivative, Numberlike
from findiff.core import Spacing
from findiff.core import reprs as reprs, DEFAULT_ACCURACY
from findiff.core.algebraic import Operation
from findiff.utils import require_exactly_one_parameter, require_parameter, require_at_most_one_parameter, parse_spacing


def matrix_repr(expr, **kwargs):
    spacing, shape, acc = _parse_matrix_repr_kwargs(**kwargs)
    if acc % 2:
        acc += 1
    validate_shape(shape)
    return reprs.matrix_repr(expr, spacing=spacing, shape=shape, acc=acc)


def stencils_repr(expr, **kwargs):
    spacing, ndims, acc = _parse_stencils_repr_kwargs(**kwargs)

    if ndims is None:
        ndims = _infer_dimensions(expr)

    if acc % 2:
        acc += 1

    return reprs.stencils_repr(expr, spacing=spacing, ndims=ndims, acc=acc)


def _parse_matrix_repr_kwargs(**kwargs):
    found_para = require_exactly_one_parameter(['spacing', 'grid'], kwargs, 'matrix_repr')

    if found_para == 'spacing':
        spacing = parse_spacing(kwargs[found_para])
        shape = require_parameter('shape', kwargs, 'matrix_repr')
    else:  # grid given
        grid = kwargs[found_para]
        shape = grid.shape
        spacing = grid.to_spacing()
    acc = kwargs.get('acc', DEFAULT_ACCURACY)
    return spacing, shape, acc


def _infer_dimensions(expr):

    ndims = 0
    if isinstance(expr, PartialDerivative):
        max_axis = np.max(expr.axes)
        if max_axis + 1 > ndims:
            ndims = max_axis + 1
    elif isinstance(expr, Numberlike):
        if isinstance(expr.value, np.ndarray):
            ndims = expr.value.ndim
    elif isinstance(expr, Operation):
        ndims_left = _infer_dimensions(expr.left)
        ndims_right = _infer_dimensions(expr.right)
        if ndims_left > ndims:
            ndims = ndims_left
        if ndims_right > ndims:
            ndims = ndims_right
    else:
        raise TypeError('Cannot infer dimensions from this type: %s' % type(expr).__name__)

    return ndims


def _parse_stencils_repr_kwargs(**kwargs):
    ## spacing, ndims, acc
    found_para = require_at_most_one_parameter(['spacing', 'grid'], kwargs, 'stencil_repr')
    if found_para == 'spacing':
        spacing = parse_spacing(kwargs[found_para])
        ndims = kwargs.get('ndims')
    elif found_para == 'grid':
        grid = kwargs[found_para]
        spacing = grid.to_spacing()
        ndims = grid.ndims
    else: # nothing found
        spacing = Spacing(1)
        ndims = None

    acc = kwargs.get('acc', DEFAULT_ACCURACY)
    return spacing, ndims, acc


def validate_shape(shape):
    if not hasattr(shape, '__len__') or np.any(np.array(shape) <= 0):
        raise InvalidGrid('Invalid shape: %s' % str(shape))

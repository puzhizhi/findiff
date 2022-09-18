import numbers

import numpy as np

from findiff.core import InvalidGrid
from findiff.core import Spacing
from findiff.core import reprs as reprs, DEFAULT_ACCURACY
from findiff.utils import require_exactly_one_parameter, require_parameter


def matrix_repr(expr, **kwargs):
    spacing, shape, acc = _parse_matrix_repr_kwargs(**kwargs)
    if acc % 2:
        acc += 1
    validate_shape(shape)
    return reprs.matrix_repr(expr, spacing=spacing, shape=shape, acc=acc)


def stencils_repr(expr, **kwargs):
    spacing, ndims, acc = _parse_stencils_repr_kwargs(**kwargs)
    if acc % 2:
        acc += 1
    return reprs.stencils_repr(expr, **kwargs)


def _parse_matrix_repr_kwargs(**kwargs):
    found_para = require_exactly_one_parameter(['spacing', 'grid'], kwargs, 'matrix_repr')

    if found_para == 'spacing':
        spacing = _parse_spacing(kwargs[found_para])
        shape = require_parameter('shape', kwargs, 'matrix_repr')
    else:  # grid given
        grid = kwargs[found_para]
        shape = grid.shape
        spacing = grid.to_spacing()
    acc = kwargs.get('acc', DEFAULT_ACCURACY)
    return spacing, shape, acc


def _parse_stencils_repr_kwargs(**kwargs):
    ## spacing, ndims, acc
    found_para = require_exactly_one_parameter(['spacing', 'grid'], kwargs, 'stencil_repr')
    if found_para == 'spacing':
        spacing = _parse_spacing(kwargs[found_para])
        ndims = require_parameter('ndims', kwargs, 'stencil_repr')
    else: # found grid
        grid = kwargs[found_para]
        spacing = grid.to_spacing()
        ndims = grid.ndims
    acc = kwargs.get('acc', DEFAULT_ACCURACY)
    return spacing, ndims, acc


def _parse_spacing(spacing):
    if isinstance(spacing, Spacing):
        return spacing
    if isinstance(spacing, dict):
        return Spacing(spacing)
    if isinstance(spacing, numbers.Real):
        return Spacing(spacing)
    raise TypeError('Cannot parse this type (%s) to create Spacing instance.', type(spacing).__name__)


def validate_shape(shape):
    if not hasattr(shape, '__len__') or np.any(np.array(shape) <= 0):
        raise InvalidGrid('Invalid shape: %s' % str(shape))

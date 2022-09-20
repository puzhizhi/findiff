import numbers



def require_parameter(parameter_name, kwargs_dict, function_name):
    if parameter_name not in kwargs_dict:
        raise ValueError('%s: Parameter "%s" not given.' % (function_name, parameter_name))
    return kwargs_dict[parameter_name]


def require_exactly_one_parameter(parameter_names, kwargs_dict, function_name):
    count_paras_found = 0
    found_para = None
    for para in parameter_names:
        if para in kwargs_dict:
            count_paras_found += 1
            found_para = para
    if count_paras_found != 1:
        msg = function_name + ': Need exactly one out of ['
        msg += ', '.join(parameter_names)
        msg += ']. Found: %d.' % count_paras_found
        raise ValueError(msg)
    return found_para


def require_at_most_one_parameter(parameter_names, kwargs_dict, function_name):
    count_paras_found = 0
    found_para = None
    for para in parameter_names:
        if para in kwargs_dict:
            count_paras_found += 1
            found_para = para
    if count_paras_found > 1:
        msg = function_name + ': Need at most one out of ['
        msg += ', '.join(parameter_names)
        msg += ']. Found: %d.' % count_paras_found
        raise ValueError(msg)
    return found_para


def parse_spacing(spacing, ndims=None):
    from findiff.core import Spacing

    if isinstance(spacing, Spacing):
        if spacing.isotropic and ndims is not None:
            spacing = Spacing({axis: spacing for axis in range(ndims)})
        return spacing
    if isinstance(spacing, dict):
        return Spacing(spacing)
    if isinstance(spacing, numbers.Real):
        if spacing <= 0:
            raise ValueError('Spacing must be positive.')
        return Spacing(spacing)
    raise TypeError('Cannot parse this type (%s) to create Spacing instance.', type(spacing).__name__)

""" This module resolves name conflicts between version <1.0 and >=1.0"""


from findiff import legacy as legacy, api as api
from findiff.core import Numberlike


class Coef(Numberlike):
    """Wrapper class for constant and variable coefficients."""
    def __init__(self, value):
        self.value = value

    def __mul__(self, other):
        if hasattr(other, 'legacy'):
            resolved = legacy.Coef(self.value)
            return resolved * other
        resolved = api.Coef(self.value)
        return resolved * other

    def __rmul__(self, other):
        if hasattr(other, 'legacy'):
            resolved = legacy.Coef(self.value)
            return resolved * other
        resolved = api.Coef(self.value)
        return other * resolved

    def apply(self, target, *args, **kwargs):
        return self.value * target


class Identity(api.Identity):
    """Representation of the identity operator."""

    def __mul__(self, other):
        if hasattr(other, 'legacy'):
            resolved = legacy.Identity()
            return resolved * other
        resolved = api.Identity()
        return resolved * other

    def __rmul__(self, other):
        if hasattr(other, 'legacy'):
            resolved = legacy.Identity()
            return other * resolved
        resolved = api.Identity()
        return other * resolved

    def __add__(self, other):
        if hasattr(other, 'legacy'):
            resolved = legacy.Identity()
            return resolved + other
        resolved = api.Identity()
        return resolved + other

    def __radd__(self, other):
        if hasattr(other, 'legacy'):
            resolved = legacy.Identity()
            return other + resolved
        resolved = api.Identity()
        return other + resolved

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, target, *args, **kwargs):
        return target

    def matrix(self, shape):
        resolved = legacy.Identity()
        return resolved.matrix(shape)

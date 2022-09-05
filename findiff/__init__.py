"""
findiff is a Python package for finite difference numerical derivatives
and partial differential equations in any number of dimensions.

Classes and functions that are meant to be used by the package users
are automatically imported into 'findiff' namespace.

Classes and functions that are not imported into 'findiff' namespace
are internal and should only be used for developing the findiff package
itself, but not by package users.
"""

__version__ = '1.0.0.rc1'

# Legacy:
import findiff.legacy as legacy
import findiff.api as api
from .api import Diff, EquidistantGrid
from .legacy import FinDiff, Identity, coefficients
from .legacy import Gradient, Divergence, Laplacian, Curl
from .legacy import PDE, BoundaryConditions
from .stencils import Stencil
from .symbolics import Equation

#
# Conflicting names between v0 and v1:
#  (decide lazily which versions to use)

class Coef:
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


class Identity:

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
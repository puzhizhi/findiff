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

from .stencils import Stencil
from .symbolics import Equation

from .api import Diff, Coef, EquidistantGrid


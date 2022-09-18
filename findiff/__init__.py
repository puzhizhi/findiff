"""
findiff is a Python package for finite difference numerical derivatives
and partial differential equations in any number of dimensions.

Classes and functions that are meant to be used by the package users
are automatically imported into 'findiff' namespace.

Classes and functions that are not imported into 'findiff' namespace
are internal and should only be used for developing the findiff package
itself, but not by package users.
"""

__version__ = '1.0.0'
__deprecation_warning__ = True

import findiff.legacy as legacy
import findiff.api as api

from .api import Diff, matrix_repr, stencils_repr
from .core.grids import EquidistantGrid, Spacing
from .core.stencils import Stencil
from .core.exceptions import InvalidGrid, InvalidArraySize
from .symbolics import Equation

from .core import DEFAULT_ACCURACY

# Legacy, yields deprecation warning:
from .legacy import FinDiff, Identity, coefficients
from .legacy import Gradient, Divergence, Laplacian, Curl
from .legacy import PDE, BoundaryConditions

from .conflicts import Coef, Identity


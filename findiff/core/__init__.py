"""
Subpackage findiff.core

Classes in this subpackage are not meant for direct user interaction. The
user shall use the names exported from findiff.api. All classes in findiff.core
do only superficial input validation, if at all.

Classes in findiff.core shall be as isolated as possible. So it shall not know
anything about other subpackages like symbolics or legacy.
"""

DEFAULT_ACCURACY = 2

from .deriv import PartialDerivative
from .algebraic import Add, Mul, Numberlike, Coordinate
from .grids import Spacing, EquidistantGrid
from .reprs import matrix_repr, stencils_repr
from .stencils import StandardStencilFactory, StencilFactory
from .stencils import Stencil, SymmetricStencil, ForwardStencil, BackwardStencil
from .stencils import StencilSet, TrivialStencilSet, StandardStencilSet
from .stencils import StencilStore
from .exceptions import *

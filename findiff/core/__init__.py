"""
Subpackage findiff.core

Classes in this subpackage are not meant for direct user interaction. The
user shall use the names exported from findiff.api. All classes in findiff.core
do only superficial input validation, if at all.
"""

from .stencils import Stencil
from .deriv import InvalidGrid, InvalidArraySize, FinDiffException

import sys
import types
from functools import wraps

import sympy

Symbol = sympy.Symbol
IndexedBase = sympy.IndexedBase
Rational = sympy.Rational


names = [
 'acos',
 'acosh',
 'acot',
 'acoth',
 'acsc',
 'acsch',
 'apart',
 'asec',
 'asech',
 'asin',
 'asinh',
 'atan',
 'atan2',
 'atanh',
 'collect',
 'conjugate',
 'cos',
 'cosh',
 'cot',
 'coth',
 'csc',
 'csch',
 'diff',
 'erf',
 'erf2',
 'erf2inv',
 'erfc',
 'erfcinv',
 'erfi',
 'erfinv',
 'exp',
 'exp_polar',
 'factor',
 'latex',
 'log',
 'loggamma',
 'sec',
 'sech',
 'simplify',
 'sin',
 'sinc',
 'sinh',
 'sqrt',
 'tan',
 'tanh',
]

acos = None
acosh = None
acot = None
acoth = None
acsc = None
acsch = None
apart = None
asec = None
asech = None
asin = None
asinh = None
atan = None
atan2 = None
atanh = None
collect = None
conjugate = None
cos = None
cosh = None
cot = None
coth = None
csc = None
csch = None
diff = None
erf = None
erf2 = None
erf2inv = None
erfc = None
erfcinv = None
erfi = None
erfinv = None
exp = None
exp_polar = None
factor = None
latex = None
log = None
loggamma = None
sec = None
sech = None
simplify = None
sin = None
sinc = None
sinh = None
sqrt = None
tan = None
tanh = None

class Equation(sympy.Eq):
    """A more convenient version of SymPy's Equality class."""

    def solve(self, target):
        """Solve the equation for a given target term."""
        sols = sympy.solve(self, target)
        eqs = []
        for sol in sols:
            eq = Equation(target, sol)
            eqs.append(eq)
        if len(eqs) == 1:
            return eqs[0]
        return eqs

    def swap(self):
        return Equation(self.rhs, self.lhs)

    def as_subs(self, key_side='lhs'):
        if key_side == 'lhs':
            return {self.lhs: self.rhs}
        elif key_side == 'rhs':
            return {self.rhs: self.lhs}
        raise KeyError('key_side: only lhs or rhs allowed.')

    def __add__(self, other):
        if isinstance(other, Equation):
            return Equation(self.lhs + other.lhs, self.rhs + other.rhs)
        return Equation(self.lhs + other, self.rhs + other)

    def __radd__(self, other):
        return Equation(self.lhs + other, self.rhs + other)

    def __sub__(self, other):
        if isinstance(other, Equation):
            return Equation(self.lhs - other.lhs, self.rhs - other.rhs)
        return Equation(self.lhs - other, self.rhs - other)

    def __rsub__(self, other):
        return Equation(other - self.lhs, other - self.rhs)

    def __mul__(self, other):
        if isinstance(other, Equation):
            return Equation(self.lhs * other.lhs, self.rhs * other.rhs)
        return Equation(self.lhs * other, self.rhs * other)

    def __rmul__(self, other):
        return Equation(other * self.lhs, other * self.rhs)

    def __truediv__(self, other):
        if isinstance(other, Equation):
            return Equation(self.lhs / other.lhs, self.rhs / other.rhs)
        return Equation(self.lhs / other, self.rhs / other)

    def __rtruediv__(self, other):
        return Equation(other / self.lhs, other / self.rhs)

    def __pow__(self, power, modulo=None):
        return Equation(self.lhs**power, self.rhs**power)


def _wrap_function(func):
    @wraps(func)
    def f(*args, **kwargs):
        """
        Patch sympy function so it handles ``Eq`` as first argument correctly
        by broadcasting the `func` action to both ``eq.lhs`` and ``eq.rhs``.
        Functions `solve`, `nsolve`, etc. are handled differently, convering the
        equation object into an expression equal to zero: ``eq.lhs - eq.rhs``,
        which is the expected input for "solve" functions in SymPy.
        """
        if not args:
            return func(*args, **kwargs)
        if isinstance(args[0], Equation):
            eq = args[0]
            other_args = args[1:]
            if func.__name__ in ['solve', 'nsolve', 'dsolve']:
                return func(eq.lhs - eq.rhs, *other_args, **kwargs)
            else:
                return Equation(func(eq.lhs, *other_args, **kwargs),
                          func(eq.rhs, *other_args, **kwargs))
        else:
            return func(*args, **kwargs)

    return f

this_module = sys.modules[__name__]

for name in names:
    func = getattr(sympy, name)
    if (isinstance(func, types.FunctionType)
            or isinstance(func, sympy.FunctionClass)):
        setattr(this_module, name, _wrap_function(func))


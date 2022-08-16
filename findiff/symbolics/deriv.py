from sympy import Symbol, IndexedBase, Expr, sympify, Basic, Integer
from sympy.core.decorators import _sympifyit, call_highest_priority


class DerivativeExpr(Expr):

    _op_priority = 11.0

    is_commutative = True
    is_number = False
    is_symbol = False
    is_scalar = False

    @property
    def _mul_handler(self):
        return self.__mul__

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if isinstance(other, DerivativeExpr):
            if self.axis > other.axis:
                return Expr.__mul__(other, self)
            elif self.axis < other.axis:
                return Expr.__mul__(self, other)
            else:
                return DerivativeSymbol(self.axis, self.degree + other.degree)
        return super().__mul__(other)

    def __pow__(self, power):
        if isinstance(power, int) or isinstance(power, Integer) and power > 0:
            return DerivativeSymbol(self.axis, self.degree * power)
        raise ValueError('Can only raise derivative operators to positive integer powers.')


class DerivativeSymbol(DerivativeExpr):

    is_commutative = True
    is_symbol = True

    def __new__(cls, axis, degree=1):
        obj = Basic.__new__(cls, Integer(axis), Integer(degree))
        obj.axis = axis
        obj.degree = degree
        return obj

    def __str__(self):
        if self.degree == 1:
            return r'\partial_{%d}' % (self.axis)
        else:
            return r'\partial_{%d}^{%d}' % (self.axis, self.degree)

    def __repr__(self):
        return str(self)

    def _sympystr(self, printer):
        return printer._print(str(self))

    def _latex(self, printer):
        return printer._print(str(self))
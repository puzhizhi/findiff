import sympy


class Equation(sympy.Eq):
    """A more convenient version of SymPy's class 'Eq'."""

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

    def as_subs(self, key_side='lhs'):
        if key_side == 'lhs':
            return {self.lhs: self.rhs}
        elif key_side == 'rhs':
            return {self.rhs: self.lhs}
        raise KeyError('key_side: only lhs or rhs allowed.')


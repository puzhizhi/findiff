import unittest
from sympy import symbols

from findiff import Equation
from findiff.symbolics import Symbol, Rational


class TestEquation(unittest.TestCase):

    def test_solve_equation_with_single_solution(self):
        a, b, c, d = symbols('a b c d')
        eq = Equation(a+b, c+d)
        actual = eq.solve(c)
        expected = Equation(c, a+b-d)
        self.assertEqual(expected, actual)

    def test_solve_equation_with_two_solutions(self):
        x = symbols('x')
        eq = Equation(x**2 -1, 0)
        actual = eq.solve(x)
        self.assertEqual(2, len(actual))
        self.assertEqual(Equation(x, -1), actual[0])
        self.assertEqual(Equation(x, 1), actual[1])

    def test_as_subs_with_default(self):
        x = symbols('x')
        eq = Equation(x, 1)
        self.assertEqual({x: 1}, eq.as_subs())

    def test_as_subs_with_other_side_as_key(self):
        x = symbols('x')
        eq = Equation(x, 1)
        self.assertEqual({1: x}, eq.as_subs(key_side='rhs'))

    def test_as_subs_key_error(self):
        x = symbols('x')
        eq = Equation(x, 1)
        with self.assertRaises(KeyError):
            eq.as_subs(key_side='bla')

    def test_multiply_equation(self):
        x = Symbol('x')
        eq = Equation(x, 1)
        actual = eq * 2
        self.assertEqual(Equation(2*x, 2), actual)

    def test_rmultiply_equation(self):
        x = Symbol('x')
        eq = Equation(x, 1)
        actual = 2 * eq
        self.assertEqual(Equation(2*x, 2), actual)

    def test_multiply_equations(self):
        x = Symbol('x')
        eq = Equation(x, 2)
        actual = eq * eq
        self.assertEqual(Equation(x**2, 4), actual)

    def test_add_equation(self):
        x = Symbol('x')
        eq = Equation(x, 1)
        actual = eq + 2
        self.assertEqual(Equation(x+2, 3), actual)

    def test_radd_equation(self):
        x = Symbol('x')
        eq = Equation(x, 1)
        actual = 2 + eq
        self.assertEqual(Equation(2+x, 3), actual)

    def test_add_equations(self):
        x = Symbol('x')
        eq = Equation(x, 2)
        actual = eq + eq
        self.assertEqual(Equation(2*x, 4), actual)

    def test_sub_equation(self):
        x = Symbol('x')
        eq = Equation(x, 1)
        actual = eq - 2
        self.assertEqual(Equation(x-2, -1), actual)

    def test_rsub_equation(self):
        x = Symbol('x')
        eq = Equation(x, 1)
        actual = 2 - eq
        self.assertEqual(Equation(2-x, 1), actual)

    def test_sub_equations(self):
        x = Symbol('x')
        eq = Equation(x, 2)
        actual = eq - 2*eq
        self.assertEqual(Equation(-x, -2), actual)

    def test_div_equation(self):
        x = Symbol('x')
        eq = Equation(x, 1)
        actual = eq / x
        self.assertEqual(Equation(1, 1/x), actual)

    def test_rdiv_equation(self):
        x = Symbol('x')
        eq = Equation(x, 1)
        actual = x / eq
        self.assertEqual(Equation(1, x), actual)

    def test_div_equations(self):
        x = Symbol('x')
        y = Symbol('y')
        eq = Equation(x, 1)
        eq2 = Equation(y, 2)
        actual = eq / eq2
        self.assertEqual(Equation(x/y, Rational(1, 2)), actual)

    def test_apply_sqrt(self):
        from findiff.symbolics import sqrt
        x = Symbol('x', positive=True)
        eq = Equation(x**2, 4)
        actual = sqrt(eq)
        self.assertEqual(Equation(x, 2), actual)

    def test_swap(self):
        x = Symbol('x')
        eq = Equation(x, 2)
        self.assertEqual(Equation(2, x), eq.swap())

    def test_pow(self):
        x = Symbol('x')
        eq = Equation(x, 2)
        self.assertEqual(Equation(x**2, 4), eq**2)


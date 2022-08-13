import unittest
from sympy import symbols

from findiff import Equation


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

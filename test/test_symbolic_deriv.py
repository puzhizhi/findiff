import unittest

from sympy import latex, expand, srepr, Add, Mul, simplify

from findiff.symbolics.deriv import DerivativeSymbol as D


class TestSymbolicDerivative(unittest.TestCase):

    def test_init_d_dx(self):
        d = D(0)
        self.assertEqual(0, d.axis)
        self.assertEqual(1, d.degree)
        self.assertEqual(r'\partial_{0}', str(d))

    def test_init_d_dy(self):
        d = D(1)
        self.assertEqual(1, d.axis)
        self.assertEqual(1, d.degree)
        self.assertEqual(r'\partial_{1}', str(d))

    def test_init_d2_dy2(self):
        d = D(1, 2)
        self.assertEqual(1, d.axis)
        self.assertEqual(2, d.degree)
        self.assertEqual(r'\partial_{1}^{2}', str(d))

    def test_init_d2_dy2_plus_one(self):
        expr = D(1, 2) + 1
        self.assertEqual(Add, type(expr))
        self.assertEqual(r'\partial_{1}^{2} + 1', str(expr))

    def test_init_d2_dxdy(self):
        d = D(0) * D(1, 2)
        self.assertEqual(Mul, type(d))
        self.assertEqual(d.args[0].axis, 0)
        self.assertEqual(d.args[0].degree, 1)
        self.assertEqual(d.args[1].axis, 1)
        self.assertEqual(d.args[1].degree, 2)

    def test_init_d2_dxdy_should_use_normal_order(self):
        d = D(1, 2) * D(0)
        self.assertEqual(Mul, type(d))
        self.assertEqual(d.args[0].axis, 0)
        self.assertEqual(d.args[0].degree, 1)
        self.assertEqual(d.args[1].axis, 1)
        self.assertEqual(d.args[1].degree, 2)

    def test_init_d2_dxdx_should_merge_degree(self):
        d = D(0) * D(0)
        self.assertEqual(2, d.degree)
        self.assertEqual(0, d.axis)

    def test_init_d_dx_should_commute_with_number(self):
        d = D(0) * 2
        self.assertEqual(2, d.args[0])
        self.assertEqual(0, d.args[1].axis)
        self.assertEqual(1, d.args[1].degree)

    def test_init_d_dx_plus_one_squared_can_expand(self):
        d = expand((D(0) + 1)**2).simplify()
        self.assertEqual(Add, type(d))
        self.assertEqual(r'(1, 2*\partial_{0}, \partial_{0}^{2})', str(d.args))

    def test_sum_of_products(self):
        d = D(0)*D(1) + D(1)*D(0)
        expected = 2*D(0)*D(1)
        self.assertEqual(str(expected), str(d))

    def test_binomial_expanded_combines_with_simplify(self):
        d =(D(0) - D(1)) * (D(0) + D(1))
        actual = d.expand().simplify()

        expected = D(0, 2) - D(1, 2)
        self.assertEqual(str(expected), str(actual))


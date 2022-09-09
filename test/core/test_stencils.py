import unittest
from itertools import product

import numpy as np
from numpy.testing import assert_array_almost_equal
from sympy import Rational, Symbol, simplify

from findiff import FinDiff
from findiff.conflicts import Identity
from findiff import Stencil
from findiff.core.stencils import Stencil1D, SymmetricStencil1D, ForwardStencil1D, BackwardStencil1D
from findiff.symbolics.deriv import DerivativeSymbol


class TestStencilOperations(unittest.TestCase):

    def test_solve_laplace_2d_with_5_points(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1})

        expected = {
            (0, 0): -4,
            (-1, 0): 1, (1, 0): 1, (0, 1): 1, (0, -1): 1
        }

        self.assertEqual(expected, stencil.values)
        self.assertEqual(2, stencil.accuracy)

    def test_solve_laplace_2d_with_9_points(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1), (-2, 0), (2, 0), (0, -2), (0, 2)]

        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1})

        expected = {
            (0, 0): -5,
            (-1, 0): 4 / 3., (1, 0): 4 / 3., (0, 1): 4 / 3., (0, -1): 4 / 3.,
            (-2, 0): -1 / 12., (2, 0): -1 / 12., (0, -2): -1 / 12., (0, 2): -1 / 12.
        }

        self.assertEqual(4, stencil.accuracy)
        self.assertEqual(len(expected), len(stencil.values))
        for off, coeff in stencil.values.items():
            self.assertAlmostEqual(coeff, expected[off])

    def test_solve_laplace_2d_with_5_points_times_2(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        stencil = Stencil(offsets, {(2, 0): 2, (0, 2): 2})

        expected = {
            (0, 0): -8,
            (-1, 0): 2, (1, 0): 2, (0, 1): 2, (0, -1): 2
        }

        self.assertEqual(expected, stencil.values)
        self.assertEqual(2, stencil.accuracy)

    def test_solve_laplace_2d_with_5_points_times_2_and_spacing(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        stencil = Stencil(offsets, {(2, 0): 2, (0, 2): 2}, spacings=(0.1, 0.1))

        expected = {
            (0, 0): -800,
            (-1, 0): 200, (1, 0): 200, (0, 1): 200, (0, -1): 200
        }

        self.assertEqual(len(expected), len(stencil.values))
        for off, coeff in stencil.values.items():
            self.assertAlmostEqual(coeff, expected[off])
        self.assertEqual(2, stencil.accuracy)

    def test_apply_laplacian_laplacian(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X**3 + Y**3
        expected =  6*X + 6*Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))
        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual)

    def test_apply_laplacian_laplacian_stencil_x_form(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X**3 + Y**3
        expected =  6*X + 6*Y

        offsets = [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))
        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual)

    def test_apply_laplacian_laplacian_stencil_outside_grid(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X**3 + Y**3

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))
        at = (0, 1)
        with self.assertRaises(Exception):
            stencil(f, at)

        at = (3, 20)
        with self.assertRaises(Exception):
            stencil(f, at)

    def test_apply_mixed_deriv(self):

        x = y = np.linspace(0, 1, 101)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = np.exp(-X**2-Y**2)
        expected =  4*X*Y*f

        offsets = [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        stencil = Stencil(offsets, partials={(1, 1): 1}, spacings=(dx, dy))
        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual, places=5)

    def test_laplace_1d_9points(self):
        x = np.linspace(0, 1, 101)
        f = x**3
        expected = 6*x
        offsets = list(range(-4, 5))
        stencil = Stencil(offsets, partials={(2,): 1}, spacings=(x[1]-x[0],))
        at = 8,
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual, places=5)

    def tests_apply_stencil_on_multislice(self):
        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X ** 3 + Y ** 3
        expected = 6*X + 6*Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))

        on_slice = slice(1, -1), slice(1, -1)
        actual = stencil(f, on=on_slice)
        np.testing.assert_array_almost_equal(expected[on_slice], actual[on_slice])

    def tests_apply_stencil_on_mask(self):
        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X ** 3 + Y ** 3
        expected = 6*X + 6*Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))

        mask = np.full_like(f, fill_value=False, dtype=bool)
        mask[1:-1, 1:-1] = True
        actual = stencil(f, on=mask)
        np.testing.assert_array_almost_equal(expected[mask], actual[mask])

    def test_helmholtz_stencil_issue_60(self):
        # This is a regression test for issue #60.

        H = Identity() - FinDiff(0, 1, 2)

        stencil_set = H.stencil((10,))

        expected = {('L',): {(0,): -1.0, (1,): 5.0, (2,): -4.0, (3,): 1.0}, ('C',): {(-1,): -1.0, (0,): 3.0, (1,): -1.0},
         ('H',): {(-3,): 1.0, (-2,): -4.0, (-1,): 5.0, (0,): -1.0}}

        actual = stencil_set.data
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(expected.keys(), actual.keys())
        for key, expected_stencil in expected.items():
            actual_stencil = actual[key]

            self.assertEqual(expected_stencil.keys(), actual_stencil.keys())
            for offset, expected_coef in expected_stencil.items():
                actual_coef = actual_stencil[offset]
                self.assertAlmostEqual(expected_coef, actual_coef)

    def test_stencil_1d_symbolic(self):
        offsets = [0, 1, 2]
        stencil = Stencil(offsets, {(1,): 1}, symbolic=True)
        self.assertEqual(-Rational(3, 2), stencil[0])

    def test_stencil_2d_laplacian_symbolic(self):
        offsets = [(0, 0), (2, 0), (-2, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0,2): 1}, symbolic=True)
        self.assertEqual(-Rational(5, 2), stencil[(0, 0)])

    def test_stencil_2d_laplacian_symbolic(self):
        offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0,2): 1}, spacings=[r'\Delta x', r'\Delta y'], symbolic=True)

        dx, dy = Symbol(r'\Delta x'), Symbol(r'\Delta y')

        self.assertEqual(-2/dx**2 - 2/dy**2, simplify(stencil[0, 0]))

    def test_stencil_2d_laplacian_symbolic_with_symbol_spacings(self):
        offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = Symbol(r'\Delta x'), Symbol(r'\Delta y')
        stencil = Stencil(offsets, {(2, 0): 1, (0,2): 1}, spacings=[dx, dy], symbolic=True)
        self.assertEqual(-2/dx**2 - 2/dy**2, simplify(stencil[0, 0]))

    def test_stencil_2d_laplacian_symbolic_with_symbol_spacing(self):
        offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        dx = Symbol(r'\Delta')
        stencil = Stencil(offsets, {(2, 0): 1, (0,2): 1}, spacings=dx, symbolic=True)
        self.assertEqual(-4/dx**2, simplify(stencil[0, 0]))

    def test_stencil_2d_laplacian_symbolic_with_symbol_spacing_as_expression(self):
        offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        dx = Symbol(r'\Delta')
        stencil = Stencil(offsets, {(2, 0): 1, (0,2): 1}, spacings=dx, symbolic=True)
        expr, symbols = stencil.as_expression()
        self.assertEqual(str(expr), r'u[i_0 + 1, i_1]/\Delta**2 + u[i_0 - 1, i_1]/\Delta**2 + u[i_0, i_1 + 1]/\Delta**2 + u[i_0, i_1 - 1]/\Delta**2 - 4*u[i_0, i_1]/\Delta**2')
        self.assertEqual(symbols['spacings'], [dx]*2)

    def test_stencil_2d_laplacian_symbolic_as_expression_with_symbols(self):
        offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        dx = Symbol(r'\Delta')
        stencil = Stencil(offsets, {(2, 0): 1, (0,2): 1}, spacings=dx, symbolic=True)
        actual, _ = stencil.as_expression('u', 'ij')
        self.assertEqual(str(actual), r'u[i + 1, j]/\Delta**2 + u[i - 1, j]/\Delta**2 + u[i, j + 1]/\Delta**2 + u[i, j - 1]/\Delta**2 - 4*u[i, j]/\Delta**2')

    def test_discretize_helmholtz_1d(self):
        stencil = Stencil(offsets=[-1, 0, 1], partials={(2,): 1}, spacings=[r'\Delta'], symbolic=True)
        d2_dx2, symbols = stencil.as_expression(index_symbols=['n'])
        n, = symbols['indices']
        u = symbols['function']
        helmholtz = d2_dx2 - u[n]
        self.assertEqual(r'-u[n] + u[n + 1]/\Delta**2 + u[n - 1]/\Delta**2 - 2*u[n]/\Delta**2', str(helmholtz))

    def test_apply_stencil_should_fail_in_symbolic_mode(self):
        stencil = Stencil(offsets=[-1, 0, 1], partials={(2,): 1}, spacings=[r'\Delta'], symbolic=True)
        with self.assertRaises(NotImplementedError) as e:
            stencil(at=1)
            self.assertEqual('NotImplementedError: __call__ cannot be used in symbolic mode.', str(e))

    def test_stencil_using_derivativesymbol(self):
        D = DerivativeSymbol
        d = D(0, 2)
        stencil = Stencil(offsets=[-1, 0, 1], partials=d, symbolic=True)
        self.assertEqual(
            Stencil(offsets=[-1, 0, 1], partials={(2,): 1}, symbolic=True).values,
            stencil.values
        )

    def test_stencil_using_derivativesymbol_2d_laplace(self):
        D = DerivativeSymbol
        d = D(0, 2) + D(1, 2)
        stencil = Stencil(offsets=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], partials=d, symbolic=True)
        self.assertEqual(
            Stencil(offsets=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], partials={(2, 0): 1, (0, 2): 1}, symbolic=True).values,
            stencil.values
        )

    def test_stencil_using_derivativesymbol_2d_with_constant_factors(self):
        D = DerivativeSymbol
        d = D(0, 2) - 2 * D(1, 2)
        stencil = Stencil(offsets=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], partials=d, symbolic=True)
        self.assertEqual(
            Stencil(offsets=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], partials={(2, 0): 1, (0, 2): -2}, symbolic=True).values,
            stencil.values
        )

    def test_stencil_using_derivativesymbol_2d_with_constant_minus_one(self):
        D = DerivativeSymbol
        d = D(0, 2) - D(1, 2)
        stencil = Stencil(offsets=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], partials=d, symbolic=True)
        self.assertEqual(
            Stencil(offsets=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], partials={(2, 0): 1, (0, 2): -1}, symbolic=True).values,
            stencil.values
        )

    def test_binomial_expanded_without_simplify_does_not_combine(self):
        D = DerivativeSymbol
        d = (D(0) - D(1)) * (D(0) + D(1))
        actual = d.expand()

        with self.assertRaises(ValueError):
            stencil = Stencil(offsets=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], partials=d, symbolic=True)

    def test_apply_stencil_mixed(self):
        x = y = z = np.linspace(1-0.1, 1+0.1, 11)
        dx = dy = dz = x[1] - x[0]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        f = X**3*Y**2*Z
        offsets = list(product([-1, 0, 1], repeat=3))
        d3_dx2dy = Stencil(offsets, {(2, 1, 0): 1}, spacings=[dx, dy, dz])
        expected = 12.  # f(1,1,1)
        actual = d3_dx2dy(f, at=(5, 5, 5))
        self.assertAlmostEqual(expected, actual)

        d3_dx2dy_sym = Stencil(offsets, {(2, 1, 0): 1}, spacings=[dx, dy, dz], symbolic=True)

    def test_wave_equation_stencil(self):
        D = DerivativeSymbol
        dt = Symbol(r'\Delta t', real=True)
        dx = Symbol(r'\Delta x', real=True)
        d2_dx2 = Stencil(offsets=[(0, 1), (0, 0), (0, -1)], partials=D(1, 2), spacings=(dt, dx), symbolic=True)


class TestStencil1D(unittest.TestCase):

    def test_stencil1d(self):
        s = Stencil1D(2, [-1, 0, 1], 1)
        print(repr(s))

    def test_stencil1d(self):
        s = Stencil1D(2, [-1, 0, 1], 1)
        assert {-1: 1, 0: -2, 1: 1} == s.data

    def test_symmetricstencil1d(self):
        s = SymmetricStencil1D(2, 1, 2)
        assert {-1: 1, 0: -2, 1: 1} == s.data

    def test_forwardstencil1d(self):
        s = ForwardStencil1D(2, 1, 2)
        assert_array_almost_equal([2, -5, 4, -1], s.coefs)

        s = ForwardStencil1D(1, 1, 2)
        assert_array_almost_equal([-1.5, 2., -0.5], s.coefs)

    def test_forwardstencil1d(self):
        s = BackwardStencil1D(2, 1, 2)
        assert_array_almost_equal([-1, 4, -5, 2], s.coefs)

        s = BackwardStencil1D(1, 1, 2)
        assert_array_almost_equal([0.5, -2, 1.5], s.coefs)

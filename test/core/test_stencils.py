import unittest
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from sympy import Rational, Symbol, simplify

from findiff.core import PartialDerivative, BackwardStencil
from findiff.core import Stencil, StandardStencilFactory, SymmetricStencil, StandardStencilSet, StencilFactory
from findiff.core import Spacing

# Useful for debugging printouts of arrays
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3f" % x))


class UnitTestsStencil(unittest.TestCase):

    def test_create_stencil_from_constructor_stores_parameters(self):
        offsets = [(-1,), (0,), (1,)]
        coefs = [1., -2., 1.]
        stencil = Stencil(offsets, coefs)
        self.assertEqual(offsets, stencil.offsets)
        self.assertEqual(coefs, stencil.coefficients)

    def test_stencils_can_be_added(self):
        stencil_1 = Stencil({(0, -1): 1., (0, 0): -2., (0, 1): 1.})
        stencil_2 = Stencil({(0, -2): 1., (0, 0): -1., (0, 1): 1.})
        result = stencil_1 + stencil_2
        self.assertEqual(
            {(0, -2): 1, (0, -1): 1., (0, 0): -3., (0, 1): 2.},
            result.as_dict()
        )

    def test_stencil_can_be_applied(self):
        x = np.linspace(0, 1, 100)
        f = x ** 3

        factory = StandardStencilFactory()
        acc = 2
        dx = x[1] - x[0]
        deriv = 2
        stencil = factory.create(SymmetricStencil, deriv, dx, acc)

        mask = np.full_like(x, fill_value=False, dtype=bool)
        mask[1:-1] = True

        actual = stencil.apply(f, mask)
        assert_allclose(actual[1:-1], 6*x[1:-1], rtol=1.E-6)


class IntegrationTestsStencil(unittest.TestCase):
    ...


class UnitTestsStencilFactory(unittest.TestCase):
    ...


class IntegrationTestsStencilFactory(unittest.TestCase):

    def test_create_symmetric_stencil(self):
        factory = StandardStencilFactory()
        deriv = 2
        acc = 2
        spacing = 1

        stencil = factory.create(SymmetricStencil, deriv, spacing, acc)

        self.assertTrue(SymmetricStencil, type(stencil))

    def test_create_symmetric_stencil_correct_coefs(self):
        factory = StandardStencilFactory()
        stencil = factory.create(SymmetricStencil, deriv=2, spacing=1, acc=2)

        self.assertEqual({(0,): -2, (-1,): 1, (1,): 1}, stencil.as_dict())


class UnitTestsFlexStencilFactory(unittest.TestCase):
    ...


class IntegrationTestsFlexStencilFactory(unittest.TestCase):

    def test_flexstencilfactory_creates_generic_stencil_1d(self):
        factory = StencilFactory()
        offsets = [(0,), (-1,), (1,)]
        pds = {(2,): 1}  # this is 1 * d2_dx2
        spacing = Spacing({0: 1})
        stencil = factory.create(offsets, pds, spacing)
        self.assertEqual({(0,): -2, (-1,): 1, (1,): 1}, stencil.as_dict())

    def test_solve_laplace_2d_with_5_points(self):

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0, 2): 1}, spacing=Spacing({0: 1, 1: 1}))

        expected = {
            (0, 0): -4,
            (-1, 0): 1, (1, 0): 1, (0, 1): 1, (0, -1): 1
        }

        self.assertEqual(expected, stencil.as_dict())
        self.assertEqual(2, stencil.accuracy)

    def test_solve_laplace_2d_with_9_points(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1), (-2, 0), (2, 0), (0, -2), (0, 2)]

        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0, 2): 1}, spacing=Spacing({0: 1, 1: 1}))

        expected = {
            (0, 0): -5,
            (-1, 0): 4 / 3., (1, 0): 4 / 3., (0, 1): 4 / 3., (0, -1): 4 / 3.,
            (-2, 0): -1 / 12., (2, 0): -1 / 12., (0, -2): -1 / 12., (0, 2): -1 / 12.
        }

        self.assertEqual(4, stencil.accuracy)
        self.assertEqual(len(expected), len(stencil.as_dict()))
        for off, coeff in stencil.as_dict().items():
            self.assertAlmostEqual(coeff, expected[off])

    def test_solve_laplace_2d_with_5_points_times_2(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 2, (0, 2): 2}, spacing=Spacing({0: 1, 1: 1}))

        expected = {
            (0, 0): -8,
            (-1, 0): 2, (1, 0): 2, (0, 1): 2, (0, -1): 2
        }

        self.assertEqual(expected, stencil.as_dict())
        self.assertEqual(2, stencil.accuracy)

    def test_solve_laplace_2d_with_5_points_times_2_and_spacing(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 2, (0, 2): 2}, spacing=Spacing({0: 0.1, 1: 0.1}))

        expected = {
            (0, 0): -800,
            (-1, 0): 200, (1, 0): 200, (0, 1): 200, (0, -1): 200
        }

        self.assertEqual(len(expected), len(stencil.as_dict()))
        for off, coeff in stencil.as_dict().items():
            self.assertAlmostEqual(coeff, expected[off])
        self.assertEqual(2, stencil.accuracy)

    def test_apply_laplacian_laplacian(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X**3 + Y**3
        expected =  6*X + 6*Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0, 2): 1}, spacing=Spacing({0: dx, 1: dy}))

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
        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0, 2): 1}, spacing=Spacing({0: dx, 1: dy}))

        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual)

    def test_apply_laplacian_laplacian_stencil_outside_grid(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X**3 + Y**3

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0, 2): 1}, spacing=Spacing({0: dx, 1: dy}))

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
        factory = StencilFactory()
        stencil = factory.create(offsets, {(1, 1): 1}, spacing=Spacing({0: dx, 1: dy}))

        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual, places=5)

    def test_laplace_1d_9points(self):
        x = np.linspace(0, 1, 101)
        f = x**3
        expected = 6*x
        offsets = list(range(-4, 5))
        factory = StencilFactory()
        stencil = factory.create(offsets, {(2,): 1}, spacing=Spacing({0: x[1]-x[0]}))

        at = 8,
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual, places=5)

    def tests_apply_stencil_on_mask(self):
        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X ** 3 + Y ** 3
        expected = 6*X + 6*Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0, 2): 1}, spacing=Spacing({0: dx, 1: dy}))

        mask = np.full_like(f, fill_value=False, dtype=bool)
        mask[1:-1, 1:-1] = True
        actual = stencil(f, on=mask)
        np.testing.assert_array_almost_equal(expected[mask], actual[mask])

    def test_stencil_1d_symbolic(self):
        offsets = [0, 1, 2]
        factory = StencilFactory()
        stencil = factory.create(offsets, {(1,): 1}, spacing=Spacing({0: 1}), symbolic=True)

        self.assertEqual(-Rational(3, 2), stencil[0])

    def test_stencil_2d_laplacian_symbolic(self):
        offsets = [(0, 0), (2, 0), (-2, 0), (0, 1), (0, -1)]
        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0,2): 1}, spacing=Spacing({0: 1, 1: 1}), symbolic=True)

        self.assertEqual(-Rational(5, 2), stencil[(0, 0)])

    def test_stencil_2d_laplacian_symbolic_2(self):
        offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0, 2): 1}, spacing=Spacing({0: r'\Delta x', 1: r'\Delta y'}), symbolic=True)

        dx, dy = Symbol(r'\Delta x'), Symbol(r'\Delta y')

        self.assertEqual(-2/dx**2 - 2/dy**2, simplify(stencil[0, 0]))

    def test_stencil_2d_laplacian_symbolic_with_symbol_spacings(self):
        offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = Symbol(r'\Delta x'), Symbol(r'\Delta y')
        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0, 2): 1}, spacing=Spacing({0: dx, 1: dy}),
                                 symbolic=True)

        self.assertEqual(-2/dx**2 - 2/dy**2, simplify(stencil[0, 0]))

    def test_stencil_2d_laplacian_symbolic_with_symbol_spacing(self):
        offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        dx = Symbol(r'\Delta')
        factory = StencilFactory()
        stencil = factory.create(offsets, {(2, 0): 1, (0, 2): 1}, spacing=Spacing(dx),
                                 symbolic=True)

        self.assertEqual(-4/dx**2, simplify(stencil[0, 0]))

    def test_symbolic_stencil(self):
        offsets = list(product([-1, 0, 1], repeat=3))
        factory = StencilFactory()
        d3_dx2dy_sym = factory.create(offsets, {(2, 1, 0): 1}, spacing=Spacing(1),
                                 symbolic=True)

        assert d3_dx2dy_sym[(-1, -1, 0)] == -Rational(1, 2)


class UnitTestsStencilSet(unittest.TestCase):
    ...


class IntegrationTestsStencilSet(unittest.TestCase):

    def test_create_stencilset_1d(self):
        pd = PartialDerivative({0: 2})
        spacing = Spacing({0: 1})
        ndims = 1
        acc = 2
        stencil_set = StandardStencilSet(pd, spacing, ndims, acc)

        stencil = stencil_set[('C',)]
        self.assertEqual(
            {(0,): -2, (-1,): 1, (1,): 1},
            stencil.as_dict()
        )

        stencil = stencil_set[('L',)]
        self.assertEqual(
            {(0,): 2.0, (1,): -5.0, (2,): 4.0, (3,): -1.0},
            stencil.as_dict()
        )

        stencil = stencil_set[('H',)]
        self.assertEqual(
            {(-3,): -1.0, (-2,): 4.0, (-1,): -5.0, (0,): 2.0},
            stencil.as_dict()
        )

    def test_stencilset_can_be_applied_1d(self):
        x = np.linspace(0, 1, 100)
        dx = x[1] - x[0]
        f = x**3
        pd = PartialDerivative({0: 2})
        spacing = Spacing({0: dx})
        stencil_set = StandardStencilSet(pd, spacing, ndims=1, acc=2)

        actual = stencil_set.apply(f)

        assert_allclose(6*x, actual, atol=1E-10)

    def test_stencilset_can_be_applied_2d(self):
        x = y = np.linspace(0, 1, 10)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = X
        pd = PartialDerivative({0: 1})
        spacing = Spacing({0: dx, 1: dy})
        stencil_set = StandardStencilSet(pd, spacing, ndims=2, acc=2)

        actual = stencil_set.apply(f)

        assert_allclose(np.ones_like(f), actual, atol=1E-10)

    def test_stencilset_can_be_applied_2d_mixed(self):
        x = y = np.linspace(0, 1, 11)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = X*Y
        pd = PartialDerivative({0: 1, 1: 1})
        spacing = Spacing({0: dx, 1: dy})
        stencil_set = StandardStencilSet(pd, spacing, ndims=2, acc=2)

        actual = stencil_set.apply(f)

        assert_allclose(np.ones_like(f), actual, atol=1E-10)


class TestBackwardStencil(unittest.TestCase):

    def test_first_deriv(self):
        factory = StandardStencilFactory()

        stencil = factory.create(BackwardStencil, 1, 1, acc=2, symbolic=False)
        print(stencil)
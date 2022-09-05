import unittest

import numpy as np
import sympy
from numpy.testing import assert_array_almost_equal

from findiff import Diff
from findiff import Coef
from findiff import InvalidGrid, InvalidArraySize


class TestDiff(unittest.TestCase):

    def make_grid2d(self):
        x = y = np.linspace(0, 1, 101)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        return X, Y, dx, dy

    def test_single_first_diff(self):
        x = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        f = x ** 4
        D = Diff(0)
        actual = D(f, acc=4, spacing={0: dx})
        expected = 4 * x ** 3
        assert_array_almost_equal(expected, actual)

    def test_single_diff(self):
        x = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        f = x ** 4
        D = Diff(0, 2)
        actual = D(f, acc=4, spacing={0: dx})
        expected = 12 * x ** 2
        assert_array_almost_equal(expected, actual)

    def test_single_diff_along_axis1(self):
        X, Y, dx, dy = self.make_grid2d()
        f = X ** 3 + Y ** 3
        D = Diff(1, 2)
        actual = D(f, acc=4, spacing={1: dx})
        expected = 6 * Y
        assert_array_almost_equal(expected, actual)

    def test_mixed_diff(self):
        X, Y, dx, dy = self.make_grid2d()
        dy = dx
        f = X ** 3 * Y ** 3
        D = Diff({0: 1, 1: 2})
        actual = D(f, acc=4, spacing={0: dx, 1: dy})
        expected = 18 * X ** 2 * Y
        assert_array_almost_equal(expected, actual)

    def test_mixed_diff_default_acc2(self):
        X, Y, dx, dy = self.make_grid2d()
        dy = dx
        f = X ** 3 * Y ** 3
        D = Diff({0: 1, 1: 2})
        expected = 18 * X ** 2 * Y

        actual_acc2 = D(f, spacing={0: dx, 1: dy})
        max_err_acc2 = np.max(np.abs(expected - actual_acc2))
        print(max_err_acc2)

        actual_acc4 = D(f, acc=4, spacing={0: dx, 1: dy})
        max_err_acc4 = np.max(np.abs(expected - actual_acc4))

        assert max_err_acc2 > 1000 * max_err_acc4

    def test_apply_diff_with_single_spacing_defaults_to_same_for_all_axes(self):

        x = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        f = np.sin(x)

        # Define the derivative:
        d_dx = Diff(0, 1)


        df_dx = d_dx(f, acc=4, spacing=dx)
        assert_array_almost_equal(np.cos(x), df_dx)

    def test_diff_constructor_with_invalid_args_raises_exception(self):
        with self.assertRaises(ValueError):
            Diff(1, 2, 3)

        with self.assertRaises(ValueError):
            Diff(-1)

        with self.assertRaises(ValueError):
            Diff(1, -1)

        with self.assertRaises(ValueError):
            Diff('a')

        with self.assertRaises(ValueError):
            Diff({0: 1, 1: -2})

        with self.assertRaises(TypeError):
            Diff({0: 1, 1: -2}, acc=2)  # acc may not be used on the constructor

        with self.assertRaises(TypeError):
            Diff({0: 1, 1: -2}, 2)

    def test_applying_diff_without_spacing_raises_exception(self):
        D = Diff(1, 2)
        with self.assertRaises(InvalidGrid):
            D(np.ones((10, 10)))

    def test_applying_diff_with_invalid_nondict_spacing_raises_exception(self):
        D = Diff(1, 2)
        with self.assertRaises(InvalidGrid):
            D(np.ones((10, 10)), spacing=-0.1)

    def test_applying_diff_with_nonpositive_spacing_raises_exception(self):
        D = Diff(1, 2)
        with self.assertRaises(InvalidGrid):
            D(np.ones((10, 10)), spacing={1: -0.1})

    def test_apply_diff_to_too_small_array_raises_exception(self):
        x = np.linspace(0, 1, 101)
        D = Diff(10, 1)
        with self.assertRaises(InvalidArraySize):
            D(x**2, spacing=0.01)

    def test_single_diff_along_axis1_raises_exception_when_no_spacing_defined_along_axis(self):
        f = np.ones((10, 10))
        D = Diff(1, 2)
        with self.assertRaises(InvalidGrid):
            # should raise exception because given spacing is along axis 0, but
            # derivative is along axis 1.
            D(f, spacing={0: 0.1})

    def test_linear_combination_2d(self):
        X, Y, dx, dy = self.make_grid2d()
        f = X ** 4 + X ** 2 * Y ** 2 + Y ** 4

        D = Diff(0, 2) + 2 * Diff(0) * Diff(1) + Diff(1, 2)
        actual = D(f, acc=4, spacing={0: dx, 1: dy})

        expected = 12 * X ** 2 + 2 * X ** 2 + 2 * Y ** 2 + 8 * X * Y + 12 * Y ** 2
        assert_array_almost_equal(expected, actual)

    def test_linear_combination_2d_with_const_coefs_different_orders(self):
        X, Y, dx, dy = self.make_grid2d()
        f = X ** 4 + X ** 2 * Y ** 2 + Y ** 4

        D1 = Diff(0, 2) + 2 * Diff(0) * Diff(1) + Diff(1, 2)
        actual1 = D1(f, acc=4, spacing={0: dx, 1: dy})

        D2 = Diff(0, 2) + Diff(0) * 2 * Diff(1) + Diff(1, 2)
        actual2 = D2(f, acc=4, spacing={0: dx, 1: dy})

        D3 = Diff(0, 2) + Diff(0) * Diff(1) * 2 + Diff(1, 2)
        actual3 = D3(f, acc=4, spacing={0: dx, 1: dy})

        expected = 12 * X ** 2 + 2 * X ** 2 + 2 * Y ** 2 + 8 * X * Y + 12 * Y ** 2
        assert_array_almost_equal(expected, actual1)
        assert_array_almost_equal(expected, actual2)
        assert_array_almost_equal(expected, actual3)

    def test_linear_combination_2d_with_var_coefs(self):
        X, Y, dx, dy = self.make_grid2d()
        f = X ** 2 * Y ** 2

        D = Coef(X) * Diff(0)
        actual = D(f, spacing={0: dx, 1: dy})
        expected = 2*f
        assert_array_almost_equal(expected, actual)

    def test_linear_combination_2d_with_var_coefs_product_rule(self):
        X, Y, dx, dy = self.make_grid2d()
        f = X ** 2 * Y ** 2

        # When applying the following differential operator, the
        # product rule must be obeyed:
        D = Diff(0) * X
        actual = D(f, acc=4, spacing={0: dx, 1: dy})
        expected = 3*f
        assert_array_almost_equal(expected, actual)

        # Alternative syntax (recommended because it is symmetric)
        D = Diff(0) * Coef(X)
        actual = D(f, acc=4, spacing={0: dx, 1: dy})
        expected = 3*f
        assert_array_almost_equal(expected, actual)

    def test_chaining(self):
        X, Y, dx, dy = self.make_grid2d()
        f = X ** 4 + X ** 2 * Y ** 2 + Y ** 4

        D1 = (Diff(0) - Diff(1)) * (Diff(0) + Diff(1))
        D2 = Diff(0, 2) - Diff(1, 2)
        actual1 = D1(f, acc=4, spacing={0: dx, 1: dy})
        actual2 = D2(f, acc=4, spacing={0: dx, 1: dy})

        expected = 12 * X ** 2 + 2 * Y ** 2 - 12 * Y ** 2 - 2 * X ** 2
        assert_array_almost_equal(expected, actual1)
        assert_array_almost_equal(expected, actual2)


class TestCoefficients(unittest.TestCase):

    def test_can_be_called_with_acc_and_default_symbolics(self):
        import findiff
        out = findiff.coefficients(2, 2, symbolic=True)
        assert isinstance(out['center']['coefficients'][1], sympy.Integer)

    def test_can_be_called_with_offsets(self):
        import findiff
        out = findiff.coefficients(2, offsets=[0, 1, 2, 3], symbolic=True)
        print(out)
        some_coef = out['coefficients'][1]
        assert isinstance(some_coef, sympy.Integer)
        assert some_coef == -5

    def test_must_specify_exactly_one(self):
        import findiff
        with self.assertRaises(ValueError):
            findiff.coefficients(2, acc=2, offsets=[-1, 0, 1])


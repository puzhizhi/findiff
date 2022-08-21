import unittest

import numpy as np

from numpy.testing import assert_array_almost_equal

from findiff.api import Diff, FinDiff


class TestFinDiff(unittest.TestCase):

    def test_single_first_deriv_1d(self):
        x = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        f = x**3

        d_dx = FinDiff(0, dx, acc=4)
        actual = d_dx(f)

        assert_array_almost_equal(3*x**2, actual)

    def test_single_second_deriv_1d(self):
        x = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        f = x**3

        d_dx = FinDiff(0, dx, 2, acc=4)
        actual = d_dx(f)

        assert_array_almost_equal(6*x, actual)

    def test_single_mixed_deriv_2d(self):
        x = y = np.linspace(0, 1, 101)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = X**2*Y**2

        d2_dxdy = FinDiff((0, dx, 2), (1, dy, 2), acc=4)
        actual = d2_dxdy(f)

        assert_array_almost_equal(4, actual, decimal=5)


class TestDiff(unittest.TestCase):

    def test_single_diff(self):
        x = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        f = x**4
        D = Diff(0, 2)
        actual = D(f, acc=4, spacing={0: dx})
        expected = 12 * x**2
        assert_array_almost_equal(expected, actual)

    @unittest.skip
    def test_linear_combination_2d(self):

        x = np.linspace(0, 1, 101)
        y = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = X**4 + X**2 * Y**2 + Y**4

        #D = Diff({0: 2}) + 2 * Diff({0: 1, 1: 1}) + Diff({1: 2})
        D = Diff(0, 2) + Diff(0) * Diff(1) + Diff(1, 2) # Diff wraps continuous and discrete version
        actual = D(f, acc=4, spacing={0: dx, 1: dy})  # compute coefficients and cache them!
                            # only allow setting acc and spacing when applying => enforces consistency throughout descendants!

        expected = 12*X**2 + 2*Y**2 + 4*X*Y + 12*Y**2
        assert_array_almost_equal(expected, actual)

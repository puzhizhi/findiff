import numpy as np
import unittest

import pytest
from numpy.testing import assert_array_almost_equal

from findiff.core import Coordinate
from findiff.core import PartialDerivative
from findiff.core import EquidistantGrid
from findiff.core import Spacing


class TestsPartialDerivative(unittest.TestCase):

    def test_partial_mixed_initializes(self):
        # \frac{\partial^3}{\partial x_0 \partial x_3^2}:
        pd = PartialDerivative({0: 1, 3: 2})
        assert pd.degree(0) == 1
        assert pd.degree(1) == 0
        assert pd.degree(3) == 2

    def test_partial_invalid_input_str(self):
        with pytest.raises(AssertionError):
            PartialDerivative('abc')

    def test_partial_invalid_input_dict_float(self):
        with pytest.raises(ValueError):
            PartialDerivative({1.4: 1})

    def test_partial_invalid_input_dict_negative(self):
        with pytest.raises(ValueError):
            PartialDerivative({0: -1})

    def test_partials_add_degree_when_multiplied_on_same_axis(self):
        d2_dxdy = PartialDerivative({0: 1, 1: 1})
        d2_dxdz = PartialDerivative({0: 1, 2: 1})
        d4_dx2_dydz = d2_dxdz * d2_dxdy

        assert type(d4_dx2_dydz) == PartialDerivative
        assert d4_dx2_dydz.degree(0) == 2
        assert d4_dx2_dydz.degree(1) == 1
        assert d4_dx2_dydz.degree(2) == 1

    def test_partials_add_degree_when_powed(self):
        d2_dxdy = PartialDerivative({0: 1, 1: 1})
        actual = d2_dxdy ** 2
        assert actual.degree(0) == 2
        assert actual.degree(1) == 2

    def test_partials_are_equal(self):
        pd1 = PartialDerivative({0: 1, 1: 1})
        pd2 = PartialDerivative({1: 1, 0: 1})
        assert pd1 == pd2

    def test_partials_are_not_equal(self):
        pd1 = PartialDerivative({0: 1, 1: 2})
        pd2 = PartialDerivative({1: 1, 0: 1})
        assert not pd1 == pd2

    def test_partials_hash_correctly(self):
        d2_dx2 = PartialDerivative({0: 2})
        other_d2_dx2 = PartialDerivative({0: 2})
        assert d2_dx2 == other_d2_dx2

        a = {d2_dx2: 1}
        assert a[d2_dx2] == 1
        assert a[other_d2_dx2] == 1

    def test_empty_partial_is_identity(self):
        ident = PartialDerivative({})
        pd = PartialDerivative({0: 1})
        actual = ident * pd
        assert actual.degrees == {0: 1}

    def test_disc_part_deriv_1d(self):
        grid = EquidistantGrid((0, 1, 101))
        x = grid.coords[0]
        f = np.sin(x)
        pd = PartialDerivative({0: 1})
        df_dx = pd.apply(f, Spacing(x[1] - x[0]), 2)

        assert_array_almost_equal(np.cos(x[:1]), df_dx[:1], decimal=4)
        assert_array_almost_equal(np.cos(x[-1:]), df_dx[-1:], decimal=4)
        assert_array_almost_equal(np.cos(x[1:-1]), df_dx[1:-1], decimal=4)

    def test_disc_part_deriv_1d_acc4(self):
        grid = EquidistantGrid((0, 1, 101))
        x = grid.coords[0]
        f = np.sin(x)
        pd = PartialDerivative({0: 1})
        df_dx = pd.apply(f, Spacing(x[1] - x[0]), acc=4)

        assert_array_almost_equal(np.cos(x[:1]), df_dx[:1], decimal=6)
        assert_array_almost_equal(np.cos(x[-1:]), df_dx[-1:], decimal=6)
        assert_array_almost_equal(np.cos(x[1:-1]), df_dx[1:-1], decimal=6)

    def test_disc_part_deriv_2d_pure(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')
        x = grid.coords[0]
        f = np.sin(X) * np.sin(Y)
        pd = PartialDerivative({1: 2})

        d2f_dy2 = pd.apply(f, Spacing(x[1] - x[0]), acc=4)

        assert_array_almost_equal(-np.sin(X) * np.sin(Y), d2f_dy2)

    def test_disc_part_deriv_2d_mixed(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')
        x = grid.coords[0]
        f = np.sin(X) * np.sin(Y)
        pd = PartialDerivative({0: 1, 1: 1})

        d2f_dxdy = pd.apply(f, Spacing(x[1] - x[0]), acc=4)

        assert_array_almost_equal(np.cos(X) * np.cos(Y), d2f_dxdy)

    def test_disc_part_mul_discretize(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')
        dx = X[1, 0] - X[0, 0]
        f = np.sin(X) * np.sin(Y)
        d_dx = PartialDerivative({0: 1})

        D = 2 * d_dx

        actual = D.apply(f, Spacing(dx), acc=4)
        assert_array_almost_equal(2 * np.cos(X) * np.sin(Y), actual)

    def test_disc_part_deriv_2d_mixed_with_mul(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')
        dx = X[1, 0] - X[0, 0]
        f = np.sin(X) * np.sin(Y)
        d_dx = PartialDerivative({0: 1})
        d_dy = PartialDerivative({1: 1})

        D = d_dx * 2 * d_dy
        actual = D.apply(f, Spacing(dx), acc=4)

        assert_array_almost_equal(2 * np.cos(X) * np.cos(Y), actual)

    def test_disc_part_laplace_2d(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = X ** 4 + Y ** 4

        laplace = PartialDerivative({0: 2}) + PartialDerivative({1: 2})

        actual = laplace.apply(f, grid, acc=4)

        assert_array_almost_equal(12 * X ** 2 + 12 * Y ** 2, actual)

    def test_disc_part_with_coordinate(self):
        grid = EquidistantGrid((0, 1, 101))
        x = grid.coords[0]
        f = x ** 4

        deriv = Coordinate(0) * PartialDerivative({0: 2})

        actual = deriv.apply(f, grid, acc=4)
        assert_array_almost_equal(12 * x ** 3, actual)

    def test_disc_part_with_coordinates_2d(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = grid.meshed_coords
        f = X ** 4 + Y ** 4

        deriv = Coordinate(0) * PartialDerivative({0: 2}) + Coordinate(1) * PartialDerivative({1: 2})

        actual = deriv.apply(f, grid, acc=4)
        assert_array_almost_equal(12 * X ** 3 + 12 * Y ** 3, actual)

    def test_disc_part_with_coordinates_chaining(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = grid.meshed_coords
        f = X ** 4 + Y ** 4

        deriv1 = Coordinate(0) * PartialDerivative({0: 2}) + Coordinate(1) * PartialDerivative({1: 2})
        deriv2 = Coordinate(0) * PartialDerivative({0: 2}) + Coordinate(1) * PartialDerivative({1: 2})

        deriv = deriv1 * deriv2

        actual = deriv.apply(f, grid, acc=4)
        assert_array_almost_equal(72 * X ** 2 + 72 * Y ** 2, actual, decimal=4)

    def test_disc_part_with_coordinates_chaining_minus(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = grid.meshed_coords
        f = X ** 4 + Y ** 4

        deriv1 = Coordinate(0) * PartialDerivative({0: 2}) - Coordinate(1) * PartialDerivative({1: 2})
        deriv2 = Coordinate(0) * PartialDerivative({0: 2}) + Coordinate(1) * PartialDerivative({1: 2})

        deriv = deriv1 * deriv2

        actual = deriv.apply(f, grid, acc=4)
        assert_array_almost_equal(72 * X ** 2 - 72 * Y ** 2, actual, decimal=4)

    def test_disc_part_with_unary_minus(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = grid.meshed_coords
        f = X ** 4 + Y ** 4

        deriv = - PartialDerivative({0: 1})

        actual = deriv.apply(f, grid, acc=4)
        assert_array_almost_equal(-4 * X ** 3, actual, decimal=4)

    def test_disc_part_minus_laplace(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = grid.meshed_coords
        f = X ** 4 + Y ** 4

        laplace = PartialDerivative({0: 2}) + PartialDerivative({1: 2})

        actual = (-laplace).apply(f, grid, acc=4)
        assert_array_almost_equal(-12 * X ** 2 - 12 * Y ** 2, actual, decimal=4)

    def test_repr(self):
        pd = PartialDerivative({0: 2})
        self.assertEqual('{0: 2}', repr(pd))

    def test_str(self):
        pd = PartialDerivative({0: 2})
        self.assertEqual('{0: 2}', repr(pd))

    def test_degree_zero_raises_exception(self):
        with self.assertRaises(ValueError):
            PartialDerivative({0: 0})

import numpy as np
import unittest

import pytest
from numpy.testing import assert_array_almost_equal

from findiff.deriv import PartialDerivative
from findiff.grids import EquidistantGrid


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
        df_dx = pd.apply(f, grid, 2)

        assert_array_almost_equal(np.cos(x[:1]), df_dx[:1], decimal=4)
        assert_array_almost_equal(np.cos(x[-1:]), df_dx[-1:], decimal=4)
        assert_array_almost_equal(np.cos(x[1:-1]), df_dx[1:-1], decimal=4)

    def test_disc_part_deriv_1d_acc4(self):
        grid = EquidistantGrid((0, 1, 101))
        x = grid.coords[0]
        f = np.sin(x)
        pd = PartialDerivative({0: 1})
        df_dx = pd.apply(f, grid, acc=4)

        assert_array_almost_equal(np.cos(x[:1]), df_dx[:1], decimal=6)
        assert_array_almost_equal(np.cos(x[-1:]), df_dx[-1:], decimal=6)
        assert_array_almost_equal(np.cos(x[1:-1]), df_dx[1:-1], decimal=6)

    def test_disc_part_deriv_2d_pure(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = np.sin(X) * np.sin(Y)
        pd = PartialDerivative({1: 2})

        d2f_dy2 = pd.apply(f, grid, acc=4)

        assert_array_almost_equal(-np.sin(X) * np.sin(Y), d2f_dy2)

    def test_disc_part_deriv_2d_mixed(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = np.sin(X) * np.sin(Y)
        pd = PartialDerivative({0: 1, 1: 1})

        d2f_dxdy = pd.apply(f, grid, acc=4)

        assert_array_almost_equal(np.cos(X) * np.cos(Y), d2f_dxdy)

    def test_disc_part_mul_discretize(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = np.sin(X) * np.sin(Y)
        d_dx = PartialDerivative({0: 1})

        D = 2 * d_dx

        actual = D.apply(f, grid, acc=4)
        assert_array_almost_equal(2*np.cos(X)*np.sin(Y), actual)

    def test_disc_part_deriv_2d_mixed_with_mul(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = np.sin(X) * np.sin(Y)
        d_dx = PartialDerivative({0: 1})
        d_dy = PartialDerivative({1: 1})

        D = d_dx * 2 * d_dy
        actual = D.apply(f, grid, acc=4)

        assert_array_almost_equal(2*np.cos(X) * np.cos(Y), actual)

    def test_disc_part_laplace_2d(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = X**4 + Y**4

        laplace = PartialDerivative({0: 2}) + PartialDerivative({1: 2})

        actual = laplace.apply(f, grid, acc=4)

        assert_array_almost_equal(12*X**2 + 12*Y**2, actual)


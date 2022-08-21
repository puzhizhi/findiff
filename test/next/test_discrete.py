import numpy as np
import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from findiff.continuous import PartialDerivative, DifferentialOperator
from findiff.discrete import EquidistantGrid, Stencil1D, SymmetricStencil1D, ForwardStencil1D, BackwardStencil1D, \
    DiscretizedPartialDerivative, DiscretizedDifferentialOperator



class TestEquidistantGrid(unittest.TestCase):

    def test_equidistantgrid_gives_spacing(self):
        grid = EquidistantGrid((-1, 1, 21), (0, 1, 21))
        assert_array_almost_equal(grid.spacings, [0.1, 0.05])


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


class TestDiscretizedPartialDerivative(unittest.TestCase):

    def test_disc_part_deriv_1d(self):

        grid = EquidistantGrid((0, 1, 101))
        x = grid.coords[0]
        f = np.sin(x)
        pd = PartialDerivative({0: 1})
        d_dx = DiscretizedPartialDerivative(pd, grid, acc=2)
        df_dx = d_dx.apply(f)

        assert_array_almost_equal(np.cos(x[:1]), df_dx[:1], decimal=4)
        assert_array_almost_equal(np.cos(x[-1:]), df_dx[-1:], decimal=4)
        assert_array_almost_equal(np.cos(x[1:-1]), df_dx[1:-1], decimal=4)


    def test_disc_part_deriv_1d_acc4(self):

        grid = EquidistantGrid((0, 1, 101))
        x = grid.coords[0]
        f = np.sin(x)
        pd = PartialDerivative({0: 1})
        d_dx = DiscretizedPartialDerivative(pd, grid, acc=4)
        df_dx = d_dx.apply(f)

        assert_array_almost_equal(np.cos(x[:1]), df_dx[:1], decimal=6)
        assert_array_almost_equal(np.cos(x[-1:]), df_dx[-1:], decimal=6)
        assert_array_almost_equal(np.cos(x[1:-1]), df_dx[1:-1], decimal=6)


    def test_disc_part_deriv_2d_pure(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = np.sin(X)*np.sin(Y)
        pd = PartialDerivative({1: 2})
        d2_dy2 = DiscretizedPartialDerivative(pd, grid, acc=4)
        d2f_dy2 = d2_dy2.apply(f)

        assert_array_almost_equal(-np.sin(X)*np.sin(Y), d2f_dy2)


    def test_disc_part_deriv_2d_mixed(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = np.sin(X)*np.sin(Y)
        pd = PartialDerivative({0: 1, 1: 1})
        d2_dxdy = DiscretizedPartialDerivative(pd, grid, acc=4)
        d2f_dxdy = d2_dxdy.apply(f)

        assert_array_almost_equal(np.cos(X)*np.cos(Y), d2f_dxdy)


class TestDiscretizedDifferentialOperator(unittest.TestCase):

    def test_disc_diffoper_laplace_2d(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = X**3 + Y**3

        diff_op = DifferentialOperator((1, {0: 2}), (1, {1: 2}))

        discrete = DiscretizedDifferentialOperator(diff_op, grid, acc=4)
        actual = discrete.apply(f)

        expected = 6 * X + 6 * Y
        assert_array_almost_equal(expected, actual)


    def test_disc_diffoper_quasi_laplace_2d(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = X**3 + Y**3

        diff_op = DifferentialOperator((2, {0: 2}), (3, {1: 2}))

        discrete = DiscretizedDifferentialOperator(diff_op, grid, acc=4)
        actual = discrete.apply(f)

        expected = 12 * X + 18 * Y
        assert_array_almost_equal(expected, actual)


    def test_disc_diffoper_quasi_laplace_2d_variable_coefs(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = np.meshgrid(*grid.coords, indexing='ij')

        f = X**3 + Y**3

        diff_op = DifferentialOperator((X, {0: 2}), (X*Y, {1: 2}))

        discrete = DiscretizedDifferentialOperator(diff_op, grid, acc=4)
        actual = discrete.apply(f)

        expected = 6*X**2 + 6*X*Y**2
        assert_array_almost_equal(expected, actual)


import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from findiff import EquidistantGrid, Spacing, Coef
from findiff.core import Numberlike, Add
from findiff.core import PartialDerivative
from findiff.core import matrix_repr, stencils_repr


#np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3f" % x))
from test.base import TestBase


class TestMatrixRepr(unittest.TestCase):

    def test_matrix_repr_of_numberlike_2d(self):
        nl = Numberlike(3)
        actual = matrix_repr(nl, shape=(11, 11), spacing=Spacing(1))
        assert actual.shape == (11 ** 2, 11 ** 2)
        assert_array_equal(actual.toarray(), 3 * np.eye(11 ** 2))

    def test_matrix_repr_of_add_two_numberlikes_1d(self):
        nl1 = Numberlike(3)
        nl2 = Numberlike(4)
        actual = matrix_repr(Add(nl1, nl2), shape=(11,), spacing=Spacing(1))
        assert actual.shape == (11, 11)
        assert_array_equal(actual.toarray(), 7 * np.eye(11))

    def test_matrix_repr_of_coord_class_1d(self):
        x = np.linspace(0, 1, 11)
        X = Numberlike(x)
        actual = matrix_repr(X, shape=(11,), spacing=Spacing(1))
        assert actual.shape == (11, 11)
        assert_array_equal(actual.toarray(), X.value * np.eye(11))

    def test_matrix_repr_of_coord_class_2d(self):
        grid = EquidistantGrid((0.1, 4, 5), (0.1, 4, 5))

        X = Numberlike(grid.meshed_coords[0])
        actual = matrix_repr(X, grid=grid)
        assert actual.shape == (25, 25)

        expected = grid.meshed_coords[0].reshape(-1)
        assert_array_equal(np.diag(actual.toarray()),
                           expected)
        # and it should be diagonal:
        assert np.sum(actual.toarray() != 0) == 25

    def test_matrix_repr_of_normal_partial(self):
        grid = EquidistantGrid((0, 5, 6))
        d2_dx2 = PartialDerivative({0: 2})
        actual = matrix_repr(d2_dx2, grid=grid)
        expected = [
            [2, -5, 4, -1, 0, 0],
            [1, -2, 1, 0, 0, 0],
            [0, 1, -2, 1, 0, 0],
            [0, 0, 1, -2, 1, 0],
            [0, 0, 0, 1, -2, 1],
            [0, 0, -1, 4, -5, 2],
        ]
        assert_array_almost_equal(actual.toarray(), expected)

    def apply_with_matrix_repr(self, diff_op, f, grid, acc):
        matrix = matrix_repr(diff_op, grid=grid, acc=acc)
        return matrix.dot(f.reshape(-1)).reshape(grid.shape)

    def test_matrix_repr_of_mixed_partial(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = grid.meshed_coords
        f = X ** 2 * Y ** 2
        d2_dxdx = PartialDerivative({0: 1, 1: 1})
        expected = d2_dxdx.apply(f, grid=grid, acc=2)
        actual = self.apply_with_matrix_repr(d2_dxdx, f, grid, acc=2)
        assert_array_almost_equal(expected, actual)
        assert_array_almost_equal(4 * X * Y, actual)

    def test_matrix_repr_of_mixed_partial_linear_combination(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = grid.meshed_coords
        f = X ** 2 * Y ** 2
        diff_op = PartialDerivative({0: 1, 1: 1}) + PartialDerivative({0: 2})
        expected = diff_op.apply(f, grid=grid, acc=2)
        actual = self.apply_with_matrix_repr(diff_op, f, grid, acc=2)
        assert_array_almost_equal(expected, actual)
        assert_array_almost_equal(4 * X * Y + 2 * Y ** 2, actual)

    def test_matrix_repr_of_mixed_partial_linear_combination_constants(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = grid.meshed_coords
        f = X ** 2 * Y ** 2
        diff_op = PartialDerivative({0: 1, 1: 1}) + 2 * PartialDerivative({0: 2})
        expected = diff_op.apply(f, grid=grid, acc=2)
        actual = self.apply_with_matrix_repr(diff_op, f, grid, acc=2)
        assert_array_almost_equal(expected, actual)
        assert_array_almost_equal(4 * X * Y + 4 * Y ** 2, actual)

    def test_matrix_repr_of_mixed_partial_linear_combination_coords(self):
        grid = EquidistantGrid((0, 1, 101), (0, 1, 101))
        X, Y = grid.meshed_coords
        f = X ** 2 * Y ** 2
        diff_op = Coef(Y) * PartialDerivative({0: 1, 1: 1}) + Coef(X) * PartialDerivative({0: 2})
        expected = diff_op.apply(f, grid=grid, acc=2)
        actual = self.apply_with_matrix_repr(diff_op, f, grid, acc=2)
        assert_array_almost_equal(expected, actual)
        assert_array_almost_equal(4 * X * Y ** 2 + 2 * X * Y ** 2, actual)

    def test_unknown_type_raises_typeerror(self):
        with self.assertRaises(TypeError):
            matrix_repr({}, grid=EquidistantGrid.from_spacings({0: 1}), acc=2)

    def test_add_and_mult(self):
        laplacian = PartialDerivative({0: 2}) + 2 * PartialDerivative({1: 2})
        actual = matrix_repr(laplacian, spacing=Spacing(1), shape=(10, 10))
        print(actual.toarray())


class TestStencilsRepr(TestBase):

    def test_trivialstencilset_has_required_char_pts(self):
        nl = Numberlike(2)
        f = np.ones((5, 5))
        trivial = stencils_repr(nl, spacing=1, ndims=2)
        assert len(trivial.as_dict()) == 9

    def test_trivialstencilset_applied(self):
        nl = Numberlike(2)
        f = np.ones((5, 5))
        trivial = stencils_repr(nl, spacing=1, ndims=2)
        actual = trivial.apply(f)
        assert_array_almost_equal(2 * f, actual)

    def test_single_partial_deriv(self):
        pd = PartialDerivative({0: 1})
        stencil_set = stencils_repr(pd, spacing=Spacing(1), ndims=1)
        self.assertEqual(stencil_set[('C',)].as_dict(), {(-1,): -0.5, (0,): 0.0, (1,): 0.5})

    def test_laplacian_2d(self):
        laplacian = PartialDerivative({0: 2}) + PartialDerivative({1: 2})
        stencil_set = stencils_repr(laplacian, spacing=Spacing(1), ndims=2)
        self.assertEqual(
            {(-1, 0): 1.0, (0, 0): -4.0, (1, 0): 1.0, (0, -1): 1.0, (0, 1): 1.0},
            stencil_set[('C', 'C')].as_dict()
        )

    def test_add_and_mult(self):
        laplacian = PartialDerivative({0: 2}) + 2 * PartialDerivative({1: 2})
        stencil_set = stencils_repr(laplacian, spacing=Spacing(1), ndims=2)
        self.assertEqual(
            {(-1, 0): 1.0, (0, 0): -6.0, (1, 0): 1.0, (0, -1): 2.0, (0, 1): 2.0},
            stencil_set[('C', 'C')].as_dict()
        )

    def test_pass_invalid_type_raises_exception(self):
        with self.assertRaises(TypeError):
            stencils_repr('bla')

    def test_call_with_grid(self):
        laplacian = PartialDerivative({0: 2}) + 2 * PartialDerivative({1: 2})
        grid = EquidistantGrid((0, 10, 11), (0, 10, 11))
        stencil_set = stencils_repr(laplacian, grid=grid)
        self.assert_dict_values_almost_equal(
            {(-1, 0): 1.0, (0, 0): -6.0, (1, 0): 1.0, (0, -1): 2.0, (0, 1): 2.0},
            stencil_set[('C', 'C')].as_dict()
        )

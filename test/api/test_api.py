import unittest

import numpy as np
import sympy
from numpy.testing import assert_array_almost_equal

from findiff import Diff, Spacing
from findiff import InvalidGrid
from findiff import matrix_repr
from findiff.core.algebraic import Add
from findiff.core.deriv import PartialDerivative


class TestCoefficients(unittest.TestCase):

    def test_can_be_called_with_acc_and_default_symbolics(self):
        import findiff
        out = findiff.coefficients(2, 2, symbolic=True)
        assert isinstance(out['center']['coefficients'][1], sympy.Integer)

    def test_can_be_called_with_offsets(self):
        import findiff
        out = findiff.coefficients(2, offsets=[0, 1, 2, 3], symbolic=True)
        some_coef = out['coefficients'][1]
        assert isinstance(some_coef, sympy.Integer)
        assert some_coef == -5

    def test_must_specify_exactly_one(self):
        import findiff
        with self.assertRaises(ValueError):
            findiff.coefficients(2, acc=2, offsets=[-1, 0, 1])


class TestMatrixRepr(unittest.TestCase):

    def test_matrix_repr_of_second_deriv_1d_acc2(self):
        x = np.linspace(0, 6, 7)
        d2_dx2 = Diff(0, 2)
        actual = matrix_repr(d2_dx2, shape=x.shape, spacing=Spacing({0: 1}))

        expected = [[2, - 5, 4, - 1, 0, 0, 0],
                    [1, - 2, 1, 0, 0, 0, 0],
                    [0, 1, - 2, 1, 0, 0, 0, ],
                    [0, 0, 1, - 2, 1, 0, 0, ],
                    [0, 0, 0, 1, - 2, 1, 0, ],
                    [0, 0, 0, 0, 1, - 2, 1, ],
                    [0, 0, 0, - 1, 4, - 5, 2, ]]

        assert_array_almost_equal(actual.toarray(), expected)

    def test_matrix_repr_laplace_2d(self):
        laplace = Diff(0, 2) + Diff(1, 2)
        actual = matrix_repr(laplace, shape=(10, 10), spacing=Spacing(1))
        expected = matrix_repr(Add(PartialDerivative({0: 2}), PartialDerivative({1: 2})),shape=(10, 10), spacing=Spacing(1))
        assert_array_almost_equal(actual.toarray(), expected.toarray())

    def test_matrix_repr_with_single_spacing(self):
        laplace = Diff(0, 2) + Diff(1, 2)
        actual = matrix_repr(laplace, shape=(10, 10), spacing=Spacing(1))
        expected = matrix_repr(Add(PartialDerivative({0: 2}), PartialDerivative({1: 2})),shape=(10, 10), spacing=Spacing(1))
        assert_array_almost_equal(actual.toarray(), expected.toarray())

    def test_matrix_repr_with_negative_spacing_raises_exception(self):
        laplace = Diff(0, 2) + Diff(1, 2)
        with self.assertRaises(ValueError):
            matrix_repr(laplace, shape=(10, 10), spacing=-1)

    def test_matrix_repr_with_incomplete_spacing_raises_exception(self):
        laplace = Diff(0, 2) + Diff(1, 2)
        with self.assertRaises(InvalidGrid):
            matrix_repr(laplace, shape=(10, 10), spacing={0: 1})

    def test_matrix_repr_with_invalid_shape_raises_exception(self):
        laplace = Diff(0, 2) + Diff(1, 2)
        with self.assertRaises(InvalidGrid) as e:
            matrix_repr(laplace, shape=10, spacing=Spacing({1: 1}))
            assert 'must be tuple' in str(e)

    def test_matrix_repr_with_single_spacing_applied(self):
        num_pts = 100
        shape = num_pts, num_pts
        x = y = np.linspace(0, 1, num_pts)
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = X ** 2 + Y ** 2
        h = x[1] - x[0]

        laplace = Diff(0, 2) + Diff(1, 2)
        matrix = matrix_repr(laplace, shape=shape, spacing=h)
        actual = matrix.dot(f.reshape(-1)).reshape(shape)

        expected = 4 * np.ones_like(f)
        assert_array_almost_equal(expected, actual)

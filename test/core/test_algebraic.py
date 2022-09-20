import unittest

import numpy as np
from numpy.testing import assert_allclose

from findiff.core import PartialDerivative, Add, Numberlike, Coordinate
from findiff.core import Mul
from test.base import TestBase


class TestMul(unittest.TestCase):

    def test_multiply_partials_doesnt_return_mul(self):
        pd = PartialDerivative({0: 1})
        pd2 = PartialDerivative({0: 2})
        actual = pd * pd2
        assert type(actual) == PartialDerivative

    def test_multiply_number_with_partial_returns_mul(self):
        pd = PartialDerivative({0: 1})
        actual = 2 * pd
        assert type(actual) == Mul
        assert actual.left.value == 2
        assert actual.right == pd

    def test_multiply_partial_with_number_returns_mul(self):
        pd = PartialDerivative({0: 1})
        actual = pd * 2
        assert type(actual) == Mul
        assert actual.left == pd
        assert actual.right.value == 2

    def test_multiply_three_partials_returns_partial(self):
        pd0 = PartialDerivative({0: 1})
        pd1 = PartialDerivative({1: 1})
        pd2 = PartialDerivative({2: 1})
        actual = pd0 * pd1 * pd2
        assert {0: 1, 1: 1, 2: 1} == actual.degrees

    def test_multiply_three_factors_returns_mul(self):
        pd0 = PartialDerivative({0: 1})
        pd1 = PartialDerivative({1: 1})
        actual = pd0 * 2 * pd1
        assert actual.left.left == pd0
        assert actual.left.right.value == 2
        assert actual.right == pd1

    def test_rmul(self):
        diff = 2 * PartialDerivative({0: 1})
        assert isinstance(diff, Mul)
        assert diff.left == Numberlike(2)
        assert diff.right == PartialDerivative({0: 1})


class TestNumberlike(TestBase):

    def test_repr(self):
        assert repr(Numberlike(1)) == 'Numberlike(1)'


class TestAdd(TestBase):

    def test_repr(self):
        add = Add(PartialDerivative({0:1}), PartialDerivative({1:1}))
        assert 'Add({0: 1}, {1: 1})' == repr(add)

    def test_radd(self):
        one = Numberlike(1)
        add = 2 + one
        assert add.left == Numberlike(2)
        assert add.right == one

    def test_apply_add_right_not_numberlike(self):
        one = Numberlike(1)
        other = PartialDerivative({0:1})
        x = np.linspace(0, 1, 101)
        dx = 0.01
        f = x**2
        add = one + other
        result = add.apply(f, spacing=dx)
        assert_allclose(2*x+f, result, rtol=1.E-4)

    def test_apply_add_left_not_numberlike(self):
        one = Numberlike(1)
        other = PartialDerivative({0: 1})
        x = np.linspace(0, 1, 101)
        dx = 0.01
        f = x ** 2
        add = other + one
        result = add.apply(f, spacing=dx)
        assert_allclose(2 * x + f, result, rtol=1.E-4)


class TestCoordinate(TestBase):

    def test_equality(self):

        c1 = Coordinate(0)
        c2 = Coordinate(0)

        assert id(c1) != id(c2)
        assert c1 == c2


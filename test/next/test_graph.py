import unittest

from findiff.continuous import PartialDerivative, Coordinate
from findiff.arithmetic import Mul


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

    def test_replace_partial(self):
        pd0 = PartialDerivative({0: 1})
        pd1 = PartialDerivative({1: 1})
        mul = pd0 * 2 * pd1

        class DummyPartial:
            pass

        tester = lambda item: type(item) == PartialDerivative
        replacer = lambda item: DummyPartial()

        mul.replace(tester, replacer)
        assert type(mul.left.left) == DummyPartial
        assert mul.left.right.value == 2
        assert type(mul.right) == DummyPartial

    def test_replace_coordinate(self):
        pd0 = PartialDerivative({0: 1})
        pd1 = PartialDerivative({1: 1})
        mul = pd0 * Coordinate(0) * pd1

        tester = lambda item: type(item) == Coordinate and item.axis == 0
        replacer = lambda item: 7

        mul.replace(tester, replacer)
        assert type(mul.left.left) == PartialDerivative
        assert mul.left.right == 7
        assert type(mul.right) == PartialDerivative
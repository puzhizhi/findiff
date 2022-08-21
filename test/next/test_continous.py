import pytest
import unittest

from findiff.continuous import PartialDerivative, DifferentialOperator, Coordinate


#
# Tests for PartialDerivative
#
class TestPartialDerivative(unittest.TestCase):

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


class TestDifferentialOperator(unittest.TestCase):

    def test_diffop_get_coefficient(self):
        d2_dx2 = PartialDerivative({0: 2})
        d2_dy2 = PartialDerivative({1: 2})
        other_d2_dx2 = PartialDerivative({0: 2})
        # actual = DifferentialOperator({d2_dx2: 2, d2_dy2: 3})
        actual = DifferentialOperator((2, {0: 2}), (3, {1: 2}))
        assert actual.coefficient(d2_dx2) == 2
        assert actual.coefficient({0: 2}) == 2
        assert actual.coefficient(d2_dy2) == 3

    def test_diffop_with_coordinate_coef(self):
        x = Coordinate(0)
        other_x = Coordinate(0)
        y = Coordinate(1)
        actual = DifferentialOperator((x, {0: 2}), (y, {1: 2}))
        assert actual.coefficient({0: 2}) == x
        assert actual.coefficient({0: 2}) == other_x
        assert actual.coefficient({1: 2}) == y

    def test_add_diffops(self):
        diffop1 = DifferentialOperator((2, {0: 2}), (3, {1: 2}))
        diffop2 = DifferentialOperator((2, {0: 2}), (1, {1: 1}))
        actual = diffop1 + diffop2
        assert actual.coefficient({0: 2}) == 4
        assert actual.coefficient({1: 2}) == 3
        assert actual.coefficient({1: 1}) == 1

    def test_mul_diffop_with_partial(self):
        diffop = DifferentialOperator((2, {0: 2}), (3, {1: 2}))
        pd = PartialDerivative({0: 2})
        actual = diffop * pd
        assert actual.coefficient({0: 4}) == 2
        assert actual.coefficient({0: 2, 1: 2}) == 3

    def test_mul_partial_with_diffop_raises_exception(self):
        diffop = DifferentialOperator((2, {0: 2}), (3, {1: 2}))
        pd = PartialDerivative({0: 2})
        with self.assertRaises(TypeError):
             pd * diffop

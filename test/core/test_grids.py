import unittest

from sympy import Symbol

from findiff import Spacing, InvalidGrid
from findiff.core import EquidistantGrid
from test.base import TestBase


class TestEquidistantGrid(unittest.TestCase):

    def test_from_spacings_happy_path(self):
        ndims = 2
        spacings = {0: 1, 0: 1}
        grid = EquidistantGrid.from_spacings(ndims, spacings)

        assert len(grid.coords) == ndims
        x, y = grid.coords
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        assert grid.spacing(0) == 1
        assert grid.spacing(1) == 1

    def test_from_shape_spacings_happy_path(self):
        shape = (10, 10)
        spacings = {0: 1, 0: 1}
        grid = EquidistantGrid.from_shape_and_spacings(shape, spacings)

        assert len(grid.coords) == 2
        x, y = grid.coords
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        assert grid.spacing(0) == 1
        assert grid.spacing(1) == 1


class TestSpacing(TestBase):

    def test_init_with_valid_dict(self):
        spacing = Spacing({0: 1, 2: 3})

        assert spacing.axes == [0, 2]
        assert list(spacing.values()) == [1, 3]

    def test_init_with_invalid_dict(self):
        with self.assertRaises(ValueError):
            Spacing({-1: 1})
        with self.assertRaises(ValueError):
            Spacing({'a': 1})

    def test_init_with_valid_scalar(self):
        spacing = Spacing(1)
        assert spacing.isotropic
        assert spacing.for_axis(42) == 1

    def test_init_with_invalid_scalar(self):
        with self.assertRaises(InvalidGrid):
            Spacing(-1)

    def test_init_with_scalar_string(self):
        spacing = Spacing('r\Delta x')
        assert isinstance(spacing.for_axis(0), Symbol)

    def test_for_axis_isotropic(self):
        spacing = Spacing(3)
        assert spacing.for_axis(42) == 3

    def test_for_axis_nonisotropic(self):
        spacing = Spacing({0: 1, 2: 3})
        assert spacing.for_axis(0) == 1
        assert spacing.for_axis(2) == 3

    def test_for_axis_undefined_axis(self):
        spacing = Spacing({0: 1, 2: 3})
        with self.assertRaises(InvalidGrid):
            spacing.for_axis(1)

    def test_getitem_isotropic(self):
        spacing = Spacing(5)
        assert spacing[42] == 5

    def test_getitem_nonisotropic_defined_axis(self):
        spacing = Spacing({0: 1, 2: 3})
        assert spacing[2] == 3

    def test_getitem_nonisotropic_undefined_axis(self):
        spacing = Spacing({0: 1, 2: 3})
        with self.assertRaises(InvalidGrid):
            spacing[1]


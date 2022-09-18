import unittest

from findiff import EquidistantGrid


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

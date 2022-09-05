import unittest

from numpy.testing import assert_array_almost_equal

from findiff.core.deriv import EquidistantGrid


class TestEquidistantGrid(unittest.TestCase):

    def test_equidistantgrid_gives_spacing(self):
        grid = EquidistantGrid((-1, 1, 21), (0, 1, 21))
        assert_array_almost_equal(grid.spacings, [0.1, 0.05])

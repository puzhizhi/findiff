import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from findiff.grids import UniformGrid, EquidistantGrid


class TestUniformGrid(unittest.TestCase):

    def test_init_2d(self):
        shape = 30, 30
        spac = 0.1, 0.2
        center = 2, 3
        grid = UniformGrid(shape, spac, center)
        self.assertEqual(0.1, grid.spacing(0))
        np.testing.assert_array_equal(center, grid.center)

    def test_init_1d(self):
        grid = UniformGrid(30, 0.1)
        self.assertEqual(0.1, grid.spacing(0))
        np.testing.assert_array_equal((0,), grid.center)


class TestEquidistantGrid(unittest.TestCase):

    def test_equidistantgrid_gives_spacing(self):
        grid = EquidistantGrid((-1, 1, 21), (0, 1, 21))
        assert_array_almost_equal(grid.spacings, [0.1, 0.05])

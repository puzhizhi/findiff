import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from findiff.continuous import PartialDerivative
from findiff.discrete import EquidistantGrid, Stencil1D, SymmetricStencil1D, ForwardStencil1D, BackwardStencil1D, \
    DiscretizedPartialDerivative


#
# Tests for EquidistantGrid
#

def test_equidistantgrid_gives_spacing():
    grid = EquidistantGrid((-1, 1, 21), (0, 1, 21))
    assert_array_almost_equal(grid.spacings, [0.1, 0.05])

#
# Tests for Stencil1D
#

def test_stencil1d():
    s = Stencil1D(2, [-1, 0, 1], 1)
    print(repr(s))

#
# Tests for Stencil1D
#

def test_stencil1d():
    s = Stencil1D(2, [-1, 0, 1], 1)
    assert {-1: 1, 0: -2, 1: 1} == s.data


def test_symmetricstencil1d():
    s = SymmetricStencil1D(2, 1, 2)
    assert {-1: 1, 0: -2, 1: 1} == s.data


def test_forwardstencil1d():
    s = ForwardStencil1D(2, 1, 2)
    assert_array_almost_equal([2, -5, 4, -1], s.coefs)

    s = ForwardStencil1D(1, 1, 2)
    assert_array_almost_equal([-1.5, 2., -0.5], s.coefs)


def test_forwardstencil1d():
    s = BackwardStencil1D(2, 1, 2)
    assert_array_almost_equal([-1, 4, -5, 2], s.coefs)

    s = BackwardStencil1D(1, 1, 2)
    assert_array_almost_equal([0.5, -2, 1.5], s.coefs)


#
# Tests for DiscretizedPartialDerivative
#

def test_disc_part_deriv_1d():

    grid = EquidistantGrid((0, 1, 101))
    x = grid.coords[0]
    f = np.sin(x)
    pd = PartialDerivative({0: 1})
    d_dx = DiscretizedPartialDerivative(pd, grid, acc=2)
    df_dx = d_dx.apply(f)

    assert_array_almost_equal(np.cos(x[:1]), df_dx[:1], decimal=4)
    assert_array_almost_equal(np.cos(x[-1:]), df_dx[-1:], decimal=4)
    assert_array_almost_equal(np.cos(x[1:-1]), df_dx[1:-1], decimal=4)


def test_disc_part_deriv_1d_acc4():

    grid = EquidistantGrid((0, 1, 101))
    x = grid.coords[0]
    f = np.sin(x)
    pd = PartialDerivative({0: 1})
    d_dx = DiscretizedPartialDerivative(pd, grid, acc=4)
    df_dx = d_dx.apply(f)

    assert_array_almost_equal(np.cos(x[:1]), df_dx[:1], decimal=6)
    assert_array_almost_equal(np.cos(x[-1:]), df_dx[-1:], decimal=6)
    assert_array_almost_equal(np.cos(x[1:-1]), df_dx[1:-1], decimal=6)


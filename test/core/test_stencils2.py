import unittest

import numpy as np
from numpy.testing import assert_allclose

from findiff.core.deriv import PartialDerivative
from findiff.core.stencils2 import Stencil, StandardStencilFactory, SymmetricStencil, StencilSet, Spacing, \
    FlexStencilFactory

# Useful for debugging printouts of arrays
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3f" % x))


class UnitTestsStencil(unittest.TestCase):

    def test_create_stencil_from_constructor_stores_parameters(self):
        offsets = [(-1,), (0,), (1,)]
        coefs = [1., -2., 1.]
        stencil = Stencil(offsets, coefs)
        self.assertEqual(offsets, stencil.offsets)
        self.assertEqual(coefs, stencil.coefficients)

    def test_stencils_can_be_added(self):
        stencil_1 = Stencil({(0, -1): 1., (0, 0): -2., (0, 1): 1.})
        stencil_2 = Stencil({(0, -2): 1., (0, 0): -1., (0, 1): 1.})
        result = stencil_1 + stencil_2
        self.assertEqual(
            {(0, -2): 1, (0, -1): 1., (0, 0): -3., (0, 1): 2.},
            result.as_dict()
        )

    def test_stencil_can_be_applied(self):
        x = np.linspace(0, 1, 100)
        f = x ** 3

        factory = StandardStencilFactory()
        acc = 2
        dx = x[1] - x[0]
        deriv = 2
        stencil = factory.create(SymmetricStencil, deriv, dx, acc)

        mask = np.full_like(x, fill_value=False, dtype=bool)
        mask[1:-1] = True

        actual = stencil.apply(f, mask)
        assert_allclose(actual[1:-1], 6*x[1:-1], rtol=1.E-6)


class IntegrationTestsStencil(unittest.TestCase):
    ...


class UnitTestsStencilFactory(unittest.TestCase):
    ...


class IntegrationTestsStencilFactory(unittest.TestCase):

    def test_create_symmetric_stencil(self):
        factory = StandardStencilFactory()
        deriv = 2
        acc = 2
        spacing = 1

        stencil = factory.create(SymmetricStencil, deriv, spacing, acc)

        self.assertTrue(SymmetricStencil, type(stencil))

    def test_create_symmetric_stencil_correct_coefs(self):
        factory = StandardStencilFactory()
        stencil = factory.create(SymmetricStencil, deriv=2, spacing=1, acc=2)

        self.assertEqual({(0,): -2, (-1,): 1, (1,): 1}, stencil.as_dict())


class UnitTestsFlexStencilFactory(unittest.TestCase):
    ...


class IntegrationTestsFlexStencilFactory(unittest.TestCase):

    def test_flexstencilfactory_creates_generic_stencil_1d(self):
        factory = FlexStencilFactory()
        offsets = [(0,), (-1,), (1,)]
        pds = {(2,): 1}  # this is 1 * d2_dx2
        spacing = Spacing({0: 1})
        stencil = factory.create(offsets, pds, spacing)
        self.assertEqual({(0,): -2, (-1,): 1, (1,): 1}, stencil.as_dict())


class UnitTestsStencilSet(unittest.TestCase):
    ...


class IntegrationTestsStencilSet(unittest.TestCase):

    def test_create_stencilset_1d(self):
        pd = PartialDerivative({0: 2})
        spacing = Spacing({0: 1})
        ndims = 1
        acc = 2
        stencil_set = StencilSet(pd, spacing, ndims, acc)

        stencil = stencil_set[('C',)]
        self.assertEqual(
            {(0,): -2, (-1,): 1, (1,): 1},
            stencil.as_dict()
        )

        stencil = stencil_set[('L',)]
        self.assertEqual(
            {(0,): 2.0, (1,): -5.0, (2,): 4.0, (3,): -1.0},
            stencil.as_dict()
        )

        stencil = stencil_set[('H',)]
        self.assertEqual(
            {(-3,): -1.0, (-2,): 4.0, (-1,): -5.0, (0,): 2.0},
            stencil.as_dict()
        )

    def test_stencilset_can_be_applied_1d(self):
        x = np.linspace(0, 1, 100)
        dx = x[1] - x[0]
        f = x**3
        pd = PartialDerivative({0: 2})
        spacing = Spacing({0: dx})
        stencil_set = StencilSet(pd, spacing, ndims=1, acc=2)

        actual = stencil_set.apply(f)

        assert_allclose(6*x, actual, atol=1E-10)

    def test_stencilset_can_be_applied_2d(self):
        x = y = np.linspace(0, 1, 10)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = X
        pd = PartialDerivative({0: 1})
        spacing = Spacing({0: dx, 1: dy})
        stencil_set = StencilSet(pd, spacing, ndims=2, acc=2)

        actual = stencil_set.apply(f)

        assert_allclose(np.ones_like(f), actual, atol=1E-10)

    def test_stencilset_can_be_applied_2d_mixed(self):
        x = y = np.linspace(0, 1, 11)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = X*Y
        pd = PartialDerivative({0: 1, 1: 1})
        spacing = Spacing({0: dx, 1: dy})
        stencil_set = StencilSet(pd, spacing, ndims=2, acc=2)

        actual = stencil_set.apply(f)

        assert_allclose(np.ones_like(f), actual, atol=1E-10)

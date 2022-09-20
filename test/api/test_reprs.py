import numpy as np

from findiff import Diff, stencils_repr, Spacing, Identity, Coef
from test.base import TestBase


class TestStencilsRepr(TestBase):

    def test_stencils_for_d_dx(self):
        d_dx = Diff(0)
        actual = stencils_repr(d_dx, spacing=Spacing(1), ndims=1, acc=2)
        expected = {
            ('L',): {(0,): -1.5, (1,): 2.0, (2,): -0.5},
            ('C',): {(-1,): -0.5, (0,): 0.0, (1,): 0.5},
            ('H',): {(-2,): 0.5, (-1,): -2.0, (0,): 1.5}}

        for pos in [('L',), ('C',), ('H',)]:
            self.assertEqual(expected[pos], actual[pos].as_dict())

    def test_helmholtz_stencil_issue_60(self):
        # This is a regression test for issue #60.

        H = Identity() - Diff(0, 2)

        stencil_set = stencils_repr(H, spacing=1, ndims=1, acc=2)

        expected = {('L',): {(0,): -1.0, (1,): 5.0, (2,): -4.0, (3,): 1.0}, ('C',): {(-1,): -1.0, (0,): 3.0, (1,): -1.0},
         ('H',): {(-3,): 1.0, (-2,): -4.0, (-1,): 5.0, (0,): -1.0}}

        actual = stencil_set.as_dict()
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(expected.keys(), actual.keys())
        for key, expected_stencil in expected.items():
            actual_stencil = actual[key]

            self.assertEqual(expected_stencil, actual_stencil.as_dict())

    def test_infer_dimensions_from_diffs(self):
        diff = Diff(0, 2) + Coef(2) * Diff(2, 2)
        stencil_set = stencils_repr(diff)
        assert stencil_set.ndims == 3

    def test_infer_dimensions_from_coef(self):
        X = np.ones((10, 10, 10))
        diff = Diff(0, 2) + Coef(X) * Diff(1, 2)
        stencil_set = stencils_repr(diff)
        assert stencil_set.ndims == 3

    def test_use_default_spacing(self):
        diff = Diff(0, 2)
        stencil_set = stencils_repr(diff)
        assert stencil_set.ndims == 1
        self.assertEqual({(0,): -2, (1,): 1, (-1,): 1}, stencil_set['C',].as_dict())

    def test_use_explicit_spacing_and_inferred_ndims(self):
        diff = Diff(1, 2)
        stencil_set = stencils_repr(diff, spacing=0.1)
        assert stencil_set.ndims == 2
        inner_stencil = stencil_set['C','C']
        self.assert_dict_values_almost_equal({(0, 0): -200, (0, 1): 100, (0, -1): 100}, inner_stencil.as_dict())

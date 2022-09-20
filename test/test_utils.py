import unittest

from findiff.utils import require_parameter, require_exactly_one_parameter, require_at_most_one_parameter


class TestUtils(unittest.TestCase):

    def test_require_parameter_happy_path(self):
        assert 5 == require_parameter('para1', {'para1': 5, 'para2': 7}, 'Bla')

    def test_require_parameter_parameter_not_found(self):
        with self.assertRaises(ValueError):
            require_parameter('para42', {'para1': 5, 'para2': 7}, 'Bla')

    def test_require_exactly_one_parameter_both_given(self):
        with self.assertRaises(ValueError):
            require_exactly_one_parameter(
                ['para1', 'para2'],
                {'para1': 1, 'para2': 2},
                'bla'
            )

    def test_require_exactly_one_parameter_none_given(self):
        with self.assertRaises(ValueError):
            require_exactly_one_parameter(
                ['para1', 'para2'],
                {'para10': 1, 'para20': 2},
                'bla'
            )

    def test_require_at_most_one_parameter_both_given(self):
        with self.assertRaises(ValueError):
            require_at_most_one_parameter(
                ['para1', 'para2'],
                {'para1': 1, 'para2': 2},
                'bla'
            )

    def test_require_at_most_one_parameter_none_given(self):
        parameter = require_at_most_one_parameter(
            ['para1', 'para2'], {'para10': 1, 'para20': 2}, 'bla')
        assert parameter is None

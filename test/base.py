import unittest


class TestBase(unittest.TestCase):

    def assert_dict_values_almost_equal(self, dct1, dct2, places=7):
        assert set(dct1.keys()) == set(dct2.keys()), 'Keys do not match.'
        for key in dct1:
            self.assertAlmostEqual(
                dct1[key], dct2[key], places=places,
               msg='Mismatch in value for key: %s   Value 1: %s  Value 2: %s' %
                   (str(key), str(dct1[key]), str(dct2[key]))
            )

import unittest
from math import log2
from random import randrange
import numpy as np

from HW2 import load, split_data, class_probability, get_entropy, G, _GI, IG , CART

class HW2TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._data = load('../test.txt')

    def test_split(self):
        test_index = 1
        test_value = 30
        expected_indices_l = np.array([0, 1, 2, 4, 5, 7])
        expected_indices_r = np.array([3, 6, 8, 9])

        Dy, cy, Dn, cn = split_data(self._data, test_index, test_value)

        expected_y = np.take(self._data[0], expected_indices_l, axis=0)
        expected_cy = np.take(self._data[1], expected_indices_l)
        expected_n = np.take(self._data[0], expected_indices_r, axis=0)
        expected_cn = np.take(self._data[1], expected_indices_r)


        self.assertIsNone(np.testing.assert_equal(expected_y, Dy))
        self.assertIsNone(np.testing.assert_equal(expected_n, Dn))
        self.assertIsNone(np.testing.assert_equal(expected_cn, cn))
        self.assertIsNone(np.testing.assert_equal(expected_cy, cy))

    def test_entropy(self):
        num = 100

        for i in range(10):
            classes = np.array([randrange(0,2,1) for i in range(num)])

            instances = sum(classes)
            count = num

            p = class_probability(classes)

            try:
                expected = -(p*log2(p) + (1-p)*log2(1-p))
            except:
                expected = 0
            actual = get_entropy((None,classes))
            self.assertEqual(expected, actual)

    def test_ginni(self):
        test_index = 1
        test_value = 26

        expected_initial = 1 - ((3/10)**2 + (7/10)**2)
        g_0 = _GI(self._data[1])

        g = G(self._data, test_index, test_value)
        expected_g = 0.41904761

        self.assertAlmostEqual(expected_initial, g_0, 4)
        self.assertAlmostEqual(expected_g, g, 4)

    def test_IG(self):
        test_index = 1
        test_value = 26

        expected_gain = 0.88129089
        expected_gain = 0.001617751
        actual = IG(self._data, test_index, test_value)
        self.assertAlmostEqual(expected_gain, actual, 4)

    def test_cart(self):
        test_index = 1
        test_value = 26

        expected_cart = 0.04
        actual = CART(self._data,test_index, test_value)

        self.assertAlmostEqual(expected_cart, actual, 3)
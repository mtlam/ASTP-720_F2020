"""
Michael Lam
ASTP-720, Spring 2020

Unit tests for interpolation routine
"""
import unittest
import numpy as np
import sys
sys.path.append("../") #lazy but it works
from interpolation import linear_interpolator


class TestInterpolator(unittest.TestCase):
    """ Unit tester for interpolator.py """

    def test_linear_interpolator(self):
        """ Test linear interpolator """
        xs = [3, 4, 6.5, 10]
        ys = [15, 3, -1, 6.0]
        func = linear_interpolator(xs, ys)
        self.assertEqual(func(3.5), 9)
        self.assertAlmostEqual(func(7), 0.0) # floating point precision



if __name__ == '__main__':
    unittest.main()

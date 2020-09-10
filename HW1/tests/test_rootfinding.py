"""
Michael Lam
ASTP-720, Spring 2020

Unit tests for root finding routines
"""
import unittest
import numpy as np
import sys
sys.path.append("../") #lazy but it works
from rootfinding import bisect, newton, secant


def func_sqrt(num):
    """ Function for estimating the sqrt via root finding """
    return lambda x: x**2 - num
sqrt45 = func_sqrt(45)



class TestRootFindingMethods(unittest.TestCase):
    """ Unit tester for rootfinding.py """

    def test_bisect(self):
        """ Test Bisection Method """
        self.assertAlmostEqual(bisect(sqrt45, 6, 7), np.sqrt(45))

    def test_newton(self):
        """ Test Newton's Method """
        self.assertAlmostEqual(newton(sqrt45, lambda x: 2*x, 6.0),
                               np.sqrt(45))

    def test_secant(self):
        """ Test Secant Method """
        self.assertAlmostEqual(secant(sqrt45, 6, 7), np.sqrt(45))



if __name__ == '__main__':
    unittest.main()

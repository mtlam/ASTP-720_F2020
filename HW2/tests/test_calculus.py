"""
Michael Lam
ASTP-720, Spring 2020

Unit tests for calculus routines
"""
import unittest
import numpy as np
import sys
sys.path.append("../") #lazy but it works
sys.path.append("../../HW1/") #lazy but it works
from calculus import derivative, integrate
from interpolation import linear_interpolator




def sinsq(x):
    """ sin^2(x) tester function """
    return np.sin(x)**2

def sinsq_prime(x):
    """ Derivative of sin^2(x) tester function """
    return 2*np.sin(x)*np.cos(x)


class TestCalculus(unittest.TestCase):
    """ Unit tester for calculus.py """

    def test_derivative(self):
        """ Test derivative """
        for x in range(10):
            self.assertAlmostEqual(derivative(sinsq, x, 0.00001),
                                   sinsq_prime(x))

    def test_integrate(self):
        """ Test integral methods """
        self.assertAlmostEqual(integrate(sinsq, 0, np.pi, slices=10000, mode="midpoint"),
                               np.pi/2, places=3)
        self.assertAlmostEqual(integrate(sinsq, 0, np.pi, slices=100, mode="trapezoid"),
                               np.pi/2)
        self.assertAlmostEqual(integrate(sinsq, 0, np.pi, slices=100, mode="simpson"),
                               np.pi/2)

        

if __name__ == '__main__':
    unittest.main()

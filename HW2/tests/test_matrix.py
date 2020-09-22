"""
Michael Lam
ASTP-720, Spring 2020

Unit tests for matrix class
For simplicity, will test against numpy arrays/matrices
"""

import unittest
import sys
sys.path.append("../") #lazy but it works
import numpy as np
from matrix import Matrix




def to_matrix(array):
    """ Helper function to make a Matrix class """
    return Matrix(array.tolist())



class TestMatrixClass(unittest.TestCase):
    """ Unit tester for matrix.py """

    def test_eq(self):
        """ Test equality """
        A = np.random.rand(5, 4)
        B = A[:]
        MA = to_matrix(A)
        MB = to_matrix(B)
        self.assertTrue(MA == MB)
        self.assertEqual(MA, MB)

    def test_add(self):
        """ Test addition """
        A = np.random.rand(5, 4)
        B = np.random.rand(5, 4)
        C = A + B
        MA = to_matrix(A)
        MB = to_matrix(B)
        MC = to_matrix(C)
        self.assertTrue(MA+MB == MC)
        self.assertEqual(MA+MB, MC)

    def test_mul(self):
        """ Test multiplication """
        A = np.random.rand(5, 4)
        B = np.random.rand(4, 5)
        C = A @ B #matrix multiplication in the new age
        D = A * 3
        MA = to_matrix(A)
        MB = to_matrix(B)
        MC = to_matrix(C)
        MD = to_matrix(D)
        self.assertEqual(MA*MB, MC)
        self.assertEqual(MA*3, MD)



    def test_transpose(self):
        """ Test matrix transpose """
        A = np.random.rand(5, 4)
        B = np.transpose(A)
        MA = to_matrix(A)
        MB = to_matrix(B)
        self.assertEqual(MA.transpose(), MB)


    def test_invert(self):
        """ Test matrix inversion """
        A = np.random.rand(10, 10)
        MA = to_matrix(A)
        Ainv = np.linalg.inv(A)
        self.assertEqual(MA.inverse(), to_matrix(Ainv))


    def test_trace(self):
        """ Test the trace """
        A = np.random.rand(10, 10)
        MA = to_matrix(A)
        self.assertAlmostEqual(MA.trace(), np.trace(A))


    def test_determinant(self):
        """ Test the determinant """
        A = np.random.rand(10, 10)
        MA = to_matrix(A)
        self.assertAlmostEqual(MA.determinant(), np.linalg.det(A))


    def test_LU(self):
        """ Test the LU decomposition """
        A = np.random.rand(10, 10)
        MA = to_matrix(A)
        ML, MU = MA.decomposeLU()
        self.assertEqual(ML*MU, MA)
        self.assertTrue(ML.is_lower_triangular())
        self.assertTrue(MU.is_upper_triangular())


    # Other tests:


    def test_check_square(self):
        """ Test square checking """
        A = np.random.rand(5, 4)
        MA = to_matrix(A)
        # assertRaises must be in a wrapper to catch the exception:
        # https://ongspxm.github.io/blog/2016/11/assertraises-testing-for-errors-in-unittest/
        self.assertRaises(IndexError, lambda: MA.check_square())

    def test_triangular_checks(self):
        """ Test boolean checks for triangular matrices """
        A = np.random.rand(10, 10)
        MA = to_matrix(A)
        L, U = MA.decomposeLU()
        self.assertTrue(L.is_lower_triangular())
        self.assertTrue(U.is_upper_triangular())

    def test_forward_sub(self):
        """ Test forward_substitution """
        A = np.random.rand(10, 10)
        b = np.random.rand(10, 1)
        MA = to_matrix(A)
        Mb = to_matrix(b)

        ML, _ = MA.decomposeLU()
        My = ML.forward_substitution(Mb)

        y = np.linalg.inv(np.array(ML.array)) @ b
        self.assertEqual(ML*My, Mb) #check internal consistency
        self.assertEqual(My, to_matrix(y)) #check consistency with linalg


    def test_backward_sub(self):
        """ Test backward_substitution """
        A = np.random.rand(10, 10)
        y = np.random.rand(10, 1)
        MA = to_matrix(A)
        My = to_matrix(y)

        _, MU = MA.decomposeLU()
        Mx = MU.backward_substitution(My)

        x = np.linalg.inv(np.array(MU.array)) @ y
        self.assertEqual(MU*Mx, My) #check internal consistency
        self.assertEqual(Mx, to_matrix(x)) #check consistency with linalg



    def test_solver(self):
        """ Test solve_linear_system """
        A = np.random.rand(10, 10)
        b = np.random.rand(10, 1)
        x = np.linalg.inv(A) @ b
        MA = to_matrix(A)
        Mb = to_matrix(b)
        Mx = MA.solve_linear_system(Mb)
        self.assertEqual(Mx, to_matrix(x))
        self.assertEqual(MA*Mx, Mb)



if __name__ == '__main__':
    unittest.main()

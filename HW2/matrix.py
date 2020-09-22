"""
Michael Lam
ASTP-720, Spring 2020

Class to handle matrix operations.
While not strictly required for this assignment,
this code does not use np.array
"""
import math


def make_matrix(rows, cols, value=0):
    """ Makes a new, empty matrix """
    return Matrix([[value for i in range(cols)] for j in range(rows)])



class Matrix:
    """ Matrix class """
    def __init__(self, array):
        """ Class initialization """
        self.array = array
        self.rows = len(self.array)
        self.cols = len(self.array[0])

        # If LU decomposition has been run,
        # store the results in these variables
        self.L = None
        self.U = None


    def __getitem__(self, inds):
        """
        Get item out of the main data array
        Uses numpy-like syntax
        """
        i, j = inds
        return self.array[i][j]

    def __setitem__(self, inds, value):
        """
        Set item into the main data array
        Uses numpy-like syntax
        """
        i, j = inds
        self.array[i][j] = value



    def __str__(self):
        """ Nice printing """
        string = "["
        for i in range(self.rows):
            string += "["
            for j in range(self.cols):
                item = self[i, j]
                if isinstance(item, int):
                    string += "%i,"%item
                elif isinstance(item, float) and abs(item) > 0.001:
                    string += "%0.3f,"%item
                elif isinstance(item, float):
                    string += "%0.3e,"%item
            string = string[:-1]+"],\n"
        string = string[:-2]+"]"
        return string
    __repr__ = __str__




    def __eq__(self, other):
        """ Check equality with another matrix """
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                # Need isclose (Python >= 3.5) for float precision
                if not math.isclose(self[i, j], other[i, j]):
                    return False
        return True


    def __add__(self, other):
        """ Add two matrices together, return a new matrix """
        if self.rows != other.rows or self.cols != other.cols:
            raise IndexError("Size of matrices are not equal: (%i, %i) != (%i, %i)"%
                             (self.rows, self.cols, other.rows, other.cols))

        newmat = make_matrix(self.rows, self.cols)
        for i in range(newmat.rows):
            for j in range(newmat.cols):
                newmat[i, j] = self[i, j] + other[i, j]
        return newmat

    def __mul__(self, other):
        """ Multiplies two matrices together, or multiplies a matrix by a value """
        if isinstance(other, (int, float)):
            newmat = make_matrix(self.rows, self.cols)
            for i in range(newmat.rows):
                for j in range(newmat.cols):
                    newmat[i, j] = self[i, j] * other
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise IndexError("Row/column mismatch: (%i, %i) x (%i, %i)"%
                                 (self.rows, self.cols, other.rows, other.cols))

            newmat = make_matrix(self.rows, other.cols)

            for i in range(newmat.rows):
                for j in range(newmat.cols):
                    for k in range(self.cols):
                        newmat[i, j] += self[i, k] * other[k, j]
        return newmat
    __rmul__ = __mul__


    def transpose(self):
        """ Transpose a matrix """
        newmat = make_matrix(self.cols, self.rows)

        for i in range(newmat.rows):
            for j in range(newmat.cols):
                newmat[i, j] = self[j, i]

        return newmat



    def inverse(self):
        """ Return the inverse of this matrix """
        self.check_square()


        N = self.rows

        inverse = make_matrix(N, N)

        # Solve on a per-column basis using Ax = b formalism
        for j in range(N):
            b = make_matrix(N, 1)
            b[j, 0] = 1

            x = self.solve_linear_system(b)

            for i in range(N):
                inverse[i, j] = x[i, 0]

        return inverse


    def trace(self):
        """ Return the trace of a matrix """
        self.check_square()

        retval = 0.0
        for i in range(self.rows):
            retval += self[i, i]

        return retval



    def determinant(self):
        """ Calculate determinant via the LU decomposition """
        if self.L is None or self.U is None:
            self.decomposeLU()

        retval = 1.0
        for i in range(self.rows):
            retval *= self.L[i, i] * self.U[i, i]
        return retval



    def decomposeLU(self):
        """
        Calculate the LU decomposition, return two matrices L and U
        Does not include pivoting

        See notes. This also borrows heavily from:
        https://rosettacode.org/wiki/LU_decomposition#Python
        """
        self.check_square()

        N = self.rows
        L = make_matrix(N, N)
        U = make_matrix(N, N)
        A = self #for more math friendly notation


        for j in range(N):
            L[j, j] = 1.0 #Doolittle factorization

            #e.g., if you are in column = 5, you go down 6 rows
            for i in range(j+1):
                U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
            #e.g., if you are in column = 5,
            # you start at row 5 and go down for the lower triangular matrix
            for i in range(j, N):
                L[i, j] = (A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))) / U[j, j]

        self.L = L
        self.U = U
        return L, U



    def solve_linear_system(self, b):
        """
        Will solve a system of the form:
        Ax = b
        for x
        """
        self.check_square()

        if self.L is None or self.U is None:
            self.decomposeLU()

        L = self.L
        U = self.U

        y = L.forward_substitution(b)
        x = U.backward_substitution(y)

        return x


    def forward_substitution(self, b):
        """
        Performs forward substitution of the style
        Ly = b
        This will check to make sure that self (L) is a lower diagonal matrix
        Does not explicitly assume L_ii = 0

        Parameters
        ----------
        b : An Nx1 matrix. Lists are not excepted
        """
        if not self.is_lower_triangular():
            raise ValueError("Not a lower triangular matrix")
        if b.cols != 1:
            raise IndexError("Require an Nx1 Matrix: (%i, %i)"%
                             (b.rows, b.cols))
        if b.rows != self.rows:
            raise IndexError("Row/column mismatch: (%i, %i) x (%i, %i)"%
                             (self.rows, self.cols, b.rows, b.cols))

        L = self
        N = self.rows

        y = make_matrix(N, 1)
        for i in range(N):
            y[i, 0] = (b[i, 0] - sum(L[i, k] * y[k, 0] for k in range(i))) / L[i, i]

        return y


    def backward_substitution(self, y):
        """
        Performs backward substitution of the form:
        Ux = y
        This will check to make sure that self (U) is an upper diagonal matrix
        """

        if not self.is_upper_triangular():
            raise ValueError("Not an upper triangular matrix")
        if y.cols != 1:
            raise IndexError("Require an Nx1 Matrix: (%i, %i)"%
                             (y.rows, y.cols))
        if y.rows != self.rows:
            raise IndexError("Row/column mismatch: (%i, %i) x (%i, %i)"%
                             (self.rows, self.cols, y.rows, y.cols))

        U = self
        N = self.rows

        x = make_matrix(N, 1)
        for i in range(N-1, -1, -1):
            x[i, 0] = (y[i, 0] - sum(U[i, k+1] * x[k+1, 0] for k in range(i, N-1))) / U[i, i]

        return x


    def check_square(self):
        """ Simple error raising used multiple times """
        if self.rows != self.cols:
            raise IndexError("Matrix is not square")


    def is_lower_triangular(self):
        """ Check if matrix is lower triangular """
        self.check_square()

        for i in range(self.rows):
            for j in range(i+1, self.rows):
                if self[i, j] != 0:
                    return False
        return True

    def is_upper_triangular(self):
        """ Check if matrix is upper triangular """
        self.check_square()

        for i in range(self.rows):
            for j in range(i):
                if self[i, j] != 0:
                    return False
        return True

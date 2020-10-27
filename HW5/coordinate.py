'''
Michael Lam
ASTP-720, Fall 2020

Class to represent a coordinate
'''

import numpy as np


class Coordinate:
    """ Cartesian coordinate (x,y) with some basic operations """
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def __str__(self):
        return "(%f, %f)"%(self.x, self.y)

    def __repr__(self):
        return "Coordinate(%f, %f)"%(self.x, self.y)


    def __eq__(self, other):
        """ Equate coordinates for testing """
        if self.x == other.x and self.y == other.y:
            return True
        return False


    def __add__(self, other):
        """ Add two coordinates together """
        return Coordinate(self.x + other.x, self.y + other.y)
    __radd__ = __add__


    def addX(self, x):
        """ Convenience function to add x to a coordinate """
        return Coordinate(self.x + x, self.y)


    def addY(self, y):
        """ Convenience function to add y to a coordinate """
        return Coordinate(self.x, self.y + y)



    def __sub__(self, other):
        """ Subtract two coordinates """
        return Coordinate(self.x - other.x, self.y - other.y)


    def __neg__(self):
        """ Negate each coordinate """
        return Coordinate(-1*self.x, -1*self.y)


    def __mul__(self, other):
        """ Multiply by a constant """
        return Coordinate(self.x*other, self.y*other)
    __rmul__ = __mul__


    def __truediv__(self, other):
        """ Divide by a constant """
        return Coordinate(self.x/other, self.y/other)


    def __ge__(self, other):
        """
        If BOTH x and y are greater than that of other, return True
        Handy for our box membership check
        """
        return self.x >= other.x and self.y >= other.y


    def __le__(self, other):
        """
        If BOTH x and y are less than that of other, return True
        Handy for our box membership check
        """
        return self.x <= other.x and self.y <= other.y


    def get_distance(self, other):
        """
        Return the distance between this and another Coordinate
        """
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

'''
Michael Lam
ASTP-720, Fall 2020

Class to represent a point-mass particle
Also performs the integration
'''
import numpy as np
from coordinate import Coordinate

class Particle:
    """
    Class that contains the coordinates and
    mass of a point particle

    In order to do the integration, it will
    also keep track of velocity information
    via the simple Verlet algorithm since
    we don't particularly care about velocity
    information and it's simpler.
    """
    def __init__(self, index, m, cminus, c):
        """
        Parameters
        ----------
        index : int
            Number labeling the particle
        m : float
            Mass of particle
        cminus : Coordinate
            Coordinate of particle at previous timestep
        c : Coordinate
            Coordinate of particle
        """
        self.index = index
        self.m = m
        self.cminus = cminus
        self.c = c
        self.accels = list()


    def __eq__(self, other):
        """ Equate solely by the index """
        if self.index == other.index:
            return True
        return False


    def verlet_step(self, h=1):
        """
        Parameters
        ----------
        h : float
            Timestep

        Returns
        -------
        None

        Updates the internal coordinates, does not
        return anything
        """

        # First, figure out the sum of the x and y accelerations independently
        ax = np.mean(list(map(lambda coord: coord.x, self.accels)))
        ay = np.mean(list(map(lambda coord: coord.y, self.accels)))

        newx = 2*self.c.x - self.cminus.x + h**2 * ax
        newy = 2*self.c.y - self.cminus.y + h**2 * ay

        # Update both the past and the current step simultaneously
        self.cminus, self.c = self.c, Coordinate(newx, newy)
        # now delete the list of accelerations
        self.accels = list()
        return


    def add_accel(self, accel):
        """
        Add an acceleration, given as a coordinate, to the list
        used to calculate the total update

        Parameters
        ----------
        accel : Coordinate
            Using a Coordinate as a vector because why not
        """
        self.accels.append(accel)


    def get_index(self):
        """ Return index of particle """
        return self.index


    def get_mass(self):
        """ Return mass of particle """
        return self.m


    def get_coord(self):
        """ Return coordinate of particle """
        return self.c


    def get_separation(self, other):
        """ Return separaton between this particle and another Coordinate """
        return self.c.get_distance(other.c)

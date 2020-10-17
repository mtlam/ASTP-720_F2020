'''
Michael Lam
ASTP-720, Fall 2020

Class to represent a point-mass particle
Also performs the integration
'''



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
    def __init__(self, m, cminus, c):
        """
        Parameters
        ----------
        m : float
            Mass of particle
        cminus : Coordinate
            Coordinate of particle at previous timestep
        c : Coordinate
            Coordinate of particle
        """
        self.m = m
        self.cminus = cminus
        self.c = c
        

    def verlet(self, accel, h):
        """
        Parameters
        ----------
        accel : float
            Coordinate values of accelerations
        h : float
            Timestep

        Returns
        -------
        None

        Updates the internal coordinates, does not
        return anything
        """

        # Update both the past and the current step simultaneously
        self.cm, self.c = self.c, 2*self.c - self.cm + h**2 * accel
        return


    def get_mass(self):
        """ Return mass of particle """
        return self.m


    def get_coord(self):
        """ Return coordinate of particle """
        return self.c

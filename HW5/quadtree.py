'''
Michael Lam
ASTP-720, Fall 2020

QuadTree code, separated from the main
N-body integration code
'''

from particle import Particle
from coordinate import Coordinate

def members(SWcoord, NEcoord, particles):
    """
    Helper function:
    Return the sub-list of particles that exist in the
    box defined by SWcoord and NE coord.

    Parameters
    ----------
    SWcoord: Coordinate
        Coordinate of the SW corner of the box
    NEcoord : Coordinate
        Coordinate of the NE corner of the box
    particles : list
        list of Particles to search membership over

    Returns
    -------
    retval : list
        list of particles existing in the region
    """
    retval = list()

    for particle in particles:
        c = particle.get_coord()
        # Using the operator overload logic defined in Coordinate
        if SWcoord <= c and c <= NEcoord:
            retval.append(particle)

    return retval


class QuadTree:
    """
    A specialized quad-tree class for the Barnes-Hut method

    The QuadTree is specified by four possible child nodes
    designated with NW, NE, SW, SE.

    Parameters
    ----------
    SWcoord: Coordinate
        Coordinate of the SW corner of the box
    NEcoord : Coordinate
        Coordinate of the NE corner of the box
    particles: list
        list of Particles to subdivide on
    """
    def __init__(self, SWcoord, NEcoord, particles):
        self.SWcoord = SWcoord
        self.NEcoord = NEcoord

        self.NW = None
        self.NE = None
        self.SW = None
        self.SE = None

        self.particle = None

        self.subdivide(particles)



    def subdivide(self, particles):
        """ Subdivide the particle list into new QuadTrees recursively """
        if len(particles) == 0:
            return None
        elif len(particles) == 1:
            self.particle = particles[0]
        else:
            dx2 = (self.NEcoord.x - self.SWcoord.x)/2.0
            dy2 = (self.NEcoord.y - self.SWcoord.y)/2.0

            # Determine NW box members, then make a new QuadTree
            NWmembers = members(self.SWcoord.addY(dy2), self.NEcoord.addX(-dx2), particles)
            self.NW = QuadTree(self.SWcoord.addY(dy2), self.NEcoord.addX(-dx2), NWmembers)
            # Determine NE box members, then make a new QuadTree
            NEmembers = members(self.SWcoord.addX(dx2).addY(dy2), self.NEcoord, particles)
            self.NE = QuadTree(self.SWcoord.addX(dx2).addY(dy2), self.NEcoord, NEmembers)
            # Determine SW box members, then make a new QuadTree
            SWmembers = members(self.SWcoord, self.SWcoord.addX(dx2).addY(dy2), particles)
            self.SW = QuadTree(self.SWcoord, self.SWcoord.addX(dx2).addY(dy2), SWmembers)
            # Determine SE box members, then make a new QuadTree
            SWmembers = members(self.SWcoord.addX(dx2), self.NEcoord.addY(-dy2), particles)
            self.SE = QuadTree(self.SWcoord.addX(dx2), self.NEcoord.addY(-dy2), SWmembers)


    def plot(self, ax):
        """ Recursively plot squares on a grid around the particles """
        if self.particle is not None: #there is a single particle, so make the box
            ax.plot([self.SWcoord.x, self.NEcoord.x], [self.SWcoord.y, self.SWcoord.y], 'r') #bottom
            ax.plot([self.SWcoord.x, self.NEcoord.x], [self.NEcoord.y, self.NEcoord.y], 'r') #top
            ax.plot([self.NEcoord.x, self.NEcoord.x], [self.SWcoord.y, self.NEcoord.y], 'r') #right
            ax.plot([self.SWcoord.x, self.SWcoord.x], [self.SWcoord.y, self.NEcoord.y], 'r') #left
            c = self.particle.get_coord()
            ax.plot([c.x], [c.y], 'k.')
        elif self.NW is not None: # there are sub-trees
            self.NW.plot(ax)
            self.NE.plot(ax)
            self.SW.plot(ax)
            self.SE.plot(ax)
        else: #this is an empty leaf
            return


    def get_mass(self):
        """ Return total mass of tree """
        if self.particle is not None: #there is a single particle
            return self.particle.get_mass()
        elif self.NW is not None: # there are sub-trees
            return self.NW.get_mass() + self.NE.get_mass() + self.SW.get_mass() + self.SE.get_mass()
        else: #this is an empty leaf
            return 0


    def get_COM(self):
        """ Return center-of-mass coordinates """
        pass



if __name__ == '__main__':
    import numpy as np
    from matplotlib.pyplot import *
    x0s, y0s = np.transpose(np.load("galaxies0.npy"))
    x1s, y1s = np.transpose(np.load("galaxies1.npy"))
    m = 1e12
    particles = []
    for i in range(len(x0s)):
        c0 = Coordinate(x0s[i], y0s[i])
        c1 = Coordinate(x1s[i], y1s[i])
        p = Particle(m, c0, c1)
        particles.append(p)
    SW = Coordinate(0, 0)
    NE = Coordinate(10, 10)
    qt = QuadTree(SW, NE, particles)
    fig = figure(figsize=(6,6))
    ax = subplot(111, aspect='equal')
    qt.plot(ax)
    show()

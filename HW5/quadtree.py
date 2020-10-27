'''
Michael Lam
ASTP-720, Fall 2020

QuadTree code, separated from the main
N-body integration code
Note that the multiple recursive functions are not efficient
as they are repetitive, but are not particularly designed for efficiency
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
        self.total_mass = 0.0
        self.center_of_mass = Coordinate(0, 0) #potentially useless coordinate

        self.subdivide(particles)



    def subdivide(self, particles):
        """
        Subdivide the particle list into new QuadTrees recursively
        - Sets self.particle if there is one particle.
        - At the same time, make sure to calculate the masses and
          centers of mass of the sub-trees.
        """
        if len(particles) == 0:
            return None
        elif len(particles) == 1:
            self.particle = particles[0]
            self.total_mass = self.particle.get_mass()
            self.center_of_mass = self.particle.get_coord()
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

            # Calculate total mass and COM
            NWcom = self.NW.get_COM()
            NWmass = self.NW.get_mass()
            NEcom = self.NE.get_COM()
            NEmass = self.NE.get_mass()
            SWcom = self.SW.get_COM()
            SWmass = self.SW.get_mass()
            SEcom = self.SE.get_COM()
            SEmass = self.SE.get_mass()

            self.total_mass = NWmass + NEmass + SWmass + SEmass

            numer = NWcom*NWmass + NEcom*NEmass + SWcom*SWmass + SEcom*SEmass
            denom = NWmass + NEmass + SWmass + SEmass
            self.center_of_mass = numer/denom


    def calc_acceleration(self, comp_particle, theta=None, G=6.67e-11):
        """
        Adds the accelerations to the comp_particle

        Parameters
        ----------
        comp_particle : Particle
            Particle to report the acceleration to
        theta : float
            If given, perform a threshold on length/distance
            If length/distance <= theta, just report COM acceleration
            Else continue to recurse
        G : float
            Gravitational constant to use. Defaults to SI units
        """
        if self.particle is not None: #there is a single particle
            if self.particle == comp_particle:
                return
            M = self.particle.get_mass()
            #distance = self.particle.get_separation(comp_particle)
            dx = self.particle.get_coord().x - comp_particle.get_coord().x
            dy = self.particle.get_coord().y - comp_particle.get_coord().y
            ax = G*M/dx**2
            ay = G*M/dy**2
            comp_particle.add_accel(Coordinate(ax, ay))
        # there are sub-trees,
        elif self.NW is not None:
            distance = self.get_center().get_distance(comp_particle.get_coord())
            length = self.get_length()
            # theta is given, and threshold is met
            if theta is not None and length/distance <= theta:
                M = self.total_mass
                # redefine distance as to the COM
                #distance = self.center_of_mass.get_distance(comp_particle.get_coord())
                dx = self.center_of_mass.x - comp_particle.get_coord().x
                dy = self.center_of_mass.y - comp_particle.get_coord().y
                ax = G*M/dx**2
                ay = G*M/dy**2
                comp_particle.add_accel(Coordinate(ax, ay))
            else:
                self.NW.calc_acceleration(comp_particle, theta=theta, G=G)
                self.NE.calc_acceleration(comp_particle, theta=theta, G=G)
                self.SW.calc_acceleration(comp_particle, theta=theta, G=G)
                self.SE.calc_acceleration(comp_particle, theta=theta, G=G)



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


    def get_center(self):
        """
        Return the Coordinate of the center of the cell
        Uses funky Coordinate operator overloads.
        """
        return (self.NEcoord - self.SWcoord)/2.0 + self.SWcoord



    def get_length(self):
        """
        Return the length of the side of a cell (assuming square)
        """
        return self.NEcoord.y - self.SWcoord.y


    def get_mass(self, recalculate=False):
        """ Return total mass of tree """
        if recalculate:
            if self.particle is not None: #there is a single particle
                return self.particle.get_mass(recalculate=True)
            elif self.NW is not None: # there are sub-trees
                return self.NW.get_mass(recalculate=True) + \
                       self.NE.get_mass(recalculate=True) + \
                       self.SW.get_mass(recalculate=True) + \
                       self.SE.get_mass(recalculate=True)
            else: #this is an empty leaf
                return 0
        else:
            return self.total_mass


    def get_COM(self, recalculate=False):
        """ Return center-of-mass coordinates """
        if recalculate:
            if self.particle is not None: #there is a single particle
                return self.particle.get_coord(recalculate=True)
            elif self.NW is not None: # there are sub-trees
                NWcom = self.NW.get_COM(recalculate=True)
                NWmass = self.NW.get_mass(recalculate=True)
                NEcom = self.NE.get_COM(recalculate=True)
                NEmass = self.NE.get_mass(recalculate=True)
                SWcom = self.SW.get_COM(recalculate=True)
                SWmass = self.SW.get_mass(recalculate=True)
                SEcom = self.SE.get_COM(recalculate=True)
                SEmass = self.SE.get_mass(recalculate=True)
                numer = NWcom*NWmass + NEcom*NEmass + SWcom*SWmass + SEcom*SEmass
                denom = NWmass + NEmass + SWmass + SEmass
                return numer/denom
            else: #this is an empty leaf
                return Coordinate(0, 0) #useless coordinate multiplied by 0 anyway
        else:
            return self.center_of_mass

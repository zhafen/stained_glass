#!/usr/bin/env python
'''Tools for generating idealized structures and mock observations of them.
'''

import copy
import numpy as np

import shapely.geometry as geometry
import shapely.affinity as affinity

from augment.augment import store_parameters

########################################################################

class IdealizedProjection( object ):

    @store_parameters
    def __init__( self, sidelength=20., c=(0,0), ):
        '''A class for creating idealized projections of physical
        distributions and mock observing them.

        Args:
            sidelength (float):
                Sidelength of the square containing the projection.

            c (tuple of floats, (2,)):
                Center of the square containing the projection
        '''

        # For storing structures
        self.structs = []
        self.struct_values = []

        # Parameters based off of input
        self.x_min = self.c[0] - self.sidelength/2
        self.x_max = self.c[0] + self.sidelength/2
        self.y_min = self.c[1] - self.sidelength/2
        self.y_max = self.c[1] + self.sidelength/2

    ########################################################################
    # Mock Observations
    ########################################################################

    def generate_sightlines( self, n, seed=None ):
        '''Generate the sightlines that will be used for the mock observations.

        Args:
            n (int): Number of sightlines to generate.
            seed (int): Random seed to use for generating.

        Modifes:
            self.n (int):
                Number of sightlines generated

            self.sl_xs (np.ndarray, (n,) ):
                X-positions of sightlines.

            self.sl_ys (np.ndarray, (n,) ):
                Y-positions of sightlines.

            self.sls (MultiPoint, (n,) ):
                Collection of shapely Point objects representing sightlines.
        '''
        
        if seed is not None:
            np.random.seed( seed )

        # Store and generate
        self.n = n
        self.sl_xs = np.random.uniform( self.x_min, self.x_max, n )
        self.sl_ys = np.random.uniform( self.y_min, self.y_max, n )
        self.sls = geometry.MultiPoint( list( zip( self.sl_xs, self.sl_ys ) ) )

    ########################################################################

    def evaluate_sightlines( self, ):

        # Loop over all structures
        vs = np.zeros( self.n )
        for i, s in enumerate( self.structs ):

            # Check if inside and if so add
            inside_s = s.contains( self.sls )
            vs[inside_s] += self.struct_values[i]

        return vs

    ########################################################################
    # Idealized Structure
    ########################################################################

    def add_background( self, value=1. ):
        '''Add a background distribution.

        Args:
            value (float):
                On-sky value associated with the structure.

        Modifies:
            self.structs (list of shapely objects):
                Adds background shape to the list of structures.

            self.struct_values (list of floats):
                Adds associated value.
        '''

        # Create
        s = geometry.Polygon( [
            ( self.x_min, self.y_min, ),
            ( self.x_min, self.y_max, ),
            ( self.x_max, self.y_max, ),
            ( self.x_max, self.y_min, ),
        ] )

        # Store
        self.structs.append( s )
        self.struct_values.append( value )

    ########################################################################

    def add_ellipse( self, c, a, b=None, rotation=None, value=1. ):
        '''Add an ellipse or circle.

        Args:
            c (tuple of floats, (2,)):
                Center of the ellipse.

            a (float):
                Radius of the circle (if b is None) or axis of the ellipse.

            b (float or None):
                If given second axis of the ellipse.

            rotation (float):
                Rotation of the ellipse from the x-axis in degrees.

            value (float):
                On-sky value associated with the structure.

        Modifies:
            self.structs (list of shapely objects):
                Adds ellipse structure to the list of structures.

            self.struct_values (list of floats):
                Adds associated value.
        '''

        # Ellipse or circle
        if b is None:
            b = a

        # Create a circle
        circ = geometry.Point( c ).buffer(1.) 

        # Shape it into an ellipse
        ellipse = affinity.scale( circ, a, b )

        # Rotate
        if rotation is not None:
            ellipse = affinity.rotate( ellipse, rotation )

        # Store
        self.structs.append( ellipse )
        self.struct_values.append( value )


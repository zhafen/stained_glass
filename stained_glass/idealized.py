#!/usr/bin/env python
'''Tools for generating idealized structures and mock observations of them.
'''

import copy
import numpy as np

import palettable

import descartes
import shapely.affinity as affinity
import shapely.geometry as geometry
import shapely.ops as ops

from augment import store_parameters

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
    # Core Functions
    ########################################################################

    def generate_idealized_projection( self ): 
        '''Calculate the idealized projection, combining all contained
        structures into on-the-sky shapes.
        Structures with the same value are combined into one shape.

        Modifies:
            self.ip (list of shapely Polygons, (n_combined,) ):
                Combined list of shapely polygons.

            self.ip_values (np.ndarray, (n_combined,) ):
                Sorted and combined list of associated values.
        '''

        # Get the sorted unique values first
        unique_vals = np.unique( self.struct_values )
        self.ip_values = np.sort( unique_vals )

        # Iterate through values
        structs_arr = np.array( self.structs )
        struct_vals_arr = np.array( self.struct_values )
        self.ip = []
        for i, val in enumerate( self.ip_values ):

            # Get structures with the same value
            matches_val = np.isclose( struct_vals_arr, val )
            structs_w_val = structs_arr[matches_val]

            # Create and store union
            self.ip.append( ops.unary_union( structs_w_val ) )

    ########################################################################

    def clear( self ):
        '''Remove all contained structures for a fresh start.'''

        self.structs = []
        self.struct_values = []

        del self.ip, self.ip_values

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
        '''Calculate the value of each sightline point according to the
        structures it intercepts.
        Sightlines that intersect overlapping shapes will use the highest
        value of the overlapping shapes.

        Returns:
            vs (np.ndarray, (n,) ):
                Value of each sightline according to what shapes it intersects.
        '''

        # Generate the projected image
        self.generate_idealized_projection()

        # Loop over all structures
        vs = np.zeros( self.n )
        for i, s in enumerate( self.ip ):

            # Check if inside and if so add
            inside_s = np.array([
                s.contains( sl ) for sl in self.sls
            ])
            vs[inside_s] = self.ip_values[i]

        return vs

    ########################################################################
    # Idealized Structure
    ########################################################################

    def add_background( self, value=1 ):
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

    def add_ellipse( self, c, a, b=None, rotation=None, value=1 ):
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

    ########################################################################

    def add_clumps(
        self,
        r_clump,
        c,
        r_area,
        fcov,
        value = 1,
        verbose = False,
    ):
        '''Add clumps to a circular area. The clumps will approximately cover
        f_cov * pi * r_area**2.

        Args:
            r_clump (float):
                Radius of each clump.

            c (tuple of floats, (2,)):
                Center of the area the clumps will be distributed in.

            r_area (float):
                Radius of the area the clumps will be distributed in.

            fcov (float):
                Fraction of the area that should be covered by clumps.

            value (int or float):
                Value associated with the clumps

            verbose (bool):
                If True, print out progress in adding clumps.

        Modifies:
            self.structs (list of shapely objects):
                Adds clumps structure to the list of structures.

            self.struct_values (list of floats):
                Adds associated value.
        '''

        # Loop over until we've surpassed the requested area
        a_clumps = 0.
        area = np.pi * r_area**2.
        a_covered = fcov * area
        clumps = []
        verbose_percentiles = list( range( 0, 100, 10 ) )
        while a_clumps < a_covered:

            if verbose:
                covered = int( ( a_clumps / a_covered ) * 100 )
                if covered in verbose_percentiles:
                    if covered % 10 == 0:
                        verbose_percentiles.remove( covered )
                        print( 'Adding clumps...{:0.0f}%'.format( covered ) )

            # Draw a random location for the clump
            x = np.random.uniform( c[0] - r_area, c[0] + r_area )
            y = np.random.uniform( c[1] - r_area, c[1] + r_area )
            
            # If too far out, throw out and continue
            if ( x - c[0] )**2. + ( y - c[1] )**2. > r_area**2.:
                continue

            # Turn into a clump
            clump = geometry.Point( (x, y) )
            clump = clump.buffer( r_clump )

            # Store
            clumps.append( clump )

            # Bump up area
            a_clumps += clump.area

        # Create structure
        clumps = geometry.MultiPolygon( clumps )

        # Store
        self.structs.append( clumps )
        self.struct_values.append( value )

    ########################################################################

    def add_curve(
        self,
        v1,
        v2,
        theta_a = 20.,
        theta_b = 40.,
        sign_a = 1.,
        sign_b = 1.,
        value = 1,
    ):
        '''Adds a curve with chosen width.

        Args:
            v1 (tuple of floats, (2,)):
                First end of the curve.

            v2 (tuple of floats, (2,)):
                Second end of the curve.

            theta_a (float):
                Angle in degrees of inner arc of the curve.

            theta_b (float):
                Angle in degrees of outer arc of the curve.

            sign_a, sign_b (float):
                What direction the curves should be facing. I can't fully
                understand the logic for this, so it may require some
                playing around.
                One tip is if sign_a == sign_b it will be a convex curve,
                and if sign_a != sign_b the arcs will be facing each other.

            value (int or float):
                Value associated with the clumps

        Modifies:
            self.structs (list of shapely objects):
                Adds clumps structure to the list of structures.

            self.struct_values (list of floats):
                Adds associated value.
        '''

        # Account for user error
        sign_a = np.sign( sign_a )
        sign_b = np.sign( sign_b )

        def create_circle_from_arc( v1, v2, theta, sign ):

            # Convert theta from degrees
            theta *= np.pi / 180.

            # Turn copies into arrays for easier use
            v1 = np.array( copy.copy( v1 ) )
            v2 = np.array( copy.copy( v2 ) )

            # Calculate vectors that go into finding the location
            d = v2 - v1
            d_mag = np.linalg.norm( d )
            f_mag = d_mag / ( 2. * np.tan( theta / 2. ) )

            # Calculate the direction of a vector midway between v1 and v2
            # and pointing at the center of the circle
            if np.isclose( d[0], 0. ):
                f = np.array([
                    1. / np.sqrt( 1. + ( d[0] / d[1] )**2. ),
                    0.,
                ])
            # Edge cases where the curve is exactly on-axis
            elif np.isclose( d[1], 0. ):
                f = np.array([
                    0.,
                    -1. / np.sqrt( 1. + ( d[1] / d[0] )**2. ),
                ])
            else:
                f = np.array([
                    1. / np.sqrt( 1. + ( d[0] / d[1] )**2. ),
                    -1. / np.sqrt( 1. + ( d[1] / d[0] )**2. ),
                ])

                # Flip the direction of f when necessary
                f[1] *= np.sign( d[0] * d[1])

            # Find the center
            c = 0.5 * ( v1 + v2 ) + sign * f_mag * f

            # Get the circle radius
            radius = d_mag / ( 2. * np.sin( theta / 2. ) )

            # Create a circle
            circ = geometry.Point( c ).buffer( radius ) 

            return circ

        # Create the circles
        circ_a = create_circle_from_arc( v1, v2, theta_a, sign_a )
        circ_b = create_circle_from_arc( v1, v2, theta_b, sign_b )

        # Create the curve (difference or intersection depending on curves)
        if np.sign( sign_a * sign_b ) > 0:
            thick_curve = circ_b.difference( circ_a )
        else:
            thick_curve = circ_b.intersection( circ_a )

        # Store
        self.structs.append( thick_curve )
        self.struct_values.append( value )

    ########################################################################

    def add_concentric_structures(
        self,
        struct,
        value,
        n_concentric = 3,
        dr = 1.,
        dv = -1,
    ):
        '''Add concentric structures (in projected space) surrounding the
        provided structure.

        Args:
            struct (shapely Geometric object):
                Structure to add concentric structures around.

            value (int or float):
                Associated value of the provided structure. Values of
                concentric structures will be relative to this.

            n_concentric (int):
                Number of concentric shapes.

            dr (float):
                Increase in distance from the provided structure.

            dv (int or float):
                Change in value relative to the provided structure.

        Modifies:
            self.structs :
                Appends concentric structures.

            self.struct_values :
                Appends concentric structure values
        '''

        if n_concentric > 0:

            # Create and store the concentric structure
            concentric = struct.buffer( dr )
            concentric_value = value + dv
            self.structs.append( concentric )
            self.struct_values.append( concentric_value )

            # Call recursively
            self.add_concentric_structures(
                concentric,
                concentric_value,
                n_concentric = n_concentric - 1,
                dr = dr,
                dv = dv,
            )

    ########################################################################
    # Plotting Utilities
    ########################################################################

    def plot_idealized_projection(
        self,
        ax,
        cmap = palettable.matplotlib.Magma_16.mpl_colormap,
        vmin = None,
        vmax = None,
    ):
        '''Plot the full idealized projection.

        Args:
            ax (matplotlib.axes): Axes to plot the projection on.
            cmap : Colormap to use for the colors of the different shapes.
            vmin (float): Lower limit for the color axis.
            vmax (float): Upper limit for the color axis.
        '''

        # Create the most up-to-date projection first
        self.generate_idealized_projection()

        # Colorlimits
        if vmin is None:
            vmin = self.ip_values.min() / 1.2
        if vmax is None:
            vmax = 1.2 * self.ip_values.max()

        for i, s in enumerate( self.ip ):

            # Choose the patch color
            color_value = (
                ( self.ip_values[i] - vmin ) /
                ( vmax - vmin )
            )
            color = cmap( color_value )

            # Add the patch
            patch = descartes.PolygonPatch(
                s,
                fc = color,
                ec = color,
                zorder = i,
            )
            ax.add_patch( patch )

        ax.set_xlim( self.x_min, self.x_max )
        ax.set_ylim( self.y_min, self.y_max )

    ########################################################################
    
    def plot_sightlines(
        self,
        ax,
        s = None,
        cmap = palettable.matplotlib.Magma_16.mpl_colormap,
        vmin = None,
        vmax = None,
    ):
        '''Plot the full idealized projection.

        Args:
            ax (matplotlib.axes): Axes to plot the projection on.
            s (float): Point size for sightlines. By default scales with n.
            cmap : Colormap to use for the colors of the different shapes.
            vmin (float): Lower limit for the color axis.
            vmax (float): Upper limit for the color axis.
        '''

        # Evaluate sightlines
        vs = self.evaluate_sightlines()

        # Point sizes
        if s is None:
            s = int( 100000 / self.n )

        # Color limits
        if vmin is None:
            vmin = self.ip_values.min() / 1.2
        if vmax is None:
            vmax = 1.2 * self.ip_values.max()

        # Scatter Plot
        ax.scatter(
            self.sl_xs,
            self.sl_ys,
            c = vs,
            s = s,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
        )

        ax.set_xlim( self.x_min, self.x_max )
        ax.set_ylim( self.y_min, self.y_max )


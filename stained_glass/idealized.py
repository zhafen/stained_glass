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


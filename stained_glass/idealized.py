#!/usr/bin/env python
'''Tools for generating idealized structures and mock observations of them.
'''

import copy
import numpy as np
import scipy
import warnings

import matplotlib.pyplot as plt
import palettable

import descartes
import shapely.affinity as affinity
import shapely.geometry as geometry
import shapely.ops as ops

from augment import store_parameters

from . import generate
from . import sample
from . import stats
from .utils import shapely_utils

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
        self.nopatch_structs = []

        # Parameters based off of input
        self.x_min = self.c[0] - self.sidelength/2
        self.x_max = self.c[0] + self.sidelength/2
        self.y_min = self.c[1] - self.sidelength/2
        self.y_max = self.c[1] + self.sidelength/2

    ########################################################################
    # Core Functions
    ########################################################################

    def generate_idealized_projection( self, method='highest value' ): 
        '''Calculate the idealized projection, combining all contained
        structures into on-the-sky shapes.
        Structures with the same value are combined into one shape.

        Modifies:
            self.ip (list of shapely Polygons, (n_combined,) ):
                Combined list of shapely polygons.

            self.ip_values (np.ndarray, (n_combined,) ):
                Sorted and combined list of associated values.
        '''

        if method != 'highest value':
            raise ValueError( 'Alternative methods not available yet.' )

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

            # Edge case for some multi-polygon structures
            if len( structs_w_val.shape ) > 1:
                structs_w_val = [
                    geometry.MultiPolygon( list( s ) )
                    for s in structs_w_val
                ]

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
        self.sl_coords = np.array( list( zip( self.sl_xs, self.sl_ys ) ) )
        self.sls = geometry.MultiPoint( self.sl_coords )

    ########################################################################

    def set_sightlines( self, coords ):
        '''Set existing sightlines for use in calculations.

        Args:
            coords (np.ndarray, (n,2)):
                X-Y coordinates of the sightlines.

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

        # Store
        self.sls = geometry.MultiPoint( coords )
        self.sl_xs = coords[:,0]
        self.sl_ys = coords[:,1]
        self.sl_coords = np.array( list( zip( self.sl_xs, self.sl_ys ) ) )
        self.n = self.sl_xs.size

    ########################################################################

    def evaluate_sightlines( self, method='highest value' ):
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

            if method == 'add':
                vs[inside_s] += self.ip_values[i]
            elif method == 'highest value':
                vs[inside_s] = self.ip_values[i]
            else:
                raise ValueError( 'Unrecognized method, {}'.format( method ) )

        # Now loop over all nopatch structures
        for i, fn in enumerate( self.nopatch_structs ):
            
            vs_i = fn( self.sl_coords )

            if method == 'add':
                vs += vs_i
            elif method == 'highest value':
                is_higher = vs_i > vs
                vs[is_higher] = vs_i[is_higher]
            else:
                raise ValueError( 'Unrecognized method, {}'.format( method ) )

        return vs

    ########################################################################
    # Sample/evaluate statistics
    ########################################################################

    def evaluate_pair_sampled_tpcf(
        self,
        edges,
        v_min,
        v_max,
        n_per_bin = None,
        pair_fraction_uncertainty = 0.1,
        estimator = 'simple',
        n_f_cov = 1000,
        extra_sightlines_mult = 1.2,
    ):

        assert self.c == (0,0), "Off-centered pair-sampled TPCF not available yet."

        # Calculate the number of pairs to generate
        if n_per_bin is None:
            n_per_bin = np.round( pair_fraction_uncertainty**-2 ).astype( int )
        n_bins = len( edges ) - 1
        n = n_bins * n_per_bin

        # Calculate the number of sightlines sent through to get
        # the requested number of pairs
        self.generate_sightlines( n_f_cov )
        f_cov = (
            ( self.evaluate_sightlines() > v_min ) &
            ( self.evaluate_sightlines() < v_max )
        ).sum().astype( float ) / n_f_cov
        n_sightlines = int( n / f_cov * extra_sightlines_mult )

        # Generate data coordinates
        self.generate_sightlines( n_sightlines )
        is_valid =  (
            ( self.evaluate_sightlines() > v_min ) &
            ( self.evaluate_sightlines() < v_max )
        )
        dd_coords1 = np.array( self.sls )[is_valid,:]

        # Generate the sampling coords
        v_edges = np.array([ v_min, v_max ])
        pair_sampler = sample.PairSampler( self.sidelength, edges, v_edges )
        dd_coords1, dd_coords2 = pair_sampler.generate_pair_sampling_coords(
            dd_coords1,
            n_per_bin = n_per_bin,
            pair_fraction_uncertainty = pair_fraction_uncertainty,
        )
        dr_coords1, dr_coords2 = pair_sampler.generate_pair_sampling_coords(
            n_per_bin = n_per_bin,
            pair_fraction_uncertainty = pair_fraction_uncertainty,
        )

        # Get sightline evaluations
        vs = []
        for coords in [ dd_coords1, dd_coords2, dr_coords1, dr_coords2 ]:
            vs_bins = []
            for coords_bin in coords:
                self.set_sightlines( coords_bin )
                vs_bins.append( self.evaluate_sightlines() )
            vs.append( vs_bins )
        dd_vs1, dd_vs2, dr_vs1, dr_vs2 = np.array( vs )

        # Calculate pair counts
        n_dd = pair_sampler.estimate_pair_counts( dd_vs2 )
        n_dr, n_rr = pair_sampler.estimate_pair_counts(
            dr_vs2,
            return_n_rr = True,
        )

        # Flatten results
        n_dd, n_rr = n_dd.flatten(), n_rr.flatten()

        # Return the statistic
        estimator_fn = getattr( stats, estimator )
        return estimator_fn( n_dd, n_dr, n_rr )

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

    def add_ellipse_nopatch( self, c, a, b=None, rotation=None, value=1 ):
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
            self.nopatch_structs (list of functions):
                Adds ellipse structure to the list of structures.
        '''

        assert b is None, "add_ellipse_nopatch actually only works for' \
            ' circles right now."

        def ellipse_fn( coords ):

            result = np.zeros( coords.shape[0] )

            result[np.linalg.norm( coords - c, axis=1 ) < a] = value

            return result

        # Store
        self.nopatch_structs.append( ellipse_fn )

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

    def add_clumps_nopatch(
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

        if verbose:
            print( 'Adding clumps.' )

        # Estimate the number of clumps needed
        target_area = fcov * np.pi * r_area**2.
        n_clump = target_area / ( np.pi * r_clump**2. )

        # # Correct for probability of overlap, (2r_clump)^2/(fcov r_area^2)
        # # p_overlap = 4. / n_clump
        # # Crude estimate, can be calculated better numerically
        # mean_overlap_area = np.pi * r_clump**2.
        # # actual_area_covered = (
        # #     target_area - p_overlap * n_clump * mean_overlap_area
        # # )
        # # correction_factor = target_area / actual_area_covered
        # # Simplified result:
        # correction_factor = 1./( 1. - 4. * mean_overlap_area / target_area )
        # n_clump *= correction_factor
        # if verbose:
        #     print(
        #         '    Correcting for overlapping clumps... ' \
        #         'n_clump multiplied by {:.3g}'.format( correction_factor )
        #     )
        #     print(
        #         '    Creating {:.2g} clumps to cover area'.format( n_clump )
        #     )

        # Generate coords
        clump_coords = generate.randoms_in_annulus( n_clump, 0., r_area )
        clump_coords += c

        # Generate a kd tree for the coords
        tree = scipy.spatial.cKDTree( clump_coords )

        def clump_value_fn( coords, ):

            # # Use cKDtree to find nearest clump
            # n = tree.query_ball_point( coords, r_clump, return_length=True )

            # if n > 0:
            #     return value
            # else:
            #     return 0.

            d, inds = tree.query( coords )

            # d_all = scipy.spatial.distance.cdist( coords, clump_coords )
            # d = np.nanmin( d_all, axis=1 )

            result = np.zeros( d.shape )
            result[d<r_clump] = value

            return result

        self.nopatch_structs.append( clump_value_fn )

    ########################################################################

    def add_curve(
        self,
        value = 1,
        **kwargs
    ):
        '''Adds a curve with chosen width.

        Args:
            value (int or float):
                Value associated with the clumps

        Kwargs:
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
                Adds curve structure to the list of structures.

            self.struct_values (list of floats):
                Adds associated value.
        '''

        thick_curve = shapely_utils.create_curve( **kwargs )

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
    
    def add_sphere(
        self,
        c,
        r,
        value,
        n_annuli = 32,
        evaluate_method = 'highest value',
    ):
        '''Add a uniform density sphere with radius r centered at c and central
        surface density value.
    
        Args:
            c ((2,) tuple of floats):
                Center coordinates.

            r (float):
                Radius of the sphere.

            value (float):
                Value at the center of the sphere.

            n_annuli (int):
                Number of concentric spheres to approximate the continuous
                distribution with.

            evaluate_method (str):
                Must be consistent with the method used for e.g.
                evaluate_sightlines.

        Modifies:
            self.structs :
                Appends structures that make up the sphere

            self.struct_values :
                Appends structure values that make up the sphere
        '''

        # Convert value to a density for the sphere
        den = value / ( 2. * r )

        # The surface density for a uniform density sphere.
        def surf_den_fn( a ):
            return 2. * den * np.sqrt( r**2. - a**2. )

        self.add_spherical_profile(
            c,
            r,
            surf_den_fn,
            n_annuli,
            evaluate_method,
        )

    ########################################################################

    def add_nfw(
        self,
        center,
        r_vir,
        m_vir,
        c = 10.,
        r_stop = None,
        n_annuli = 32,
        evaluate_method = 'highest value',
    ):
        ''' Add a structure representing an NFW (Navarro-Frank-White) profile.
        Uses the surface density calculated in Lokas&Mamon2001.
    
        Args:
            center ((2,) tuple of floats):
                Center coordinates.

            r_vir (float):
                Radius of the halo.

            m_vir (float):
                Mass of the halo. Technically best to choose a mass
                consistent with r_vir and an overdensity criterion.

            c (float):
                Concentration of the NFW profile.

            r_stop (float):
                Where to cut off the profile. Defaults to 2 r_vir.

            n_annuli (int):
                Number of concentric spheres to approximate the continuous
                distribution with.

            evaluate_method (str):
                Must be consistent with the method used for e.g.
                evaluate_sightlines.

        Modifies:
            self.structs :
                Appends structures that make up the profile

            self.struct_values :
                Appends structure values that make up the profile
        '''

        g_c = ( np.log( 1. + c ) - c / ( 1. + c ) )**-1.

        def surf_den_fn( a ):

            rf = a / r_vir

            if a > r_vir / c:
                C_inv = np.arccos( 1. / c / rf )
            else:
                C_inv = np.arccosh( 1. / c / rf )

            result = (
                ( c **2. * g_c / ( 2. * np.pi ) )
                * m_vir / r_vir**2.
                * ( 1. - np.abs( c**2. * rf**2. - 1. )**-0.5 * C_inv )
                / ( c**2. * rf**2. - 1. )
            )

            return result

        if r_stop is None:
            r_stop = 2. * r_vir

        self.add_spherical_profile(
            c = center,
            r = r_stop,
            surf_den_fn = surf_den_fn,
            n_annuli = n_annuli,
            evaluate_method = evaluate_method,
        )

    ########################################################################

    def add_spherical_profile(
        self,
        c,
        r,
        surf_den_fn,
        n_annuli = 32,
        evaluate_method = 'highest value',
    ):
        '''Add a spherical profile specified by surf_den_fn with radius r
        centered at c.
    
        Args:
            c ((2,) tuple of floats):
                Center coordinates.

            r (float):
                Radius of the sphere.

            surf_den_fn (function):
                Specifies the surface density as a function of projected
                distance from the center.

            n_annuli (int):
                Number of concentric spheres to approximate the continuous
                distribution with.

            evaluate_method (str):
                Must be consistent with the method used for e.g.
                evaluate_sightlines.

        Modifies:
            self.structs :
                Appends structures that make up the profile

            self.struct_values :
                Appends structure values that make up the profile
        '''

        # These circles make up the projected sphere.
        circle_radii = np.linspace( 0., r, n_annuli+1 )[1:][::-1]

        prev_val = 0.
        for a in circle_radii:

            value = surf_den_fn( a )

            if evaluate_method == 'add':
                new_value = copy.copy( value - prev_val )
                prev_val = value
                value = new_value
            elif evaluate_method == 'highest value':
                pass
            else:
                raise ValueError( 'Unrecognized evaluate_method, {}'.format(
                    evaluate_method ) )

            self.add_ellipse( c, a, value=value )

    ########################################################################
    # Plotting Utilities
    ########################################################################

    def plot_idealized_projection(
        self,
        ax,
        cmap = palettable.matplotlib.Magma_16.mpl_colormap,
        vmin = None,
        vmax = None,
        log_color_scale = False,
        patch_kwargs = None,
        **kwargs
    ):
        '''Plot the full idealized projection.

        Args:
            ax (matplotlib.axes): Axes to plot the projection on.
            cmap : Colormap to use for the colors of the different shapes.
            vmin (float): Lower limit for the color axis.
            vmax (float): Upper limit for the color axis.
        '''

        # Create the most up-to-date projection first
        self.generate_idealized_projection( **kwargs )

        values = self.ip_values
        if log_color_scale:
            values = np.log10( values )

        # Colorlimits
        if vmin is None:
            vmin = values.min() / 1.2
        if vmax is None:
            vmax = 1.2 * values.max()

        def color_chooser( value ):
            # Choose the patch color
            color_value = (
                ( value - vmin ) /
                ( vmax - vmin )
            )
            color = cmap( color_value )

            return color
        self.color_chooser = color_chooser

        used_patch_kwargs = {
            'linewidth': 0,
        }
        if patch_kwargs is not None:
            used_patch_kwargs.update( patch_kwargs )
        for i, s in enumerate( self.ip ):

            color = color_chooser( values[i] )

            # Add the patch
            patch = descartes.PolygonPatch(
                s,
                zorder = i,
                fc = color,
                **used_patch_kwargs
            )
            ax.add_patch( patch )

        ax.set_xlim( self.x_min, self.x_max )
        ax.set_ylim( self.y_min, self.y_max )

    def plot_idealized_projection_pixel(
        self,
        ax,
        resolution = ( 1024, 1024 ),
        cmap = palettable.matplotlib.Magma_16.mpl_colormap,
        vmin = None,
        vmax = None,
        log_color_scale = False,
        patch_kwargs = None,
    ):

        if hasattr( self, 'sls' ):
            warnings.warn( 'Overriding existing sightlines...' )

        # Create the sightlines
        # Lots of reshaping...
        xs = np.linspace( self.x_min, self.x_max, resolution[0] )
        ys = np.linspace( self.y_min, self.y_max, resolution[0] )
        xs_grid, ys_grid = np.meshgrid( xs, ys )
        xs, ys = xs_grid.flatten(), ys_grid.flatten()
        sl_coords = np.array( [ xs, ys ] ).transpose()
        self.set_sightlines( sl_coords )

        # Evaluate and shape back
        vs = self.evaluate_sightlines()
        values = np.reshape( vs, xs_grid.shape )

        if log_color_scale:
            values = np.log10( values )

        # Colorlimits
        if vmin is None:
            vmin = values.min() / 1.2
        if vmax is None:
            vmax = 1.2 * values.max()

        # Plot
        plt.imshow(
            values,
            vmin = vmin,
            vmax = vmax,
            cmap = cmap,
            extent = ( self.x_min, self.x_max, self.y_min, self.y_max ),
        )

        ax.set_xlim( self.x_min, self.x_max )
        ax.set_ylim( self.y_min, self.y_max )

        return values

    ########################################################################
    
    def plot_sightlines(
        self,
        ax,
        s = None,
        cmap = palettable.matplotlib.Magma_16.mpl_colormap,
        vmin = None,
        vmax = None,
        **kwargs
    ):
        '''Plot the full idealized projection.

        Args:
            ax (matplotlib.axes): Axes to plot the projection on.
            s (float): Point size for sightlines. By default scales with n.
            cmap : Colormap to use for the colors of the different shapes.
            vmin (float): Lower limit for the color axis.
            vmax (float): Upper limit for the color axis.
        '''

        # Store cmap
        self.cmap = cmap

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
            **kwargs
        )

        ax.set_xlim( self.x_min, self.x_max )
        ax.set_ylim( self.y_min, self.y_max )


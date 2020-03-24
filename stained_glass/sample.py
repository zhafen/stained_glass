#!/usr/bin/env python
'''Tools for generating idealized structures and mock observations of them.
'''

import copy
import numpy as np
import scipy

from augment import store_parameters

########################################################################
########################################################################

class PairSampler( object ):

    @store_parameters
    def __init__( self, sidelength, edges, v_edges ):
        '''Class for sampling data in pairs, aimed at efficient calculation of
        a TPCF.

        Args:
            edges (np.ndarray):
                Edges of the spacing bins for generating the pairs.
        '''

        # Parameters of bins
        self.n_bins = len( edges ) - 1
        self.n_v_bins = len( v_edges ) - 1

        # For storing coordinates
        self.coords = {}

    ########################################################################

    def generate_pair_sampling_coords(
        self,
        data_coords = None,
        n_per_bin = None,
        pair_fraction_uncertainty = 0.1,
        label = None,
    ):
        '''Generate the coordinates to be probed to produce the TPCF using
        the pair method.

        Args:
            data_coords (int or None):
                Pairs will be found randomly within the spacing bins relative
                to the data_coords being passed. This will usually be used to
                produce a count of data-data pairs later.
                If None, then pairs will be found relative to random points
                within the sidelength. This option will usually be used to
                produce a count of data-rando pairs later.

            n_per_bin (int or None):
                Number of pairs per spacing bin to generate. If None,
                calculated using pair_fraction_uncertainty = 1 / sqrt( n ).

            pair_fraction_uncertainty (float):
                If the output is used to calculate the fraction of pairs in
                a given spacing bin, this is the desired uncertainty in that
                quantity, based on a Poisson estimate.

            label (str):
                If given, store the coordinates to the class under this label.

        Returns:
            coords1 (np.ndarray, (n_bin, n,)):
                Primary coordinates for each pair, i.e. the points that are
                chosen first before choosing the paired points.

            coords2 (np.ndarray, (n_bin, n,)):
                Secondary coordinates for each pair, i.e. the points that are
                chosen based on coords1 and the spacing bins.
        '''

        # Choose the number of sightlines
        if n_per_bin is None:
            n_per_bin = np.round( pair_fraction_uncertainty**-2 ).astype( int )

        # Generate coords1
        if data_coords is not None:
            # If data given, then by sampling coords
            inds1 = np.random.choice(
                np.arange( data_coords.shape[0] ),
                (self.n_bins, n_per_bin),
                replace = False,
            )
            coords1 = np.array([ data_coords[inds] for inds in inds1 ])
        else:
            # Otherwise, random
            n_random = n_per_bin * self.n_bins
            coords1 = np.random.uniform(
                -self.sidelength/2.,
                self.sidelength/2.,
                ( self.n_bins, n_per_bin, 2),
            )

        # Generate coords2 based on coords1
        coords2 = []
        for i in range( self.n_bins ):

            # Get the spacing at which to sample
            ds_i = self.edges[i] + 0.5 * ( self.edges[i+1] - self.edges[i] )

            # Make random values for sampling
            coords1_i = coords1[i]
            dx_i = np.random.uniform( -1, 1, coords1_i.shape )
            # Scale to the right radius
            dx_i *= ds_i / np.linalg.norm( dx_i, axis=1 )[:,np.newaxis]

            # Create and append
            coords2.append( coords1_i + dx_i )

        coords2 = np.array( coords2 )

        if label is not None:
            self.coords[label] = {}
            self.coords[label]['coords1'] = coords1
            self.coords[label]['coords2'] = coords2

        return coords1, coords2

    ########################################################################

    def estimate_pair_counts(
        self,
        vs2,
        n_random = 1000,
    ):
        '''Estimate normalized pair counts from pair fractions in a given bin.
        '''

        n_per_bin = vs2.shape[1]

        # Calculate pair fractions
        v_fracs = []
        for j in range( self.n_v_bins ):

            v_count = (
                ( vs2 > self.v_edges[j] ) &
                ( vs2 <= self.v_edges[j+1] )
            ).sum( axis=1 ).astype( float )
            
            v_fracs.append( v_count / n_per_bin )
        v_fracs = np.array( v_fracs ).transpose()

        # Calculate pair counts for random points
        randoms = np.random.uniform(
            -self.sidelength/2.,
            self.sidelength/2.,
            ( n_random, 2),
        )
        tree = scipy.spatial.cKDTree( randoms )
        n_rr = tree.count_neighbors( tree, self.edges, cumulative=False )[1:]

        # Convolve and normalize to create normalized pair counts
        pair_counts = v_fracs * n_rr[:,np.newaxis]
        pair_counts /= pair_counts.sum( axis=0 )

        return pair_counts

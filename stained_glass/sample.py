#!/usr/bin/env python
'''Tools for generating idealized structures and mock observations of them.
'''

import copy
import numpy as np

from augment import store_parameters

########################################################################
########################################################################

class PairSampler( object ):

    @store_parameters
    def __init__( self, sidelength, ):
        '''Class for sampling data in pairs, aimed at efficient calculation of
        a TPCF.
        '''

        pass

    ########################################################################

    def generate_pair_sampling_coords(
        self,
        edges,
        data_coords = None,
        n_per_bin = None,
        pair_fraction_uncertainty = 0.1,
    ):
        '''Generate the coordinates to be probed to produce the TPCF using
        the pair method.

        Args:
            edges (np.ndarray):
                Edges of the spacing bins for generating the pairs.

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
        n_bins = len( edges ) - 1
        if data_coords is not None:
            # If data given, then by sampling coords
            inds1 = np.random.choice(
                np.arange( data_coords.shape[0] ),
                (n_bins, n_per_bin),
                replace = False,
            )
            coords1 = np.array([ data_coords[inds] for inds in inds1 ])
        else:
            # Otherwise, random
            n_random = n_per_bin * n_bins
            coords1 = np.random.uniform(
                -self.sidelength/2.,
                self.sidelength/2.,
                ( n_bins, n_per_bin, 2),
            )

        # Generate coords2 based on coords1
        coords2 = []
        for i in range( n_bins ):

            # Get the spacing at which to sample
            ds_i = edges[i] + 0.5 * ( edges[i+1] - edges[i] )

            # Make random values for sampling
            coords1_i = coords1[i]
            dx_i = np.random.uniform( -1, 1, coords1_i.shape )
            # Scale to the right radius
            dx_i *= ds_i / np.linalg.norm( dx_i, axis=1 )[:,np.newaxis]

            # Create and append
            coords2.append( coords1_i + dx_i )

        coords2 = np.array( coords2 )

        return coords1, coords2

    ########################################################################

    def estimate_pair_counts_from_pair_fractions( self ):
        '''Estimate DD or DR from pair fractions in a given bin.
        '''

        pass


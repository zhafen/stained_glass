#!/usr/bin/env python
'''Simple functions and variables for easily accessing common files and choices
of parameters.
'''

import numpy as np
import os

import scipy.spatial.distance as sci_dist

########################################################################
########################################################################

def two_point_cf(
    coords,
    mins,
    maxs,
    n_bins = 16,
    estimator = 'ls',
):
    '''Two-point radial autocorrelation function. A value of 0 means no
    correlation beyond that expected due to randomness. 1 + xi, where
    xi is the two-point correlation function is the probability of finding a
    point at that distance.

    Args:
        coords (array-like, (n_samples, n_dimensions):
            Input coordinates to evaluate.

        mins, maxs (array-likes, (n_dimensions) ):
            Minimum and maximum viable values for coordinates.

        n_bins (int):
            Number of bins over which to evaluate the metric.

        estimator (str):
            CF estimator to use. Defaults to the Landy&Szalay1993 metric.
            Options...
                'ls': Landy & Szalay (1993) metric, ( DD - 2DR + RR ) / RR
                'simple': DD / RR - 1
                'dp': Davis & Peebles (1983) metric, DD / DR -1

    Returns:
        A tuple containing...
            result ( array-like, (n_bins) ): 
                Evaluated function in each bin.

            edges ( array-like, (n_bins+1) ):
                Bin edges.
    '''

    # Create random points
    randoms = []
    n_samples = coords.shape[0]
    for i in range( len( mins ) ):
        randoms.append( np.random.uniform( mins[i], maxs[i], n_samples ) )
    randoms = np.array( randoms ).transpose()

    # Get raw distances
    dd_dists = sci_dist.cdist(
        coords,
        coords,
    )
    dr_dists = sci_dist.cdist(
        coords,
        randoms,
    )
    rr_dists = sci_dist.cdist(
        randoms,
        randoms,
    )

    # Choose only unique pairs
    dd_dists = np.tril( dd_dists )
    rr_dists = np.tril( rr_dists )

    # Create radial bins
    r_max = np.sqrt( ( ( maxs - mins )**2. ).sum() )
    edges = np.linspace( 0., r_max, n_bins+1 )

    # Estimators
    def ls( n_dd, n_dr, n_rr ):
        return ( n_dd - 2. * n_dr + n_rr ) / n_rr
    def simple( n_dd, n_dr, n_rr ):
        return n_dd / n_rr - 1.
    def dp( n_dd, n_dr, n_rr ):
        return n_dd / n_dr - 1.
    estimators = {
        'ls': ls,
        'simple': simple,
        'dp': dp,
    }

    # Function for counting number of pairs
    def count( dists, inner, outer ):

        # Count
        n = (
            ( dists > inner ) &
            ( dists < outer )
        ).sum()

        return n

    # Loop through radial bins
    result = []
    for i in range( n_bins ):

        # Count pairs
        n_dd = count( dd_dists, edges[i], edges[i+1] )
        n_dr = count( dr_dists, edges[i], edges[i+1] )
        n_rr = count( rr_dists, edges[i], edges[i+1] )

        # Normalize
        n_dd /= n_samples * ( n_samples - 1 ) / 2.
        n_dr /= n_samples**2.
        n_rr /= n_samples * ( n_samples - 1 ) / 2.

        bin_result = estimators[estimator]( n_dd, n_dr, n_rr )
        result.append( bin_result )

        # if ls_estimator < -1:
        #     #DEBUG
        #     import pdb; pdb.set_trace()

    result = np.array( result )

    return result, edges

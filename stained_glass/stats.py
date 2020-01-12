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
):
    '''Two-point radial autocorrelation function, evaluated using the
    Landy&Szalay1993 metric. A value of 0 means no correlation beyond that
    expected due to randomness.

    Args:
        coords (array-like, (n_samples, n_dimensions):
            Input coordinates to evaluate.

        mins, maxs (array-likes, (n_dimensions) ):
            Minimum and maximum viable values for coordinates.

        n_bins (int):
            Number of bins over which to evaluate the metric.

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

    # Choose only unique distances
    # dd_dists = np.tril( dd_dists )
    # rr_dists = np.tril( rr_dists )
    # Commented this out because double-counting is okay
    # (otherwise dr gets too much weight)

    # Create radial bins
    r_max = np.sqrt( ( ( maxs - mins )**2. ).sum() )
    edges = np.linspace( 0., r_max, n_bins+1 )

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

        ls_estimator = ( n_dd - 2. * n_dr + n_rr ) / n_rr
        result.append( ls_estimator )

    result = np.array( result )

    return result, edges

#!/usr/bin/env python
'''Simple functions and variables for easily accessing common files and choices
of parameters.
'''

import copy
import numpy as np
import os

import scipy.spatial.distance as sci_dist

########################################################################
########################################################################

def f_cov(
    coords,
):

    pass

########################################################################

def two_point_autocf(
    coords,
    randoms = None,
    mins = None,
    maxs = None,
    bins = 16,
    estimator = 'ls',
    n_realizations = None,
    return_est_input = False,
    max_value = None,
):
    '''Two-point radial correlation function. A value of 0 means no
    correlation beyond that expected due to randomness. 1 + xi, where
    xi is the two-point correlation function is the probability of finding a
    point at that distance.

    Args:
        coords (array-like, (n_samples, n_dimensions):
            Input coordinates to evaluate.

        randoms (None or array-like):
            Random values to use as comparison for the correlation function.
            If None, create random values with same dimension as the coords.

        mins, maxs (np.ndarrays, (n_dimensions) ):
            Minimum and maximum viable values for coordinates. Used when
            creating random values.

        bins (int or array-like):
            Number of bins over which to evaluate the metric or bins to use
            for the correlation function.

        estimator (str):
            CF estimator to use. Defaults to the Landy&Szalay1993 metric.
            Options...
                'ls': Landy & Szalay (1993) metric, ( DD - 2DR + RR ) / RR
                'simple': DD / RR - 1
                'dp': Davis & Peebles (1983) metric, DD / DR -1

        n_realizations (None or int):
            Number of realizations of the correlation function, changing the
            random input each time.
            NOTE: This likely provides an underestimate of the uncertainty...

        return_est_input (bool):
            If True return an array containing [ DD, DR, RR ] for each bin.

        max_value (None or float):
            When the number of samples is small the correlation function can
            be infinite if there are too few random samples. This option caps
            the value of the TPCF to avoid infinities.

    Returns:
        A tuple containing...
            result ( array-like, (n_bins) ): 
                Evaluated function in each bin.

            edges ( array-like, (n_bins+1) ):
                Bin edges.

            est_input ( Optional, array-like, (3,) ):
                An array containing [ DD, DR, RR ] for each bin.
    '''

    # When doing multiple realizations, turns into a recursive function
    if n_realizations is not None:

        assert randoms is None, "n_realizations doesn't work with randoms != None."

        input_args = copy.deepcopy( locals() )
        input_args['n_realizations'] = None
        result = []
        est_input = []
        for i in np.arange( n_realizations ):
            out = two_point_autocf( **input_args )

            # Account for variable output
            if input_args['return_est_input']:
                tpcf, edges, est_input_i = out
                est_input.append( est_input_i )
            else:
                tpcf, edges = out

            result.append( tpcf )

        result = np.array( result )

        if input_args['return_est_input']:
            return result, edges, est_input

        return result, edges

    # Pre-requisite for many parts...
    n_samples = coords.shape[0]

    # Create random points
    if randoms is None:
        randoms = []
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
    if isinstance( bins, int ):
        r_max = np.sqrt( ( ( maxs - mins )**2. ).sum() )
        edges = np.linspace( 0., r_max, bins+1 )
        n_bins = bins
    else:
        edges = bins
        n_bins = len( edges ) - 1

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
    est_input = []
    for i in range( n_bins ):

        # Count pairs
        n_dd = count( dd_dists, edges[i], edges[i+1] )
        n_dr = count( dr_dists, edges[i], edges[i+1] )
        n_rr = count( rr_dists, edges[i], edges[i+1] )

        # Normalize
        n_dd_n = n_dd / ( n_samples * ( n_samples - 1 ) / 2. )
        n_dr_n = n_dr / n_samples**2.
        n_rr_n = n_rr / ( n_samples * ( n_samples - 1 ) / 2. )

        bin_result = estimators[estimator]( n_dd_n, n_dr_n, n_rr_n )

        if max_value is not None:
            if bin_result > max_value:
                bin_result = max_value

        result.append( bin_result )
        est_input.append( [ n_dd, n_dr, n_rr, ] )

    result = np.array( result )
    est_input = np.array( est_input )

    if return_est_input:
        return result, edges, est_input

    return result, edges

########################################################################

def cf_med_and_interval( cf, max_value=10., q_lower=16., q_upper=84. ):
    '''Estimate the median and interval of a number of correlation functions.
    This applies a few additional tricks to handle edge cases. In particular...
    A) The correlation function can jump to infinity when there are not many
        samples. Therefore we bound the correlation above function by max_value.
    B) When the correlation function is bounded above we estimate the upper
        interval using the lower interval as an estimate.
    C) Being bounded is a sign of a poorly constrainted correlation function,
        so we set the median to np.nan when bounded.
    D) When even the lower interval is bounded we just add a large shaded
        region (0, 10*max_value) to indicate that the CF is poorly constrained.

    Args:
        cf (array-like, (n_bins,) ):
            Correlation function values.

        max_value (float):
            Value above which the CF counts as infinite.

        q_lower, q_upper (float):
            Lower and upper intervals.

    Returns:
        med, lower, upper (array-like, (n_bins,) ):
            CF median, lower interval, and uppper interval.
    '''

    med = 1. + np.nanpercentile( cf, 50, axis=0 )
    lower = 1. + np.nanpercentile( cf, 16, axis=0 )
    upper = 1. + np.nanpercentile( cf, 84, axis=0 )
    
    max_arr = np.full( med.shape, max_value + 1. )
    
    med_bounded = np.isclose( med, max_arr )
    lower_bounded = np.isclose( lower, max_arr )
    upper_bounded = np.isclose( upper, max_arr )
    
    mu_bounded = np.logical_and( med_bounded, upper_bounded )
    all_bounded = np.logical_and( med_bounded, lower_bounded, upper_bounded )
    
    # When bounded above use lower interval as an estimate
    upper[mu_bounded] = ( med + ( med - lower ) )[mu_bounded]
    
    # When the median is bounded, remove it
    med[med_bounded] = np.nan
    
    # When all bounded, add big limits
    upper[all_bounded] = max_value * 10.
    lower[all_bounded] = 0.
    
    return med, lower, upper

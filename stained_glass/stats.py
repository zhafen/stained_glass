#!/usr/bin/env python
'''Simple functions and variables for easily accessing common files and choices
of parameters.
'''

import copy
import numba
import numpy as np
import os

import scipy.spatial
import scipy.spatial.distance as sci_dist

from . import generate

########################################################################
# TPCF Estimators
########################################################################

def ls( n_dd, n_dr, n_rr ):
    return ( n_dd - 2. * n_dr + n_rr ) / n_rr
def simple( n_dd, n_dr, n_rr ):
    return n_dd / n_rr - 1.
def dp( n_dd, n_dr, n_rr ):
    return n_dd / n_dr - 1.
def n_dd( n_dd, n_dr, n_rr ):
    return n_dd
def n_dr( n_dd, n_dr, n_rr ):
    return n_dr
def n_rr( n_dd, n_dr, n_rr ):
    return n_rr

estimators = {
    'ls': ls,
    'simple': simple,
    'dp': dp,
    'n_dd': n_dd,
    'n_dr': n_dr,
    'n_rr': n_rr,
}

########################################################################

def two_point_autocf(
    coords,
    randoms = None,
    mins = None,
    maxes = None,
    bins = 16,
    estimator = 'ls',
    n_realizations = None,
    max_value = None,
    brute_force = False,
):
    '''Two-point distance correlation function. A value of 0 means no
    correlation beyond that expected due to randomness. 1 + xi, where
    xi is the two-point correlation function is the probability of finding a
    point at that distance.

    Args:
        coords (array-like, (n_samples, n_dimensions):
            Input coordinates to evaluate.

        randoms (None or array-like):
            Random values to use as comparison for the correlation function.
            If None, create random values with same dimension as the coords.

        mins, maxes (np.ndarrays, (n_dimensions) ):
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

    Returns:
        A tuple containing...
            result ( array-like, (n_bins) ):
                Evaluated function in each bin.

            edges ( array-like, (n_bins+1) ):
                Bin edges.
    '''

    # When doing multiple realizations, turns into a recursive function
    if n_realizations is not None:

        assert randoms is None, "n_realizations doesn't work with randoms != None."

        input_args = copy.deepcopy( locals() )
        input_args['n_realizations'] = None
        result = []
        for i in np.arange( n_realizations ):
            out = two_point_autocf( **input_args )

            tpcf, edges = out

            result.append( tpcf )

        result = np.array( result )

        return result, edges

    # Create bins
    if isinstance( bins, int ):
        r_max = np.sqrt( ( ( maxes - mins )**2. ).sum() )
        edges = np.linspace( 0., r_max, bins+1 )
        n_bins = bins
    else:
        edges = bins
        n_bins = len( edges ) - 1

    # Pre-requisite for many calcs
    n_samples = coords.shape[0]

    # Create random points
    if randoms is None:
        randoms = []
        for i in range( len( mins ) ):
            randoms.append( np.random.uniform( mins[i], maxes[i], n_samples ) )
        randoms = np.array( randoms ).transpose()
    n_randoms = randoms.shape[0]

    if not brute_force:
        # Setup KD Trees
        data_tree = scipy.spatial.cKDTree( coords )
        rand_tree = scipy.spatial.cKDTree( randoms )

        # Count
        n_dd = data_tree.count_neighbors( data_tree, edges, cumulative=False )
        n_dr = data_tree.count_neighbors( rand_tree, edges, cumulative=False )
        n_rr = rand_tree.count_neighbors( rand_tree, edges, cumulative=False )

        # Ignore the first bin, because thats everything with r < edges[0]
        n_dd, n_dr, n_rr = n_dd[1:], n_dr[1:], n_rr[1:]

        # Normalizations
        f = float( n_samples ) / float( n_randoms )
        n_dd_n = n_dd
        n_dr_n = f * n_dr
        n_rr_n = f**2. * n_rr

    else:
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

        # Function for counting number of pairs
        def count( dists, inner, outer ):

            # Count
            n = (
                ( dists > inner ) &
                ( dists < outer )
            ).sum()

            return n

        n_dd = np.array([
            count( dd_dists, edges[i], edges[i+1] ) for i in range( n_bins )
        ])
        n_dr = np.array([
            count( dr_dists, edges[i], edges[i+1] ) for i in range( n_bins )
        ])
        n_rr = np.array([
            count( rr_dists, edges[i], edges[i+1] ) for i in range( n_bins )
        ])

        # Normalize
        n_dd_n = n_dd / ( n_samples * ( n_samples - 1 ) / 2. )
        n_dr_n = n_dr / ( n_samples * n_randoms )
        n_rr_n = n_rr / ( n_randoms * ( n_randoms - 1 ) / 2. )

    result = estimators[estimator]( n_dd_n, n_dr_n, n_rr_n )

    return result, edges

########################################################################

def annuli_two_point_autocf(
    coords,
    radial_bins = 16,
    bins = 16,
    **kwargs
):
    '''Two-point distance correlation function normalized by the density in
    different radial bins, going outwards.

    Args:
        coords (array-like, (n_samples, n_dimensions):
            Input coordinates to evaluate.

    Keyword Args:
        mins, maxes (np.ndarrays, (n_dimensions) ):
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

            r_edges ( array-like, (n_annuli+1) ):
                Radial bin edges.
    '''

    assert 'randoms' not in kwargs, 'Radial TPCF does not take random values.'

    # Start by calcing radial distances
    r = np.sqrt( ( coords**2. ).sum( axis=1 ) )

    # Create radial bins
    if isinstance( radial_bins, int ):
        max_r = np.nanmax( r )
        r_edges = np.linspace( 0., max_r, radial_bins+1 )
        n_annuli = radial_bins
    else:
        r_edges = radial_bins
        n_annuli = len( r_edges ) - 1

    # Count the number of bins
    if isinstance( bins, int ):
        n_bins = bins
    else:
        n_bins = len( bins ) - 1

    # Loop over autocf
    radial_tpcfs = []
    for i in range( n_annuli ):
        r_in = r_edges[i]
        r_out = r_edges[i+1]

        # Select the coordinates in the radial bin
        in_r_bin = ( r_in < r ) & ( r < r_out )
        coords_r_bin = coords[in_r_bin,:]

        # When no data is valid, continue
        if in_r_bin.sum() == 0:
            radial_tpcfs.append( np.zeros( ( n_bins, ) ) )
            continue

        randoms = generate.randoms_in_annulus(
            coords_r_bin[:,0].size,
            r_in,
            r_out,
        )

        # Call auto tpcf
        tpcf, edges = two_point_autocf(
            coords_r_bin,
            bins = bins,
            randoms = randoms,
            **kwargs
        )

        radial_tpcfs.append( tpcf )

    return radial_tpcfs, edges, r_edges

########################################################################

def weighted_tpcf(
    coords,
    weights,
    edges,
    offset = 'square of mean weight',
    scaling = 'mean of weight squared',
    ignore_first_bin = True,
    min_n_per_bin = 3,
    convolve = False,
    return_info = False,
):
    '''Returns a weighted two-point autocorrelation function, where estimators
    involve a product of the weights at different locations.

    Args:
        coords (array-like, (n_samples, n_dimensions):
            Input coordinates to evaluate.

        weights (array-like, (n_samples,)):
            Input weights.

        edges (array-like):
            Spacing bins to use for the correlation function.

        offset (str or None):
            Subtract this from the result.

        scaling (str or None):
            Divide the result by this. The offset will be
            subtracted from this too.

        ignore_first_bin (bool):
            The first bin contains every pair with a separation below
            edges[0] (including the pairs themselves). By default we
            don't report this bin.

        min_n_per_bin (int or None):
            If an integer, this is the minimum number of data points that must
            be in a bin to report the weighted correlation function.
            Note that not only does calculating the correlation function using
            only two data points in a bin likely not make sense, but
            it can be derived that calculating the correlation function
            with default offset and scaling using only two data points will
            always produce a value of -1.

        convolve (bool):
            If True, each weight must be an array, and
            ww_ij_conv = sum( w_i * w_j ), as opposed to ww_ij = w_i * w_j.
            Normalization is done carefully s.t.
            if w_i_conv = w_i * [array of ones] then ww_conv = ww.
            This is also true for the offset and scaling, which are equally
            carefully normalized.

    Returns:
        A tuple containing...
            result ( array-like, (n_bins) ):
                Evaluated function in each bin.

            edges ( array-like, (n_bins+1) ):
                Bin edges.
    '''

    if len( weights ) == 0:
        return np.full( edges.size - 1, np.nan ), edges

    # Regular count
    data_tree = scipy.spatial.cKDTree( coords )
    dd = data_tree.count_neighbors( data_tree, edges, cumulative=False )

    if not convolve:
        result = data_tree.count_neighbors(
            data_tree,
            edges,
            weights = weights,
            cumulative = False,
        )

        result /= dd
    else:
        n, n_conv = weights.shape
        max_dist = edges[-1]

        @numba.njit
        def count_neighbors():
            dd = np.zeros( edges.size )
            result = np.zeros( edges.size )
            for i in range( n ):

                if i % int( n * 0.05 ) == 0:
                    print( i / n )

                for j in range( i, n ):

                    r = np.sqrt( ( ( coords[i] - coords[j] )**2. ).sum() )

                    # Skip points outside the bins
                    if max_dist < r:
                        continue

                    ww_ij = ( weights[i] * weights[j] ).sum()

                    # Store the result
                    k = np.searchsorted( edges, r )
                    result[k] += ww_ij
                    dd[k] += 1

            return result, dd

        result, dd_c = count_neighbors()
        result /= n_conv * dd_c

    info = {}
    info['initial'] = copy.copy( result )

    # Offset the result
    def apply_offset( values ):
        if offset is None:
            return values
        elif offset == 'square of mean weight':

            if not convolve:
                avg_val = weights
            else:
                avg_val = weights.mean( axis=1 )

            bin_sum = data_tree.count_neighbors(
                data_tree,
                edges,
                weights = ( avg_val, None ),
                cumulative = False,
            )
            bin_average = bin_sum / dd
            offset_value = bin_average**2.
            values -= offset_value
        else:
            raise ValueError( 'Unrecognized offset, {}'.format( offset ) )
        return values, offset_value

    if offset is None:
        result = apply_offset( result )
    else:
        result, info['offset'] = apply_offset( result )

    # Scale the result
    # Even when the scaling is none, we still want to normalize by the bin count
    if scaling is None:
        pass
    elif scaling == 'mean of weight squared':

        if not convolve:
            avg_val = weights**2.
        else:
            avg_val = ( weights**2. ).mean( axis=1 )

        bin_square_sum = data_tree.count_neighbors(
            data_tree,
            edges,
            weights = ( avg_val, None ),
            cumulative = False,
        )

        # These two lines are what's actually happening to the scaling, but
        # the dd cancels out
        # bin_square_average = bin_square_sum / dd
        # scaling = dd * bin_square_average
        scaling = bin_square_sum / dd

        # Apply the offset to the scaling too
        scaling, offset_value = apply_offset( scaling )

        info['scaling'] = scaling

        result /= scaling
    else:
        raise ValueError( 'Unrecognized scaling, {}'.format( scaling ) )

    # Correct for bins that have too few data points
    if min_n_per_bin is not None:
        result[dd<min_n_per_bin] = np.nan

    # Ignore the first bin, because thats everything with r < edges[0]
    if ignore_first_bin:
        result = result[1:]

        if return_info:
            for key, item in info.items():
                info[key] = item[1:]

    if not return_info:
        return result, edges
    else:
        return result, edges, info

########################################################################

def radial_weighted_tpcf(
    coords,
    weights,
    edges,
    r_bins = 16,
    accounting = 'subtraction',
    **kwargs
):
    '''Returns a weighted two-point autocorrelation function, where estimators
    involve a product of the weights at different locations.
    The mean value in a given radial bin is subtracted from the weight

    Args:
        coords (array-like, (n_samples, n_dimensions):
            Input coordinates to evaluate.

        weights (array-like, (n_samples,)):
            Input weights.

        edges (array-like):
            Spacing bins to use for the correlation function.

    Returns:
        A tuple containing...
            result ( array-like, (n_bins) ):
                Evaluated function in each bin.

            edges ( array-like, (n_bins+1) ):
                Bin edges.
    '''

    # Start by calcing radial distances
    r = np.sqrt( ( coords**2. ).sum( axis=1 ) )

    # Setup radial bins
    if isinstance( r_bins, int ):
        r_edges = np.linspace( 0., r.max(), r_bins+1 )
    else:
        r_edges = r_bins

    # Adjust weights
    used_weights = copy.copy( weights )
    for i in range( r_edges.size - 1 ):
        in_annuli = ( r_edges[i] < r ) & ( r < r_edges[i+1] )
        if np.isclose( np.nanmean( used_weights[in_annuli] ), 0. ):
            #DEBUG
            import pdb; pdb.set_trace()
        if accounting == 'subtraction':
            used_weights[in_annuli] -= np.nanmean( used_weights[in_annuli] )
        elif accounting == 'division':
            used_weights[in_annuli] /= np.nanmean( used_weights[in_annuli] )
        else:
            raise Exception( 'Unknown argument for accounting.' )

    result, edges = weighted_tpcf(
        coords,
        used_weights,
        edges,
        **kwargs
    )

    return result, edges

########################################################################

def annuli_weighted_tpcf(
    coords,
    weights,
    edges,
    r_bins = 16,
    **kwargs
):
    '''Returns a weighted two-point autocorrelation function, where estimators
    involve a product of the weights at different locations.
    The TPCF is calculated differently for each annuli.

    Args:
        coords (array-like, (n_samples, n_dimensions):
            Input coordinates to evaluate.

        weights (array-like, (n_samples,)):
            Input weights.

        edges (array-like):
            Edges to use for spacing in the TPCF.

        r_bins (array-like):
            Inner and outer annuli radii to use for the correlation function.

    Kwargs:
        Same as weighted_tpcf.

    Returns:
        A tuple containing...
            result ( array-like, (n_bins) ):
                Evaluated function in each bin.

            edges ( array-like, (n_bins+1) ):
                Bin edges.
    '''

    # Start by calcing radial distances
    r = np.sqrt( ( coords**2. ).sum( axis=1 ) )

    # Setup radial bins
    if isinstance( r_bins, int ):
        r_edges = np.linspace( 0., r.max(), r_bins + 1 )
    else:
        r_edges = r_bins

    result = []
    for i in range( len( r_edges ) - 1 ):

        in_annuli = ( r_edges[i] < r ) & ( r < r_edges[i+1] )
        used_weights = weights[in_annuli]
        used_coords = coords[in_annuli,:]

        result_a, edges = weighted_tpcf(
            used_coords,
            used_weights,
            edges,
            **kwargs
        )
        result.append( result_a )

    return result, edges

########################################################################

def spacing_distribution(
    coords,
    a_vals,
    edges,
    a_edges,
):

    # Count bins
    n_bins = len( edges ) - 1
    n_a_bins = len( a_edges ) - 1

    # Loop through a_edges
    result = []
    for i in np.arange( n_a_bins ):

        in_a_bin = (
            ( a_edges[i] < a_vals ) &
            ( a_vals < a_edges[i+1] )
        )
        selected_coords = coords[:,in_a_bin]

        # Count neighbors
        data_tree = scipy.spatial.cKDTree( selected_coords )
        n_dd = data_tree.count_neighbors( data_tree, edges, cumulative=False )

        result.append( n_dd )

    result = np.array( result )

    return result

########################################################################

def cf_med_and_interval( cf, max_value=10., q_lower=16., q_upper=84. ):
    '''Estimate the median and interval of a list of correlation functions.
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

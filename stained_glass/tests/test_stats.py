#!/usr/bin/env python
'''Testing for select.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import numpy.testing as npt
import unittest

import stained_glass.stats as stats

########################################################################
########################################################################


class TestTwoPointAutoCorrelationFunction( unittest.TestCase ):

    def test_default( self ):

        np.random.seed( 1234 )

        # Test input params
        x_min = 0.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 1000
        n_bins = 3

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        mins = np.array([ x_min, y_min ])
        maxes = np.array([ x_max, y_max ])

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        # Calculate the two point correlation function
        actual, edges = stats.two_point_autocf(
            coords,
            mins = mins,
            maxes = maxes,
            bins = n_bins,
        )

        npt.assert_allclose(
            expected,
            actual,
            atol = 1e-2,
        )

    ########################################################################

    def test_all_estimators( self ):

        np.random.seed( 1234 )

        # Test input params
        x_min = -1.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 1000
        n_bins = 3

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        mins = np.array([ x_min, y_min ])
        maxes = np.array([ x_max, y_max ])

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        for estimator in [ 'ls', 'simple', 'dp' ]:

            print( estimator )

            # Calculate the two point correlation function
            actual, edges = stats.two_point_autocf(
                coords,
                mins = mins,
                maxes = maxes,
                bins = n_bins,
                estimator = estimator,
            )

            # The inner area should agree well
            if estimator == 'ls':
                npt.assert_allclose(
                    expected,
                    actual,
                    atol = 1e-2,
                )
            # The simple and dp estimators are sensitive to edge effects
            else:
                npt.assert_allclose(
                    expected[1],
                    actual[1],
                    atol = 3e-2,
                )
                npt.assert_allclose(
                    expected[1:],
                    actual[1:],
                    atol = 2e-1,
                )

    ########################################################################

    def test_multiple_realizations( self ):

        np.random.seed( 1234 )

        # Test input params
        x_min = 0.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 1000
        n_bins = 3
        n_realizations = 100

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        mins = np.array([ x_min, y_min ])
        maxes = np.array([ x_max, y_max ])

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        # Calculate the two point correlation function
        actual, edges = stats.two_point_autocf(
            coords,
            mins = mins,
            maxes = maxes,
            bins = n_bins,
            n_realizations = n_realizations,
        )
        actual_med = np.nanmedian( actual, axis=0 )

        npt.assert_allclose(
            expected,
            actual_med,
            atol = 1e-2,
        )

########################################################################


class TestTwoPointAutoCorrelationFunctionBruteForce( unittest.TestCase ):

    def test_default( self ):

        np.random.seed( 1234 )

        # Test input params
        x_min = 0.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 1000
        n_bins = 3

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        mins = np.array([ x_min, y_min ])
        maxes = np.array([ x_max, y_max ])

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        # Calculate the two point correlation function
        actual, edges = stats.two_point_autocf(
            coords,
            mins = mins,
            maxes = maxes,
            bins = n_bins,
            brute_force = True,
        )

        npt.assert_allclose(
            expected,
            actual,
            atol = 1e-2,
        )

    ########################################################################

    def test_all_estimators( self ):

        np.random.seed( 1234 )

        # Test input params
        x_min = -1.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 1000
        n_bins = 3

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        mins = np.array([ x_min, y_min ])
        maxes = np.array([ x_max, y_max ])

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        for estimator in [ 'ls', 'simple', 'dp' ]:

            print( estimator )

            # Calculate the two point correlation function
            actual, edges = stats.two_point_autocf(
                coords,
                mins = mins,
                maxes = maxes,
                bins = n_bins,
                estimator = estimator,
                brute_force = True,
            )

            # The inner area should agree well
            if estimator == 'ls':
                npt.assert_allclose(
                    expected,
                    actual,
                    atol = 1e-2,
                )
            # The simple and dp estimators are sensitive to edge effects
            else:
                npt.assert_allclose(
                    expected[1],
                    actual[1],
                    atol = 3e-2,
                )
                npt.assert_allclose(
                    expected[1:],
                    actual[1:],
                    atol = 2e-1,
                )

    ########################################################################

    def test_multiple_realizations( self ):

        np.random.seed( 1234 )

        # Test input params
        x_min = 0.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 1000
        n_bins = 3
        n_realizations = 100

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        mins = np.array([ x_min, y_min ])
        maxes = np.array([ x_max, y_max ])

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        # Calculate the two point correlation function
        actual, edges = stats.two_point_autocf(
            coords,
            mins = mins,
            maxes = maxes,
            bins = n_bins,
            n_realizations = n_realizations,
            brute_force = True,
        )
        actual_med = np.nanmedian( actual, axis=0 )

        npt.assert_allclose(
            expected,
            actual_med,
            atol = 1e-2,
        )

########################################################################


class TestAnnuliTwoPointAutoCorrelationFunction( unittest.TestCase ):

    def test_default( self ):

        np.random.seed( 123 )

        # Test input params
        x_min = -2.
        x_max = 2.
        y_min = -2.
        y_max = 2.
        n_samples = 2000
        n_bins = 3

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        mins = np.array([ x_min, y_min ])
        maxes = np.array([ x_max, y_max ])

        # Calculate the two point correlation function
        actual, edges, annuli_edges = stats.annuli_two_point_autocf(
            coords,
            radial_bins = 2,
            mins = mins,
            maxes = maxes,
            bins = n_bins,
        )

        for i, r_tpcf in enumerate( actual ):

            # With fully random data we expect the array to be equal
            # to 0 in each bin
            # Account for nans on unprobed scales
            if i == 0:
                expected = np.zeros( ( n_bins, ) )
                expected[-1] = np.nan
            else:
                expected = np.zeros( ( n_bins, ) )

            npt.assert_allclose(
                expected,
                r_tpcf,
                atol = 0.1,
            )

########################################################################


class TestWeightedTPCF( unittest.TestCase ):

    def test_default( self ):

        np.random.seed( 1234 )

        # Test input params
        x_min = -1.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 1000
        n_bins = 3
        edges = np.linspace( 0., np.sqrt( 2. ), n_bins + 1 )

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        values = np.random.uniform( 0., 5., n_samples )

        # Function call

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.ones( ( n_bins, ) )

        # Calculate the two point correlation function
        actual, edges, info = stats.weighted_tpcf(
            coords,
            values,
            edges,
            offset = None,
            scaling = None,
            return_info = True,
        )

        npt.assert_allclose(
            expected,
            actual / actual.mean(),
            atol = 1e-2,
        )

    ########################################################################

    def test_offset( self ):

        # Test input params
        x_min = -1.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 1000
        n_bins = 3
        edges = np.linspace( 0., np.sqrt( 2. ), n_bins + 1 )

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        values = np.full( xs.shape, 5. )

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        # Function call
        # Calculate the two point correlation function
        actual, edges = stats.weighted_tpcf(
            coords,
            values,
            edges,
            scaling = None,
        )

        npt.assert_allclose(
            expected,
            actual,
        )

    ########################################################################

    def test_offset_one_bin( self ):

        # Test input params
        x_min = -1.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 5000
        n_bins = 1
        edges = ( 0., 10. )

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        values = np.random.normal( 100., 2., n_samples )

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        # Function call
        # Calculate the two point correlation function
        actual, edges = stats.weighted_tpcf(
            coords,
            values,
            edges,
        )

        npt.assert_allclose(
            expected,
            actual,
            atol = 1e-3,
        )

    ########################################################################

    def test_normalization_complex( self ):

        np.random.seed( 1234 )

        # Test input params
        sidelength = 200.
        x_min = -sidelength / 2.
        x_max = sidelength / 2.
        y_min = -sidelength / 2.
        y_max = sidelength / 2.
        n_samples = 1000
        n_bins = 5
        r_elevated = 50.
        elevated_value = 10.
        edges = np.logspace( 0., np.log10( sidelength * np.sqrt( 2. ) ), n_bins + 1 )

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        values = np.full( xs.shape, 5. )
        values[n_samples//2:] = 1. # Add a second population
        values[3*n_samples//4:] = 0. # Add a third population
        r = np.sqrt( ( coords**2. ).sum( axis=1 ) )
        values[r < r_elevated] = elevated_value

        # Function call
        # Calculate the two point correlation function
        actual, edges, info = stats.weighted_tpcf(
            coords,
            values,
            edges,
            ignore_first_bin = False,
            return_info = True,
        )

        assert info['initial'].shape == actual.shape
        assert info['initial_normalization'].shape == actual.shape
        assert info['offset'].shape == actual.shape
        assert info['scaling'].shape == actual.shape

        # For sightlines that don't probe the length scale we expect values of
        # 1.
        npt.assert_allclose(
            0.,
            actual[-1],
            atol = 0.1,
        )
        # The value expected for sightlines that probe elevated regions.
        npt.assert_allclose(
            1.,
            actual[0],
            atol = 0.05
        )

    ########################################################################

    def test_wiwj_distribution( self ):

        np.random.seed( 1234 )

        # Test input params
        sidelength = 200.
        x_min = -sidelength / 2.
        x_max = sidelength / 2.
        y_min = -sidelength / 2.
        y_max = sidelength / 2.
        n_samples = 1000
        n_bins = 5
        r_elevated = 50.
        elevated_value = 10.
        edges = np.logspace( 0., np.log10( sidelength * np.sqrt( 2. ) ), n_bins + 1 )
        distribution_bins = 16

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        values = np.full( xs.shape, 5. )
        values[n_samples//2:] = 1. # Add a second population
        values[3*n_samples//4:] = 0. # Add a third population
        r = np.sqrt( ( coords**2. ).sum( axis=1 ) )
        values[r < r_elevated] = elevated_value

        # Function call
        # Calculate the two point correlation function
        actual, edges, info = stats.weighted_tpcf(
            coords,
            values,
            edges,
            return_distribution = True,
            distribution_bins = distribution_bins,
            ignore_first_bin = False,
        )

        assert info['initial'].shape == actual.shape
        assert info['initial_normalization'].shape == actual.shape
        assert info['offset'].shape == actual.shape
        assert info['scaling'].shape == actual.shape
        assert info['distribution'].shape == ( actual.size, distribution_bins - 1 )

        # For sightlines that don't probe the length scale we expect values of
        # 1.
        npt.assert_allclose(
            0.,
            actual[-1],
            atol = 0.1,
        )
        # The value expected for sightlines that probe elevated regions.
        npt.assert_allclose(
            1.,
            actual[0],
            atol = 0.05
        )

    ########################################################################

    def test_behavior_insufficient_data( self ):
        np.random.seed( 1234 )

        # Test input params
        x_min = -1.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        n_samples = 10
        n_bins = 5
        edges = np.logspace( -2., np.log10( np.sqrt( 2. ) ), n_bins + 1 )

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        values = np.random.uniform( 0., 5., n_samples )

        # Function call

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.ones( ( n_bins, ) )

        # Calculate the two point correlation function
        actual, edges = stats.weighted_tpcf(
            coords,
            values,
            edges,
        )

        assert np.isnan( actual[2] )

########################################################################

class TestConvolvedWeightedTPCF( unittest.TestCase ):

    def test_equals_weighted_tpcf( self ):

        np.random.seed( 1234 )

        # Test input params
        sidelength = 200.
        x_min = -sidelength / 2.
        x_max = sidelength / 2.
        y_min = -sidelength / 2.
        y_max = sidelength / 2.
        n_samples = 1000
        n_bins = 5
        r_elevated = 50.
        elevated_value = 10.
        edges = np.logspace( 0., np.log10( sidelength * np.sqrt( 2. ) ), n_bins + 1 )

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        values = np.full( xs.shape, 5. )
        values[n_samples//2:] = 1. # Add a second population
        values[3*n_samples//4:] = 0. # Add a third population
        r = np.sqrt( ( coords**2. ).sum( axis=1 ) )
        values[r < r_elevated] = elevated_value

        # Second independent dimension
        profile = np.array([ 1., 1., 1., 1., 1., 1. ])
        con_values = []
        for value in values:
            con_values.append( value * profile )
        con_values = np.array( con_values )

        # Calculate the unweighted two point correlation function
        tpcf, edges = stats.weighted_tpcf(
            coords,
            values,
            edges,
        )

        # Function call
        mtpcf, edges = stats.weighted_tpcf(
            coords,
            con_values,
            edges,
            convolve = True,
        )

        npt.assert_allclose( tpcf, mtpcf )

    ########################################################################

    def test_wiwj_distribution( self ):

        np.random.seed( 1234 )

        # Test input params
        sidelength = 200.
        x_min = -sidelength / 2.
        x_max = sidelength / 2.
        y_min = -sidelength / 2.
        y_max = sidelength / 2.
        n_samples = 1000
        n_bins = 5
        r_elevated = 50.
        elevated_value = 10.
        edges = np.logspace( 0., np.log10( sidelength * np.sqrt( 2. ) ), n_bins + 1 )
        distribution_bins = 16

        # Test data
        xs = np.random.uniform( x_min, x_max, n_samples )
        ys = np.random.uniform( y_min, y_max, n_samples )
        coords = np.array([ xs, ys ]).transpose()
        values = np.full( xs.shape, 5. )
        values[n_samples//2:] = 1. # Add a second population
        values[3*n_samples//4:] = 0. # Add a third population
        r = np.sqrt( ( coords**2. ).sum( axis=1 ) )
        values[r < r_elevated] = elevated_value

        # Second independent dimension
        profile = np.array([ 1., 1., 1., 1., 1., 1. ])
        con_values = []
        for value in values:
            con_values.append( value * profile )
        con_values = np.array( con_values )

        # Function call
        # Calculate the two point correlation function
        actual, edges, info = stats.weighted_tpcf(
            coords,
            con_values,
            edges,
            convolve = True,
            return_distribution = True,
            distribution_bins = distribution_bins,
        )

        assert info['initial'].shape == actual.shape
        assert info['initial_normalization'].shape == actual.shape
        assert info['offset'].shape == actual.shape
        assert info['scaling'].shape == actual.shape
        assert info['distribution'].shape == ( actual.size, distribution_bins - 1 )

########################################################################

# class TestSpacingDistribution( unittest.TestCase ):
# 
#     def test_default( self ):
# 
#         assert False, "This isn't working yet."
# 
#         # Test input params
#         x_min = -2.
#         x_max = 2.
#         a_min = 0.
#         a_max = 1.
#         y_min = -2.
#         y_max = 2.
#         n_samples = 2000
#         n_bins = 16
#         n_a_bins = 10
# 
#         # Test data
#         xs = np.random.uniform( x_min, x_max, n_samples )
#         ys = np.random.uniform( y_min, y_max, n_samples )
#         a_vals = np.random.uniform( a_min, a_max, n_samples )
#         coords = np.array([ xs, ys ]).transpose()
# 
#         actual, edges, a_edges = stats.spacing_distribution(
#             coords,
#             a_vals,
#             np.linspace( 0., 4. * np.sqrt( 2. ), n_bins + 1 ),
#             np.linspace( a_min, a_max, n_a_bins + 1 ),
#         )
# 
#         assert actual.shape == ( n_bins, n_a_bins )
#         assert edges.shape == ( n_bins + 1, )
#         assert a_edges.shape == ( n_a_bins + 1, )
# 
# ########################################################################
# 
# 
# class TestTwoPointCorrelationFunction( unittest.TestCase ):
# 
#     def test_default( self ):
# 
#         assert False, "This isn't working yet."
# 
#         np.random.seed( 1234 )
# 
#         # Test input params
#         x_min = 0.
#         x_max = 2.
#         y_min = 0.
#         y_max = 2.
#         n_samples = 1000
#         n_corr = 100
#         n_bins = 3
# 
#         # Test data
#         xs = np.random.uniform( x_min, x_max, n_samples )
#         ys = np.random.uniform( y_min, y_max, n_samples )
#         coords = np.array([ xs, ys ]).transpose()
#         xs = np.random.uniform( x_min, x_max, n_corr )
#         ys = np.random.uniform( y_min, y_max, n_corr )
#         corr_coords = np.array([ xs, ys ]).transpose()
#         mins = np.array([ x_min, y_min ])
#         maxes = np.array([ x_max, y_max ])
# 
#         # With fully random data we expect the array to be equal
#         # to 0 in each bin
#         expected = np.zeros( ( n_bins, ) )
# 
#         # Calculate the two point correlation function
#         actual, edges = stats.two_point_cf(
#             coords,
#             corr_coords,
#             mins,
#             maxes,
#             n_bins,
#         )
# 
#         npt.assert_allclose(
#             expected,
#             actual,
#             atol = 1e-2,
#         )

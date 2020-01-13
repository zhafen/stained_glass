#!/usr/bin/env python
'''Testing for select.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import h5py
import mock
import numpy as np
import numpy.testing as npt
import os
import unittest

import stained_glass.stats as stats

########################################################################
########################################################################


class TestTwoPointCorrelationFunction( unittest.TestCase ):

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
        maxs = np.array([ x_max, y_max ])

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        # Calculate the two point correlation function
        actual, edges = stats.two_point_cf(
            coords,
            mins,
            maxs,
            n_bins,
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
        maxs = np.array([ x_max, y_max ])

        # With fully random data we expect the array to be equal
        # to 0 in each bin
        expected = np.zeros( ( n_bins, ) )

        for estimator in [ 'ls', 'simple', 'dp' ]:

            print( estimator )

            # Calculate the two point correlation function
            actual, edges = stats.two_point_cf(
                coords,
                mins,
                maxs,
                n_bins,
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



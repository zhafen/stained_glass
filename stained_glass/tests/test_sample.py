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

import stained_glass.idealized as idealized
import stained_glass.sample as sample
import stained_glass.stats as stats

########################################################################
########################################################################

class TestPairSamplingTPCF( unittest.TestCase ):

    def test_default( self ):
        '''Hard to do piecewise tests for this, so test the results.
        '''

        # Create an idealized projection to base it on.
        ip = idealized.IdealizedProjection()
        ip.add_ellipse(
            c = (0., 0.),
            a = 5.,
        )

        # Set-up bins
        edges = np.linspace( 0., ip.sidelength/2., 5 )

        # Generate some sightlines through the data for the primary data coords.
        n_data_coords = 5000
        ip.generate_sightlines( n_data_coords )
        is_valid = ip.evaluate_sightlines() >  0
        data_coords = np.array( ip.sls )[is_valid,:]

        # Generate the sampling coords
        pair_sampler = sample.PairSampler( ip.sidelength, )
        dd_coords1, dd_coords2 = pair_sampler.generate_pair_sampling_coords(
            edges,
            data_coords,
        )
        dr_coords1, dr_coords2 = pair_sampler.generate_pair_sampling_coords(
            edges,
        )

        # Get sightline evaluations
        dd_vs1, dd_vs2 = ip.evaluate_sightlines()
        dr_vs1, dr_vs2 = ip.evaluate_sightlines()

        # Calculate pair counts from
        actual = {}
        actual['dd'] = pair_sampler.estimate_pair_counts()
        actual['dr'] = pair_sampler.estimate_pair_counts()

        # Compare to traditional counts
        ip.generate_sightlines( pair_sampler.n_tot )
        is_valid = ip.evaluate_sightlines() >  0
        coords = np.array( ip.sls )[is_valid,:]
        for count in [ 'dd', 'dr' ]:

            count_standard = two_point_autocf(
                coords,
                mins = [ -sidelength/2., -sidelength/2. ],
                maxes = [ -sidelength/2., -sidelength/2. ],
                edges = pair_sampler.spacing_edges,
                estimator = count,
            )

            npt.assert_allclose(
                actual[count],
                count_standard,
                rtol = np.sqrt( count_standard )
            )


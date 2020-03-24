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
        v_edges = np.array([ -0.5, 0.5, 1.5 ])

        # Generate some sightlines through the data for the primary data coords.
        n_data_coords = 5000
        ip.generate_sightlines( n_data_coords )
        is_valid = ip.evaluate_sightlines() >  0
        data_coords = np.array( ip.sls )[is_valid,:]

        # Generate the sampling coords
        pair_sampler = sample.PairSampler( ip.sidelength, edges, v_edges )
        dd_coords1, dd_coords2 = pair_sampler.generate_pair_sampling_coords(
            data_coords,
        )
        dr_coords1, dr_coords2 = pair_sampler.generate_pair_sampling_coords()

        # Get sightline evaluations
        vs = []
        for coords in [ dd_coords1, dd_coords2, dr_coords1, dr_coords2 ]:
            vs_bins = []
            for coords_bin in coords:
                ip.set_sightlines( coords_bin )
                vs_bins.append( ip.evaluate_sightlines() )
            vs.append( vs_bins )
        dd_vs1, dd_vs2, dr_vs1, dr_vs2 = np.array( vs )

        # Calculate pair counts from
        actual = {}
        actual['n_dd'] = pair_sampler.estimate_pair_counts( dd_vs2 )
        actual['n_dr'] = pair_sampler.estimate_pair_counts( dr_vs2 )

        # Compare to traditional counts
        ip.generate_sightlines( 1000 )
        is_valid = ip.evaluate_sightlines() >  0
        coords = np.array( ip.sls )[is_valid,:]
        for count in [ 'n_dd', 'n_dr' ]:

            count_standard, edges = stats.two_point_autocf(
                coords,
                mins = [ -ip.sidelength/2., -ip.sidelength/2. ],
                maxes = [ ip.sidelength/2., ip.sidelength/2. ],
                bins = pair_sampler.edges,
                estimator = count,
            )
            normalized_standard = count_standard / count_standard.sum()

            npt.assert_allclose(
                actual[count][:,1],
                normalized_standard,
                # atol = ( 1. / np.sqrt( count_standard ) ).max(),
                atol = 0.2
            )


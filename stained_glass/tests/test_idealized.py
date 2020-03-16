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

########################################################################
########################################################################

class TestMockObserve( unittest.TestCase ):

    def test_generate_sightlines( self ):

        ip = idealized.IdealizedProjection()

        # Generate sightlines
        n = 1000
        ip.generate_sightlines( n, seed=1234 )

        # Test
        assert ip.sl_xs.shape == ( n, )
        assert ip.sl_ys.shape == ( n, )
        assert ip.sl_xs.min() > -10.
        assert ip.sl_ys.min() > -10.
        assert ip.sl_xs.max() < 10.
        assert ip.sl_ys.max() < 10.

    ########################################################################

    def test_evaluate_sightlines( self ):

        # Setup
        ip = idealized.IdealizedProjection()
        ip.generate_sightlines( 1000, seed=1234 )
        value = 1.
        ip.add_background( value )

        # Evaluate
        vs = ip.evaluate_sightlines()

        # Test
        npt.assert_allclose( vs, np.full( ( ip.n, ), value, ) )

    ########################################################################

    def test_evaluate_sightlines_two_structs( self ):

        # Setup
        ip = idealized.IdealizedProjection()
        ip.generate_sightlines( 1000, seed=1234 )
        value = 1.
        ip.add_background( value )
        ip.add_ellipse( c=(0.,0.), a=3., value=value )

        # Evaluate
        vs = ip.evaluate_sightlines()
        is_value = np.isclose( vs, value )
        is_twice_value = np.isclose( vs, 2.*value )

        # Check
        # The number of points with that value should scale as the area of
        # the ellipse
        npt.assert_allclose(
            is_twice_value.sum() / float( ip.n ),
            ip.structs[1].area / ip.structs[0].area,
            rtol = 0.05
        )

########################################################################

class TestAddStructures( unittest.TestCase ):

    def setUp( self ):

        self.ip = idealized.IdealizedProjection()

        # Generate sightlines
        n = 1000
        self.ip.generate_sightlines( n, seed=1234 )

    ########################################################################

    def test_add_background( self ):

        # Function itself
        self.ip.add_background()

        # Check output
        npt.assert_allclose(
            self.ip.structs[0].bounds,
            ( -10., -10., 10., 10., ),
        )

    ########################################################################

    def test_add_circle( self ):

        # Function itself
        center = ( 1., 2. )
        radius = 2.
        self.ip.add_ellipse(
            center,
            radius,
        )

        # Check output
        npt.assert_allclose(
            self.ip.structs[0].area,
            np.pi * radius**2.,
            rtol = 1e-2,
        )

    ########################################################################

    def test_add_ellipse( self ):

        # Function itself
        center = ( 1., 2. )
        a = 2.
        b = 3.
        rotation = 45.
        self.ip.add_ellipse(
            center,
            a,
            b,
            rotation = rotation,
        )

        # Check output
        npt.assert_allclose(
            self.ip.structs[0].area,
            np.pi * a * b,
            rtol = 1e-2,
        )

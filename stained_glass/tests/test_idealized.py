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

class TestCoreFunctions( unittest.TestCase ):

    def test_calculate_idealized_projection( self ):

        # Setup
        ip = idealized.IdealizedProjection()
        ip.add_background()
        ip.add_ellipse( (0., 0.), 2. )

        # Actual calculation
        ip.generate_idealized_projection()

        # Check the structures
        # The background and the ellipse have the same value, so they should
        # be merged
        assert len( ip.ip ) == 1.
        assert ip.ip[0].almost_equals( ip.structs[0] )

        # Check the values
        npt.assert_allclose( np.array( ip.ip_values ), np.array([ 1., ]) )

    ########################################################################

    def test_calculate_idealized_projection_multipolygons_only( self ):

        # Setup
        ip = idealized.IdealizedProjection()
        ip.add_clumps(
            r_clump = 0.2,
            c = (0., 0.),
            r_area = 5.,
            fcov = 0.5,
        )
        ip.add_clumps(
            r_clump = 0.2,
            c = (-2., 0.),
            r_area = 5.,
            fcov = 0.5,
        )

        # Actual calculation
        ip.generate_idealized_projection()

        # Check the values
        npt.assert_allclose( np.array( ip.ip_values ), np.array([ 1., ]) )

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
        ip.add_ellipse( c=(0.,0.), a=3., value=2.*value )

        # Evaluate
        vs = ip.evaluate_sightlines( method='highest value' )
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

    def test_evaluate_sightlines_three_structs( self ):

        # Setup
        ip = idealized.IdealizedProjection()
        ip.generate_sightlines( 1000, seed=1234 )
        value = 1.
        ip.add_background( value )
        ip.add_ellipse( c=(5.,0.), a=3., value=2.*value )
        ip.add_ellipse( c=(-5.,0.), a=3., value=2.*value )

        # Evaluate
        vs = ip.evaluate_sightlines( method='highest value' )
        is_value = np.isclose( vs, value )
        is_twice_value = np.isclose( vs, 2.*value )

        # Check
        # The number of points with that value should scale as the area of
        # the ellipse
        npt.assert_allclose(
            is_twice_value.sum() / float( ip.n ),
            2. * ip.structs[1].area / ip.structs[0].area,
            rtol = 0.05
        )

    ########################################################################

    def test_evaluate_sightlines_three_structs_one_nopatch( self ):

        # Setup
        ip = idealized.IdealizedProjection()
        ip.generate_sightlines( 1000, seed=1234 )
        value = 1.
        ip.add_background( value )
        ip.add_ellipse( c=(5.,0.), a=3., value=2.*value )
        ip.add_ellipse_nopatch( c=(-5.,0.), a=3., value=2.*value )

        # Evaluate
        vs = ip.evaluate_sightlines( method='highest value' )
        is_value = np.isclose( vs, value )
        is_twice_value = np.isclose( vs, 2.*value )

        # Check
        # The number of points with that value should scale as the area of
        # the ellipse
        npt.assert_allclose(
            is_twice_value.sum() / float( ip.n ),
            2. * ip.structs[1].area / ip.structs[0].area,
            rtol = 0.05
        )

    ########################################################################

    def test_evaluate_sightlines_two_structs_add( self ):

        # Setup
        ip = idealized.IdealizedProjection()
        ip.generate_sightlines( 1000, seed=1234 )
        value = 1.
        ip.add_background( value )
        ip.add_ellipse( c=(0.,0.), a=3., value=2.*value )

        # Evaluate
        vs = ip.evaluate_sightlines( method='add' )
        is_value = np.isclose( vs, value )
        is_thrice_value = np.isclose( vs, 3.*value )

        # Check
        # The number of points with that value should scale as the area of
        # the ellipse
        npt.assert_allclose(
            is_thrice_value.sum() / float( ip.n ),
            ip.structs[1].area / ip.structs[0].area,
            rtol = 0.05
        )

    ########################################################################

    def test_evaluate_sphere( self ):

        # Evaluation method 1
        ip1 = idealized.IdealizedProjection()
        ip1.generate_sightlines( 10, seed=1234 )
        ip1.add_sphere(
            c = (0., 0.),
            r = ip1.sidelength / 2.,
            value = 1.,
            evaluate_method = 'add',
        )
        v1s = ip1.evaluate_sightlines( method='add' )

        # Evaluation method 2
        ip2 = idealized.IdealizedProjection()
        ip2.generate_sightlines( 10, seed=1234 )
        ip2.add_sphere(
            c = (0., 0.),
            r = ip2.sidelength / 2.,
            value = 1.,
            evaluate_method = 'highest value',
        )
        v2s = ip2.evaluate_sightlines( method='highest value' )

        npt.assert_allclose( v1s, v2s )

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

    ########################################################################

    def test_add_clumps( self ):

        r_clump = 0.1
        c = (0., 0.)
        r_area = 3.
        fcov = 0.5
        self.ip.add_clumps(
            r_clump,
            c,
            r_area,
            fcov,
        )

        # Check output
        npt.assert_allclose(
            self.ip.structs[0].area,
            fcov * np.pi * r_area**2.,
            rtol = 0.001,
        )

    ########################################################################

    def test_add_clumps_nopatch( self ):

        np.random.seed( 1234 )

        r_clump = 0.2
        c = (0., 0.)
        r_area = 100.
        fcov = 1.
        self.ip.add_clumps_nopatch(
            r_clump,
            c,
            r_area,
            fcov,
            verbose = True,
        )

        # Check output
        n_check = 1000
        coords = np.random.uniform(
            -self.ip.sidelength/2.,
            self.ip.sidelength/2.,
            ( n_check, 2 ),
        )
        values = self.ip.nopatch_structs[0]( coords )
        actual_fcov = values.sum()/n_check
        expected_fcov = ( fcov * np.pi * r_area**2. ) / self.ip.sidelength**2.

        #DEBUG
        import pdb; pdb.set_trace()
        npt.assert_allclose(
            actual_fcov,
            expected_fcov
        )
    
    ########################################################################

    def test_add_concentric_structures( self ):

        # Setup
        center = ( 1., 2. )
        radius = 2.
        self.ip.add_ellipse(
            center,
            radius,
            value = 5,
        )

        # Function itself
        self.ip.add_concentric_structures(
            self.ip.structs[0],
            self.ip.struct_values[0],
        )

        # Check output
        assert len( self.ip.structs ) == 4
        for i, struct in enumerate( self.ip.structs ): 
        
            npt.assert_allclose(
                struct.area,
                np.pi * ( radius + 1. * i )**2.,
                rtol = 0.01,
            )

            assert self.ip.struct_values[i] == 5 - i

    ########################################################################

    def test_add_nfw( self ):

        r_vir = 300.
        m_vir = 1e12
        c = 10.

        # Setup
        center = ( 0., 0. )
        self.ip.add_nfw(
            center,
            r_vir = r_vir,
            m_vir = m_vir,
            r_stop = r_vir,
            c = c,
        )

        assert len( self.ip.structs ) == 32

        g_c = ( np.log( 1. + c ) - c / ( 1. + c ) )**-1.
        C = np.arccos( 1./c )
        expected_edge_value = (
            ( c**2. * g_c / ( 2. * np.pi ) )
            * m_vir / r_vir**2.
            * ( 1. - ( c**2. - 1. )**-0.5 * C )
            / ( c**2. - 1. )
        )
        npt.assert_allclose(
            self.ip.struct_values[0],
            expected_edge_value
        )

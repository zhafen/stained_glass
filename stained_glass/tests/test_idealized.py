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

class TestIdealizedProjection( unittest.TestCase ):

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

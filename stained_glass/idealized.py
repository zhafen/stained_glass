#!/usr/bin/env python
'''Tools for generating idealized structures and mock observations of them.
'''

import copy
import numpy as np

from augment.augment import store_parameters

########################################################################

class IdealizedProjection( object ):

    @store_parameters
    def __init__( self, sidelength=20., c=(0,0), ):
        '''A class for creating idealized projections of physical
        distributions and mock observing them.
        '''
        pass

    ########################################################################
    # Mock Observations
    ########################################################################

    def generate_sightlines( self, n, seed=None ):
        '''Generate the sightlines that will be used for the mock observations.

        Args:
            n (int): Number of sightlines to generate.
            seed (int): Random seed to use for generating.

        Modifes:
            self.sl_xs (np.ndarray, (n,) ): X-positions of sightlines.
            self.sl_ys (np.ndarray, (n,) ): Y-positions of sightlines.
        '''
        
        if seed is not None:
            np.random.seed( seed )

        self.sl_xs = np.random.uniform(
            self.c[0] - self.sidelength/2.,
            self.c[0] + self.sidelength/2.,
            n
        )
        self.sl_ys = np.random.uniform(
            self.c[1] - self.sidelength/2.,
            self.c[1] + self.sidelength/2.,
            n
        )

    ########################################################################
    # Idealized Structure
    ########################################################################

    def add_background( self, value=1. ):
        '''Add a background distribution
        '''

        pass

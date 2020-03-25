#!/usr/bin/env python
'''Tools for generating data uniformly distributed in random configurations.
'''

import copy
import numpy as np
import scipy

from augment import store_parameters

########################################################################
########################################################################

def randoms_in_annulus(
    n_annulus,
    r_in,
    r_out,
):
    '''Generate random data spread uniformly in a circular annulus.

    Args:
        n_annulus (int):
            Number of random data that should be in the annulus.

        r_in (float):
            Inner radius. Set to 0 to generate in a circle.

        r_out (float):
            Outer radius.

    Returns:
        np.ndarray of floats (~n_annulus,2):
            Random coordinates. The number out may not equal n_annulus exactly.
    '''

    # Calculate the number of random points to draw
    area = np.pi * ( r_out**2. - r_in**2. )
    det_den = n_annulus / area
    sidelength = 2. * r_out
    area_random = sidelength**2.
    n_overall = int( det_den * area_random )

    # Create random points. We sample uniformly from a square of sidelength
    # 2*r_out, and then choose only particles in the given radial annulus
    # If we wanted to we could probably be clever and change the sampling
    # into a function of the geometry scaled by the number of points,
    # but this is the cautious option that lets the computer do the work.
    x_rand = np.random.uniform( -sidelength/2., sidelength/2., n_overall )
    y_rand = np.random.uniform( -sidelength/2., sidelength/2., n_overall )
    r_rand = np.sqrt( x_rand**2. + y_rand**2. )

    # Select valid random data
    in_r_bin_rand = ( r_in < r_rand ) & ( r_rand < r_out )
    randoms = np.array([
        x_rand[in_r_bin_rand],
        y_rand[in_r_bin_rand],
    ]).transpose()

    return randoms

########################################################################

def randoms_in_rectangle( n, sidelength, c, width=None, height=None, ):
    '''Simple function for generating randoms distributed uniformly within
    a rectangular area.

    Args:
        n (int): Number of points to generate.
        sidelength (float): Default dimensions of rectangle.
        c (tuple of floats, (2,)): Center of rectangle.
        width (float): Width, defaults to sidelength.
        height (float): Height, defaults to sidelength.

    Returns:
        np.ndarray of floats (n,2):
            2D coordinates of generated randoms.
    '''

    # Setup dimensions
    if height is None:
        height = sidelength
    if width is None:
        width = sidelength

    # Generate
    xs = np.random.uniform( c[0] - width/2., c[0] + width/2., n )
    ys = np.random.uniform( c[1] - height/2., c[1] + height/2., n )
    randoms = np.array([ xs, ys ]).transpose()

    return randoms

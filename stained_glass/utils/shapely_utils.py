
def create_curve(
        v1,
        v2,
        theta_a = 20.,
        theta_b = 40.,
        sign_a = 1.,
        sign_b = 1.,
        value = 1,
    ):
    '''Creates a curve with chosen width.

    Args:
        v1 (tuple of floats, (2,)):
            First end of the curve.

        v2 (tuple of floats, (2,)):
            Second end of the curve.

        theta_a (float):
            Angle in degrees of inner arc of the curve.

        theta_b (float):
            Angle in degrees of outer arc of the curve.

        sign_a, sign_b (float):
            What direction the curves should be facing. I can't fully
            understand the logic for this, so it may require some
            playing around.
            One tip is if sign_a == sign_b it will be a convex curve,
            and if sign_a != sign_b the arcs will be facing each other.

    Returns:
        thick_curve (shapely object)
    '''

    # Account for user error
    sign_a = np.sign( sign_a )
    sign_b = np.sign( sign_b )

    def create_circle_from_arc( v1, v2, theta, sign ):

        # Convert theta from degrees
        theta *= np.pi / 180.

        # Turn copies into arrays for easier use
        v1 = np.array( copy.copy( v1 ) )
        v2 = np.array( copy.copy( v2 ) )

        # Calculate vectors that go into finding the location
        d = v2 - v1
        d_mag = np.linalg.norm( d )
        f_mag = d_mag / ( 2. * np.tan( theta / 2. ) )

        # Calculate the direction of a vector midway between v1 and v2
        # and pointing at the center of the circle
        if np.isclose( d[0], 0. ):
            f = np.array([
                1. / np.sqrt( 1. + ( d[0] / d[1] )**2. ),
                0.,
            ])
        # Edge cases where the curve is exactly on-axis
        elif np.isclose( d[1], 0. ):
            f = np.array([
                0.,
                -1. / np.sqrt( 1. + ( d[1] / d[0] )**2. ),
            ])
        else:
            f = np.array([
                1. / np.sqrt( 1. + ( d[0] / d[1] )**2. ),
                -1. / np.sqrt( 1. + ( d[1] / d[0] )**2. ),
            ])

            # Flip the direction of f when necessary
            f[1] *= np.sign( d[0] * d[1])

        # Find the center
        c = 0.5 * ( v1 + v2 ) + sign * f_mag * f

        # Get the circle radius
        radius = d_mag / ( 2. * np.sin( theta / 2. ) )

        # Create a circle
        circ = geometry.Point( c ).buffer( radius ) 

        return circ

    # Create the circles
    circ_a = create_circle_from_arc( v1, v2, theta_a, sign_a )
    circ_b = create_circle_from_arc( v1, v2, theta_b, sign_b )

    # Create the curve (difference or intersection depending on curves)
    if np.sign( sign_a * sign_b ) > 0:
        thick_curve = circ_b.difference( circ_a )
    else:
        thick_curve = circ_b.intersection( circ_a )

    return thick_curve

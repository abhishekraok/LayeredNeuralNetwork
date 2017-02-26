def check_2d_shape(X, expected_width):
    if len(X.shape) is not 2 and (X.shape[1] is not expected_width):
        raise ValueError('Expected 2d input with width ' + str(expected_width) + ' got ' + str(X.shape))
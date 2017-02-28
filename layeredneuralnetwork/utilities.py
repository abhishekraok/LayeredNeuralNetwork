import random
import string


def check_2d_shape(X, expected_width):
    if len(X.shape) is not 2 and (X.shape[1] is not expected_width):
        raise ValueError('Expected 2d input with width ' + str(expected_width) + ' got ' + str(X.shape))


node_version_separator = '_nnv_'


def generate_random_string(length):
    """
    Generates a random string of given length consisting of ASCII letters and numbers.

    :type length: int
    :rtype: str
    """
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(length))

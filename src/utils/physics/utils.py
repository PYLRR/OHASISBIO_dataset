import numpy as np


def deg_to_m(coordinates):
    """
    Converts degrees to meters (considering equator longitude degrees).
    :param coordinates: A scalar or numpy array of degrees.
    :return: A scalar or numpy array of meters.
    """
    return coordinates * 111_111

def euclidian_distance(p1, p2):
    """ Calculates the Euclidian distance between two points.
    :param p1: First point as a list or numpy array of any dimension.
    :param p2: Second point as a list or numpy array of any dimension.
    :return: The Euclidian distance between the points.
    """
    return np.sqrt(np.sum(np.square(np.array(p1)-np.array(p2)), axis=0))
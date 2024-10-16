import geopy.distance
import numpy as np
from pyproj import Geod


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

def get_geodesic(p1, p2, step):
    """ Sample points along the geoid separating p1 and p2.
    :param p1: First point as a list or numpy array of 2 dimension.
    :param p2: Second point as a list or numpy array of 2 dimension.
    :param step: Step expected between two sampled points (in degrees).
    :return: The list of sampled points together with the step distance (in degrees) actually used.
    """
    d = geopy.distance.geodesic(p1, p2).nautical / 60
    npoints = int(np.ceil(d / step))
    geoid = Geod(ellps="WGS84")
    points = np.array(geoid.npts(p1[1], p1[0], p2[1], p2[0], npoints))  # caution : (lon, lat format)
    real_step = d / npoints
    return points[:, ::-1], real_step

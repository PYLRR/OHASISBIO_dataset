import numpy as np


def deg_to_m(coordinates):
    return coordinates * 111_000

def euclidian_distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1)-np.square(p2)))
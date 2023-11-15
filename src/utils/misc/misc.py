import os
import random

import numpy as np


def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def shuffle_lists_by_group(list_of_lists_groups, seed=0):
    for lists_group in list_of_lists_groups:
        l = len(lists_group[0])
        for _list in lists_group:
            assert len(_list) == l, "different lists of the same group have different lengths"
            np.random.seed(seed)
            np.random.shuffle(_list)
            pass
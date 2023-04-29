'''
Functions used to read from the raw data and return numpy array / gvar list of 2pt, 3pt, ratio, meff, etc. 
'''

import numpy as np


def pt2_to_meff(pt2_ls):
    """convert the 2pt correlator list to the effective mass list

    Args:
        pt2_ls (array): array of 2pt correlator

    Returns:
        array: array of effective mass
    """

    meff_ls = []
    for i in range(len(pt2_ls) - 1):
        val = np.log(pt2_ls[i]) - np.log(pt2_ls[i + 1])
        meff_ls.append(val)
    meff_ls = np.array(meff_ls)

    return meff_ls


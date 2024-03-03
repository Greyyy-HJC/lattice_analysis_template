'''
Functions used to read from the raw data and return numpy array / gvar list of 2pt, 3pt, ratio, meff, etc. 
'''

import numpy as np


def pt2_to_meff(pt2_array, boundary="periodic"):
    """
    Convert a given array of pt2 values to an array of meff values.

    Parameters:
    - pt2_array (ndarray): Array of pt2 values.
    - boundary (str, optional): Boundary condition. Can be "periodic" or "anti-periodic". Defaults to "periodic".

    Returns:
    - meff_array (ndarray): Array of meff values.

    """
    if boundary == "periodic":
        meff_array = np.arccosh( (pt2_array[2:] + pt2_array[:-2]) / (2 * pt2_array[1:-1]) )

    elif boundary == "anti-periodic":
        meff_array = np.arcsinh((pt2_array[:-2] + pt2_array[2:]) / (2 * pt2_array[1:-1]))

    return meff_array


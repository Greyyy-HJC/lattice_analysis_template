"""
Constants used in the lattice package.
"""
import numpy as np

GEV_FM = 0.1973269631  # 1 = 0.197 GeV . fm
CF = 4 / 3 # color factor
NF = 3 # number of flavors


def lat_unit_convert(val, a, Ls, dimension):
    """Convert Lattice unit to GeV / fm.

    Args:
        val (float): The value to be converted.
        a (float): The lattice spacing in fm.
        Ls (int): The lattice size in the space directions.
        dimension (str): 'P'(like P=8), 'M'(like effective mass).
    """
    if dimension == "P":
        #! m * (2pi * 0.197 / Ls / a)
        return val * 2 * np.pi * GEV_FM / Ls / a  # return in GeV

    elif dimension == "M":
        return val / a * GEV_FM  # return in GeV

    else:
        print("dimension not recognized")
        return None

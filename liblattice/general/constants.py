"""
Constants used in the lattice package.
"""
# %%
import numpy as np

GEV_FM = 0.1973269631  # 1 = 0.197 GeV . fm
CF = 4 / 3  # color factor
NF = 3  # number of flavors
CA = 3
TF = 1 / 2


def lat_unit_convert(val, a, Ls, dimension):
    """Convert Lattice unit to GeV / fm.

    Args:
        val (float): The value to be converted.
        a (float): The lattice spacing in fm.
        Ls (int): The lattice size in the space directions.
        dimension (str): 'P'(like P=8), 'M'(like effective mass).
    """
    if dimension == "P":
        #! mom * (2pi * 0.197 / Ls / a)
        return val * 2 * np.pi * GEV_FM / Ls / a  # return in GeV

    elif dimension == "M":
        return val / a * GEV_FM  # return in GeV

    else:
        print("dimension not recognized")
        return None


def beta(order=0, Nf=3):
    if order == 0:
        return 11 / 3 * CA - 4 / 3 * TF * Nf
    elif order == 1:
        return 34 / 3 * CA**2 - (20 / 3 * CA + 4 * CF) * TF * Nf
    elif order == 2:
        return (
            2857 / 54 * CA**3
            + (2 * CF**2 - 205 / 9 * CF * CA - 1415 / 27 * CA**2) * TF * Nf
            + (44 / 9 * CF + 158 / 27 * CA) * TF**2 * Nf**2
        )
    else:
        print(">>> NNNLO beta not coded.")


# n-loop alphas; mu = [GeV]
def alphas_nloop(mu, order=0, Nf=3):
    aS = 0.293 / (4 * np.pi)
    temp = 1 + aS * beta(0, Nf) * np.log((mu / 2) ** 2)

    if order == 0:
        return aS * 4 * np.pi / temp
    elif order == 1:
        return aS * 4 * np.pi / (temp + aS * beta(1, Nf) / beta(0, Nf) * np.log(temp))
    elif order == 2:
        return (
            aS
            * 4
            * np.pi
            / (
                temp
                + aS * beta(1, Nf) / beta(0, Nf) * np.log(temp)
                + aS**2
                * (
                    beta(2, Nf) / beta(0, Nf) * (1 - 1 / temp)
                    + beta(1, Nf) ** 2 / beta(0, Nf) ** 2 * (np.log(temp) / temp + 1 / temp - 1)
                )
            )
        )
    else:
        print("NNNLO running coupling not coded.")


# %%

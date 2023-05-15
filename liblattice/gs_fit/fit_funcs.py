"""
All kinds of fit functions for the gs fit.
"""

import numpy as np


#! check to make sure be consistent with the paper
def pt2_re_fcn(pt2_t, p):
    de = p["dE1"]

    val = p["re_c0"] * np.exp(-p["E0"] * pt2_t) * (1 + p["re_c1"] * np.exp(-de * pt2_t))

    return val


def pt2_im_fcn(pt2_t, p):
    de = p["dE1"]

    val = p["im_c0"] * np.exp(-p["E0"] * pt2_t) * (1 + p["im_c1"] * np.exp(-de * pt2_t))

    return val


def ra_re_fcn(ra_t, ra_tau, p):
    de = p["dE1"]

    numerator = (
        p["pdf_re"]
        + p["re_c2"] * (np.exp(-de * (ra_t - ra_tau)) + np.exp(-de * ra_tau))
        + p["re_c3"] * np.exp(-de * ra_t)
    )
    val = numerator / (1 + p["re_c1"] * np.exp(-de * ra_t))

    return val


def ra_im_fcn(ra_t, ra_tau, p):
    de = p["dE1"]

    numerator = (
        p["pdf_im"]
        + p["im_c2"] * (np.exp(-de * (ra_t - ra_tau)) + np.exp(-de * ra_tau))
        + p["im_c3"] * np.exp(-de * ra_t)
    )
    val = numerator / (1 + p["re_c1"] * np.exp(-de * ra_t)) #* note here you should also divide by the 2pt real

    return val


# * This is for the Feynman-Hellmann fit
def fh_re_fcn(fh_t, p):
    de = p["dE1"]

    val = (
        p["pdf_re"]
        + p["re_f1"] * np.exp(-de * fh_t)
        + p["re_f2"] * fh_t * np.exp(-de * fh_t)
    )

    return val


def fh_im_fcn(fh_t, p):
    de = p["dE1"]

    val = (
        p["pdf_im"]
        + p["im_f1"] * np.exp(-de * fh_t)
        + p["im_f2"] * fh_t * np.exp(-de * fh_t)
    )

    return val

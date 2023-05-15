"""
All kinds of fit functions for the z-dependence extrapolation.
"""

import numpy as np


def with_exp_re_fcn(lam_ls, p):
    """The extrapolation formular with exponential decay / finite correlation length, usually used for quasi distribution.
    Note here the input is lambda list, not z list.

    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    val = (
        p["c1"] / (lam_ls ** p["n1"]) * np.cos(np.pi / 2 * p["n1"])
        + p["c2"] / (lam_ls ** p["n2"]) * np.cos(lam_ls - np.pi / 2 * p["n2"])
    ) * np.exp(-lam_ls / p["lam0"])

    return val


def with_exp_im_fcn(lam_ls, p):
    """The extrapolation formular with exponential decay / finite correlation length, usually used for quasi distribution.
    Note here the input is lambda list, not z list.
    Note here imag part should go upward when z increase from z = 0.


    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    val = p["c1"] / (lam_ls ** p["n1"]) * np.sin(np.pi / 2 * p["n1"]) + p["c2"] / (
        lam_ls ** p["n2"]
    ) * np.sin(lam_ls - np.pi / 2 * p["n2"]) * np.exp(-lam_ls / p["lam0"])

    return val


def without_exp_re_fcn(lam_ls, p):
    """The extrapolation formular without exponential decay, usually used for light-cone distribution.
    Note here the input is lambda list, not z list.

    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    val = p["c1"] / (lam_ls ** p["n1"]) * np.cos(np.pi / 2 * p["n1"]) + p["c2"] / (
        lam_ls ** p["n2"]
    ) * np.cos(lam_ls - np.pi / 2 * p["n2"])

    return val


def without_exp_im_fcn(lam_ls, p):
    """The extrapolation formular without exponential decay, usually used for light-cone distribution.
    Note here the input is lambda list, not z list.
    Note here imag part should go upward when z increase from z = 0.


    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    val = p["c1"] / (lam_ls ** p["n1"]) * np.sin(np.pi / 2 * p["n1"]) + p["c2"] / (
        lam_ls ** p["n2"]
    ) * np.sin(lam_ls - np.pi / 2 * p["n2"])

    return val

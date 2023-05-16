"""
FT and inverse FT, with chosen convention.
"""

import numpy as np


def sum_ft(x_ls, fx_ls, delta_x, output_k):
    """FT: f(x) -> f(k), coordinate to momentum by discrete sum, produce complex numbers
    the f(x) cannot be gvar list, because of the complex calculation

    Args:
        x_ls (list): x list of f(x)
        fx_ls (list): y list of f(x)
        delta_x (float): the gap between two values in x_ls
        output_k (float): the k value to output f(k)

    Returns:
        float: f(k)
    """
    x_ls = np.array(x_ls)
    fx_ls = np.array(fx_ls)
    val = delta_x / (2 * np.pi) * np.sum(np.exp(1j * x_ls * output_k) * fx_ls)

    return val


def sum_ft_re_im(x_ls, fx_re_ls, fx_im_ls, delta_x, output_k):
    """FT: f(x) -> f(k), coordinate to momentum by discrete sum, produce real and imaginary part separately
    the f(x) can be gvar list

    Args:
        x_ls (list): x list of f(x)
        fx_re_ls (list): y list of the real part of f(x)
        fx_im_ls (list): y list of the imaginary part of f(x)
        delta_x (float): the gap between two values in x_ls
        output_k (float): the k value to output f(k)

    Returns:
        float: f(k) real and imaginary part separately
    """
    x_ls = np.array(x_ls)
    fx_re_ls = np.array(fx_re_ls)
    fx_im_ls = np.array(fx_im_ls)
    val_re = delta_x / (2 * np.pi) * np.sum(
        np.cos(x_ls * output_k) * fx_re_ls
    ) - delta_x / (2 * np.pi) * np.sum(np.sin(x_ls * output_k) * fx_im_ls)
    val_im = delta_x / (2 * np.pi) * np.sum(
        np.sin(x_ls * output_k) * fx_re_ls
    ) + delta_x / (2 * np.pi) * np.sum(np.cos(x_ls * output_k) * fx_im_ls)

    return val_re, val_im


def sum_ft_inv(k_ls, fk_ls, delta_k, output_x):
    """Inverse FT: f(k) -> f(x), momentum to coordinate by discrete sum, produce complex numbers

    Args:
        k_ls (list): k list of f(k)
        fk_ls (list): y list of f(k)
        delta_k (float): the gap between two values in k_ls
        output_x (float): the x value to output f(x)

    Returns:
        float: f(x)
    """
    k_ls = np.array(k_ls)
    fk_ls = np.array(fk_ls)
    val = delta_k * np.sum(np.exp(-1j * k_ls * output_x) * fk_ls)

    return val


def sum_ft_inv_re_im(k_ls, fk_re_ls, fk_im_ls, delta_k, output_x):
    """Inverse FT: f(k) -> f(x), momentum to coordinate by discrete sum, produce real and imaginary part separately

    Args:
        k_ls (list): k list of f(k)
        fk_re_ls (list): y list of the real part of f(k)
        fk_im_ls (list): y list of the imaginary part of f(k)
        delta_k (float): the gap between two values in k_ls
        output_x (float): the x value to output f(x)

    Returns:
        float: f(x) real and imaginary part separately
    """
    k_ls = np.array(k_ls)
    fk_re_ls = np.array(fk_re_ls)
    fk_im_ls = np.array(fk_im_ls)
    val_re = delta_k * np.sum(np.cos(k_ls * output_x) * fk_re_ls) - delta_k * np.sum(
        np.sin(k_ls * output_x) * fk_im_ls
    )
    val_im = delta_k * np.sum(np.sin(k_ls * output_x) * fk_re_ls) + delta_k * np.sum(
        np.cos(k_ls * output_x) * fk_im_ls
    )

    return val_re, val_im

"""
Do the coordinate extrapolation by fitting the large z data.
Fit a single list independently.
Only fit the z > 0 part, and then mirror the z < 0 part with symmetry.
"""

import lsqfit as lsf
import matplotlib.pyplot as plt

from liblattice.coor_to_mom.prior_setting import *
from liblattice.coor_to_mom.fit_funcs import *
from liblattice.general.plot_settings import *


extrapolated_length = 200  # todo do the extrapolation in the coordinate space till lambda = extrapolated_length

lambda_label = r"$\lambda$ = x P^z"


def z_ls_to_lam_ls(z_ls, a, Ls, mom):
    """convert z_ls to lambda_ls

    Args:
        z_ls (list): z list
        a (float): lattice spacing
        Ls (int): lattice size in the space directions
        mom (int): momentum in lattice unit, like mom = 8

    Returns:
        list: lambda list
    """
    from liblattice.general.constants import lat_unit_convert, GEV_FM

    pz = lat_unit_convert(mom, a, Ls, "P")  # in GeV
    z_array = np.array(z_ls)
    lam_array = z_array * a * pz / GEV_FM

    return list(lam_array)


def extrapolate_quasi(lam_ls, re_gv, im_gv, fit_idx_range):
    """fit and extrapolate the quasi distribution at large lambda, note here we need to concatenate the data points and extrapolated points together

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit

    Returns:
        three lists after extrapolation: lambda list, real part of quasi distribution, imag part of quasi distribution
    """
    lam_gap = abs(lam_ls[1] - lam_ls[0])  # the gap between two discrete lambda

    lam_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    lam_dic = {"re": np.array(lam_fit), "im": np.array(lam_fit)}
    pdf_dic = {
        "re": re_gv[fit_idx_range[0] : fit_idx_range[1]],
        "im": im_gv[fit_idx_range[0] : fit_idx_range[1]],
    }

    def fcn(x, p):
        val = {}
        val["re"] = with_exp_re_fcn(x["re"], p)
        val["im"] = with_exp_im_fcn(x["im"], p)

        return val

    fit_result = lsf.nonlinear_fit(
        data=(lam_dic, pdf_dic),
        prior=large_z_extrapolation_prior(),
        fcn=fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    # * two parts: data points and extrapolated points
    lam_ls_part1 = lam_ls[: fit_idx_range[0]]
    re_gv_part1 = re_gv[: fit_idx_range[0]]
    im_gv_part1 = im_gv[: fit_idx_range[0]]

    lam_ls_part2 = np.arange(lam_ls[fit_idx_range[0]], extrapolated_length, lam_gap)

    lam_dic_read = {
        "re": lam_ls_part2,
        "im": lam_ls_part2,
    }  # used to read the extrapolated points from the fit results
    re_gv_part2 = fcn(lam_dic_read, fit_result.p)["re"]
    im_gv_part2 = fcn(lam_dic_read, fit_result.p)["im"]

    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)
    extrapolated_re_gv = list(re_gv_part1) + list(re_gv_part2)
    extrapolated_im_gv = list(im_gv_part1) + list(im_gv_part2)

    return extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_result


def bf_aft_extrapolation_plot(
    lam_ls,
    re_gv,
    im_gv,
    extrapolated_lam_ls,
    extrapolated_re_gv,
    extrapolated_im_gv,
    fit_range_idx,
    title,
    ylim=None,
    save=True,
):
    """Make a comparison plot of the coordinate distribution in lambda dependence before and after extrapolation, default save to `output/plots/`.

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit
        save (bool, optional): whether save it. Defaults to True.
    """

    # * plot the real part
    plt_axes = [0.12, 0.12, 0.8, 0.8]  # left, bottom, width, height
    fs_p = {"fontsize": 13}  # font size of text, label, ticks
    ls_p = {"labelsize": 13}

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    ax.errorbar(
        lam_ls, [v.mean for v in re_gv], [v.sdev for v in re_gv], label="data", **errorb
    )
    ax.fill_between(
        extrapolated_lam_ls,
        [v.mean - v.sdev for v in extrapolated_re_gv],
        [v.mean + v.sdev for v in extrapolated_re_gv],
        alpha=0.4,
        label="extrapolated",
    )

    ax.axvline(lam_ls[fit_range_idx[0]], ymin=0, ymax=0.5, color=red, linestyle="--")
    ax.axvline(
        lam_ls[fit_range_idx[1] - 1], ymin=0, ymax=0.5, color=red, linestyle="--"
    )

    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_xlabel(lambda_label, **fs_p)
    ax.set_ylim(ylim)
    plt.title(title + "_real", **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("output/plots/" + title + "_real.pdf", transparent=True)

    # * plot the imag part
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    ax.errorbar(
        lam_ls, [v.mean for v in im_gv], [v.sdev for v in im_gv], label="data", **errorb
    )
    ax.fill_between(
        extrapolated_lam_ls,
        [v.mean - v.sdev for v in extrapolated_im_gv],
        [v.mean + v.sdev for v in extrapolated_im_gv],
        alpha=0.4,
        label="extrapolated",
    )

    ax.axvline(lam_ls[fit_range_idx[0]], ymin=0, ymax=0.5, color=red, linestyle="--")
    ax.axvline(
        lam_ls[fit_range_idx[1] - 1], ymin=0, ymax=0.5, color=red, linestyle="--"
    )

    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_xlabel(lambda_label, **fs_p)
    ax.set_ylim(ylim)
    plt.title(title + "_imag", **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("output/plots/" + title + "_imag.pdf", transparent=True)

    return

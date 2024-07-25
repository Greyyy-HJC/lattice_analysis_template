import os
import logging
import numpy as np
import gvar as gv

from liblattice.general.plot_settings import *
from liblattice.general.general_plot_funcs import errorbar_fill_between_ls_plot


def set_up_log():
    if os.path.exists("../log/bad_fit.log"):
        os.remove("../log/bad_fit.log")
    logging.basicConfig(
        level=logging.DEBUG,
        filename="../log/bad_fit.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    global fit_count, bad_fit_count
    fit_count = 0
    bad_fit_count = 0

def log_count_fit(message):
    """
    Increments the fit_count and bad_fit_count variables and logs a message if provided.

    Parameters:
    message (str): The bad fit message to be logged.

    Returns:
    None
    """
    global fit_count, bad_fit_count
    fit_count += 1

    if message:
        bad_fit_count += 1
        logging.info(message)


def plot_ratio_fit_on_data_log(
    get_ratio_data,
    ra_fit_res,
    ra_re_fcn,
    ra_im_fcn,
    px,
    py,
    pz,
    b,
    z,
    err_tsep_ls,
    fill_tsep_ls,
    Lt,
    err_tau_cut=1,
    fill_tau_cut=1,
):
    """
    Plot the ratio fit on data.

    Args:
        px (float): Momentum in the x-direction.
        py (float): Momentum in the y-direction.
        pz (float): Momentum in the z-direction.
        b (float): Impact parameter.
        z (float): Light-cone momentum fraction.
        ss_fit_res (FitResult): Fit result for the 2pt SS fit.
        err_tsep_ls (list): List of time separations for error bars.
        fill_tsep_ls (list): List of time separations for filled regions.
        Lt (int): The temporal size of the lattice, for fcn.
        err_tau_cut (int, optional): Cut for the range of tau values used for error bars. Defaults to 1.
        fill_tau_cut (int, optional): Cut for the range of tau values used for filled regions. Defaults to 1.

    Returns:
        None
    """
    from liblattice.preprocess.resampling import bs_ls_avg

    tsep_ls = [6, 8, 10, 12]
    ra_re, ra_im = get_ratio_data(px, py, pz, b, z, tsep_ls, jk_bs="bs")

    # Reshape and average the data only once.
    ra_re_avg = bs_ls_avg(ra_re.reshape(len(ra_re), -1)).reshape(
        len(tsep_ls), -1
    )  # (tsep, tau)
    ra_im_avg = bs_ls_avg(ra_im.reshape(len(ra_im), -1)).reshape(
        len(tsep_ls), -1
    )  # (tsep, tau)

    ra_re_avg_dic = {}
    ra_im_avg_dic = {}
    for id, tsep in enumerate(tsep_ls):
        ra_re_avg_dic[f"tsep_{tsep}"] = ra_re_avg[id]
        ra_im_avg_dic[f"tsep_{tsep}"] = ra_im_avg[id]

    def plot_part(part, ra_avg, ra_fcn, pdf_key):
        x_ls = []
        y_ls = []
        yerr_ls = []
        label_ls = []
        plot_style_ls = []

        for id, tsep in enumerate(err_tsep_ls):
            tau_range = np.arange(err_tau_cut, tsep + 1 - err_tau_cut)
            x_ls.append(tau_range - tsep / 2)
            y_ls.append(gv.mean(ra_avg[id, err_tau_cut : tsep + 1 - err_tau_cut]))
            yerr_ls.append(gv.sdev(ra_avg[id, err_tau_cut : tsep + 1 - err_tau_cut]))
            label_ls.append(f"tsep = {tsep}")
            plot_style_ls.append("errorbar")

        for id, tsep in enumerate(fill_tsep_ls):
            fit_tau = np.linspace(fill_tau_cut - 0.5, tsep - fill_tau_cut + 0.5, 100)
            fit_t = np.ones_like(fit_tau) * tsep
            fit_ratio = ra_fcn(fit_t, fit_tau, ra_fit_res.p, Lt)

            x_ls.append(fit_tau - tsep / 2)
            y_ls.append(gv.mean(fit_ratio))
            yerr_ls.append(gv.sdev(fit_ratio))
            label_ls.append(None)
            plot_style_ls.append("fill_between")

        band_x = np.arange(-6, 7)
        x_ls.append(band_x)
        y_ls.append(np.ones_like(band_x) * gv.mean(ra_fit_res.p[pdf_key]))
        yerr_ls.append(np.ones_like(band_x) * gv.sdev(ra_fit_res.p[pdf_key]))
        label_ls.append("fit")
        plot_style_ls.append("fill_between")

        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
        errorbar_fill_between_ls_plot(
            x_ls,
            y_ls,
            yerr_ls,
            label_ls,
            plot_style_ls,
            title=f"Ratio_fit_on_data_P{px}_{part}_b{b}_z{z}",
            save=False,
            head=ax,
        )
        plt.savefig(
            f"../log/gsfit/Ratio_fit_on_data_P{px}_{part}_b{b}_z{z}.pdf",
            transparent=True,
        )

    # Plot real part
    plot_part("real", ra_re_avg, ra_re_fcn, "pdf_re")

    # Plot imaginary part
    plot_part("imag", ra_im_avg, ra_im_fcn, "pdf_im")


def plot_sum_fit_on_data_log(
    get_sum_data,
    sum_fit_res,
    sum_re_fcn,
    sum_im_fcn,
    px,
    py,
    pz,
    b,
    z,
    err_tsep_ls,
    fill_tsep_ls,
    err_tau_cut=2,
    fill_tau_cut=2,
):
    """
    Plot the sum fit on data.

    Args:
        px (float): Momentum in the x-direction.
        py (float): Momentum in the y-direction.
        pz (float): Momentum in the z-direction.
        b (float): Impact parameter.
        z (float): Light-cone momentum fraction.
        ss_fit_res (FitResult): Fit result for the 2pt SS fit.
        err_tsep_ls (list): List of time separations for error bars.
        fill_tsep_ls (list): List of time separations for filled regions.
        err_tau_cut (int, optional): Cut for the range of tau values used for error bars. Defaults to 1.
        fill_tau_cut (int, optional): Cut for the range of tau values used for filled regions. Defaults to 1.

    Returns:
        None
    """
    from liblattice.preprocess.resampling import bs_ls_avg

    sum_re, sum_im = get_sum_data(
        px, py, pz, b, z, err_tsep_ls, jk_bs="bs", tau_cut=err_tau_cut
    )

    # Reshape and average the data only once.
    sum_re_avg = bs_ls_avg(sum_re)
    sum_im_avg = bs_ls_avg(sum_im)

    def plot_part(part, sum_avg, sum_fcn):
        fit_t = np.linspace(fill_tsep_ls[0] - 0.5, fill_tsep_ls[-1] + 0.5, 100)
        fit_sum = sum_fcn(fit_t, fill_tau_cut, sum_fit_res.p)

        x_ls = [np.array(err_tsep_ls), fit_t]
        y_ls = [gv.mean(sum_avg), gv.mean(fit_sum)]
        yerr_ls = [gv.sdev(sum_avg), gv.sdev(fit_sum)]
        label_ls = ["data", "fit"]
        plot_style_ls = ["errorbar", "fill_between"]

        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
        errorbar_fill_between_ls_plot(
            x_ls,
            y_ls,
            yerr_ls,
            label_ls,
            plot_style_ls,
            title=f"Sum_fit_on_data_P{px}_{part}_b{b}_z{z}",
            save=False,
            head=ax,
        )
        plt.savefig(
            f"../log/gsfit/Sum_fit_on_data_P{px}_{part}_b{b}_z{z}.pdf", transparent=True
        )

    # Plot real part
    plot_part("real", sum_re_avg, sum_re_fcn)

    # Plot imaginary part
    plot_part("imag", sum_im_avg, sum_im_fcn)

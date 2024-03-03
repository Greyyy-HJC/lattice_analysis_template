# %%
import logging

# set logging config
logging.basicConfig(filename='../log/bad_fit.log', level=logging.INFO, format='%(asctime)s %(message)s')

import numpy as np
import gvar as gv
import lsqfit as lsf

from read_data import get_2pt_data, get_ratio_data
from gs_fit_2pt import single_2pt_fit
from liblattice.gs_fit.prior_setting import two_state_fit
from liblattice.preprocess.resampling import bs_ls_avg
from liblattice.gs_fit.fit_funcs import ra_re_fcn, ra_im_fcn
from liblattice.general.general_plot_funcs import errorbar_ls_plot, errorbar_fill_between_ls_plot


def single_ra_fit(ra_re_avg_dic, ra_im_avg_dic, b, z, tsep_ls, tau_cut, ss_fit_res, Ls):
    """
    Perform a single ratio fit.

    Args:
        ra_re_avg_dic (dict): Dictionary containing the real part of the ratio average for different tseps. Keys like 'tsep_6'.
        ra_im_avg_dic (dict): Dictionary containing the imaginary part of the ratio average for different tseps. Keys like 'tsep_6'.
        b (float): The value of b.
        z (float): The value of z.
        tsep_ls (list): List of tseps.
        tau_cut (int): The value of tau cut.
        ss_fit_res (object): Object containing the 2pt fit results.
        Ls (int): The value of Ls.

    Returns:
        object: Object containing the fit results.
    """
    def ra_fcn(x, p):
        ra_t, ra_tau = x
        return {'re': ra_re_fcn(ra_t, ra_tau, p, Ls), 'im': ra_im_fcn(ra_t, ra_tau, p, Ls)}

    # Set 2pt fit results as priors
    priors = two_state_fit()
    priors.update({key: ss_fit_res.p[key] for key in ['E0', 'log(dE1)', 're_z0', 're_z1']})

    # Prepare data for fit
    temp_t, temp_tau, ra_fit_re, ra_fit_im = [], [], [], []
    for tsep in tsep_ls:
        for tau in range(tau_cut, tsep + 1 - tau_cut):
            temp_t.append(tsep)
            temp_tau.append(tau)
            ra_fit_re.append(ra_re_avg_dic[f'tsep_{tsep}'][tau])
            ra_fit_im.append(ra_im_avg_dic[f'tsep_{tsep}'][tau])

    # Perform the fit
    tsep_tau_ls = [np.array(temp_t), np.array(temp_tau)]
    ra_fit = {'re': ra_fit_re, 'im': ra_fit_im}
    ra_fit_res = lsf.nonlinear_fit(data=(tsep_tau_ls, ra_fit), prior=priors, fcn=ra_fcn, maxit=10000)

    # Check the quality of the fit
    if ra_fit_res.Q < 0.05:
        logging.info(f'>>> Bad fit for z = {z}, b = {b} with Q = {ra_fit_res.Q}')

    return ra_fit_res


# %%
if __name__ == "__main__":

    #todo To adjust
    # Constants and parameters
    Ls = 48
    px = py = 4
    pz = 0
    b_array = np.arange(2, 12, 2)
    z_array = np.arange(25)
    tau_cut = 3
    tsep_ls = [6, 8, 10, 12]
    tsep_ls_str = ''.join([str(t) for t in tsep_ls])
    ifdump = True
    #todo



    # * 2pt fit
    # Priors for two-state fit
    priors = two_state_fit()

    # Retrieve 2pt data (real parts only, as imaginary parts are not used)
    pt2_ss_re = get_2pt_data('SS', px, py, pz, jk_bs="bs")[0]
    pt2_sp_re = get_2pt_data('SP', px, py, pz, jk_bs="bs")[0]

    # Average over bootstrap samples
    pt2_ss_avg = bs_ls_avg(pt2_ss_re)
    pt2_sp_avg = bs_ls_avg(pt2_sp_re)

    # Perform single 2pt fits
    ss_fit_res, sp_fit_res = single_2pt_fit(pt2_ss_avg, pt2_sp_avg, Ls, priors, tmin=3, tmax=13)


    # * ratio fit
    # Precompute these lists as they are used multiple times.
    x_ls = [z_array for _ in b_array]
    label_ls = [f'b = {b}' for b in b_array]

    # Initialize lists to store results.
    re_bz_ls = []
    im_bz_ls = []

    for b in b_array:
        # Lists to store intermediate results for each 'b'.
        re_ls = []
        im_ls = []

        for z in z_array:
            # Get ratio data for the current 'b' and 'z'.
            ra_re, ra_im = get_ratio_data(px, py, pz, b, z, tsep_ls, jk_bs="bs")

            # Average over bootstraps.
            ra_re_avg = bs_ls_avg(ra_re.reshape(len(ra_re), -1)).reshape(len(tsep_ls), -1)
            ra_im_avg = bs_ls_avg(ra_im.reshape(len(ra_im), -1)).reshape(len(tsep_ls), -1)

            ra_re_avg_dic = {}
            ra_im_avg_dic = {}
            for id, tsep in enumerate(tsep_ls):
                ra_re_avg_dic[f'tsep_{tsep}'] = ra_re_avg[id]
                ra_im_avg_dic[f'tsep_{tsep}'] = ra_im_avg[id]

            # Fit the averaged data.
            ra_fit_res = single_ra_fit(ra_re_avg_dic, ra_im_avg_dic, b, z, tsep_ls, tau_cut, ss_fit_res, Ls)
            
            # Append the fit parameters to the lists.
            re_ls.append(ra_fit_res.p['pdf_re'])
            im_ls.append(ra_fit_res.p['pdf_im'])

        # Store intermediate results in the main lists.
        re_bz_ls.append(re_ls)
        im_bz_ls.append(im_ls)

    # * plot z-dependence
    # Define a function to plot error bars.
    def plot_errorbars(y_data, suffix):
        y_ls = [gv.mean(ls) for ls in y_data]
        yerr_ls = [gv.sdev(ls) for ls in y_data]
        errorbar_ls_plot(x_ls, y_ls, yerr_ls, label_ls, title=f'Ratio_fit_mixb_{tsep_ls_str}_P{px}_cut{tau_cut}_{suffix}', save=True)

    # Plot real and imaginary parts using the defined function.
    plot_errorbars(re_bz_ls, "real")
    plot_errorbars(im_bz_ls, "imag")


    # * Dump data conditionally.
    if ifdump:
        dump_dic = {}
        for i, b in enumerate(b_array):
            dump_dic[f'b={b}_z_array'] = z_array
            dump_dic[f'b={b}_re'] = np.array(re_bz_ls[i])
            dump_dic[f'b={b}_im'] = np.array(im_bz_ls[i])

        gv.dump(dump_dic, f'../output/dump/z_dep_ratio_fit_{tsep_ls_str}_P{px}_cut{tau_cut}_mixb.dat')


    # %%
    #! plot to compare the ratio fit and sum fit
    if True:
        # * load data
        ratio_fit = gv.load(f'../cache/z_dep_ratio_fit_681012_P4_cut3_mixb.dat')
        sum_fit = gv.load(f'../cache/z_dep_sum_fit_81012_P4_cut3_mixb.dat')

        # * plot
        for b in range(2, 12, 2):
            z_array = ratio_fit[f'b={b}_z_array']
            re_ratio = ratio_fit[f'b={b}_re']
            im_ratio = ratio_fit[f'b={b}_im']
            re_sum = sum_fit[f'b={b}_re']
            im_sum = sum_fit[f'b={b}_im']

            x_ls = [z_array for _ in range(2)]
            y_ls = [gv.mean(re_ratio), gv.mean(re_sum)]
            yerr_ls = [gv.sdev(re_ratio), gv.sdev(re_sum)]
            label_ls = ['ratio', 'sum']

            errorbar_ls_plot(x_ls, y_ls, yerr_ls, label_ls, title='Ratio_vs_sum_b{}_real'.format(b), save=True)

            x_ls = [z_array for _ in range(2)]
            y_ls = [gv.mean(im_ratio), gv.mean(im_sum)]
            yerr_ls = [gv.sdev(im_ratio), gv.sdev(im_sum)]
            label_ls = ['ratio', 'sum']

            errorbar_ls_plot(x_ls, y_ls, yerr_ls, label_ls, title='Ratio_vs_sum_b{}_imag'.format(b), save=True)

    # %%
    #! plot ratio fit results on data points
    if True:
        def plot_ra_fit_on_data(px, py, pz, b, z, ss_fit_res, err_tsep_ls, fill_tsep_ls, Ls, err_tau_cut=1, fill_tau_cut=1):
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
                Ls (list): List of lattice sizes.
                err_tau_cut (int, optional): Cut for the range of tau values used for error bars. Defaults to 1.
                fill_tau_cut (int, optional): Cut for the range of tau values used for filled regions. Defaults to 1.

            Returns:
                None
            """
            tsep_ls = [6, 8, 10, 12]
            ra_re, ra_im = get_ratio_data(px, py, pz, b, z, tsep_ls, jk_bs="bs")

            # Reshape and average the data only once.
            ra_re_avg = bs_ls_avg(ra_re.reshape(len(ra_re), -1)).reshape(len(tsep_ls), -1)  # (tsep, tau)
            ra_im_avg = bs_ls_avg(ra_im.reshape(len(ra_im), -1)).reshape(len(tsep_ls), -1)  # (tsep, tau)

            ra_re_avg_dic = {}
            ra_im_avg_dic = {}
            for id, tsep in enumerate(tsep_ls):
                ra_re_avg_dic[f'tsep_{tsep}'] = ra_re_avg[id]
                ra_im_avg_dic[f'tsep_{tsep}'] = ra_im_avg[id]

            ra_fit_res = single_ra_fit(ra_re_avg_dic, ra_im_avg_dic, b, z, fill_tsep_ls, fill_tau_cut, ss_fit_res, Ls)

            def plot_part(part, ra_avg, ra_fcn, pdf_key):
                x_ls = []
                y_ls = []
                yerr_ls = []
                label_ls = []
                plot_style_ls = []

                for id, tsep in enumerate(err_tsep_ls):
                    tau_range = np.arange(err_tau_cut, tsep + 1 - err_tau_cut)
                    x_ls.append(tau_range - tsep / 2)
                    y_ls.append(gv.mean(ra_avg[id, err_tau_cut:tsep + 1 - err_tau_cut]))
                    yerr_ls.append(gv.sdev(ra_avg[id, err_tau_cut:tsep + 1 - err_tau_cut]))
                    label_ls.append(f'tsep = {tsep}')
                    plot_style_ls.append('errorbar')

                for id, tsep in enumerate(fill_tsep_ls):
                    fit_tau = np.linspace(fill_tau_cut - 0.5, tsep - fill_tau_cut + 0.5, 100)
                    fit_t = np.ones_like(fit_tau) * tsep
                    fit_ratio = ra_fcn(fit_t, fit_tau, ra_fit_res.p, Ls)

                    x_ls.append(fit_tau - tsep / 2)
                    y_ls.append(gv.mean(fit_ratio))
                    yerr_ls.append(gv.sdev(fit_ratio))
                    label_ls.append(None)
                    plot_style_ls.append('fill_between')

                band_x = np.arange(-6, 7)
                x_ls.append(band_x)
                y_ls.append(np.ones_like(band_x) * gv.mean(ra_fit_res.p[pdf_key]))
                yerr_ls.append(np.ones_like(band_x) * gv.sdev(ra_fit_res.p[pdf_key]))
                label_ls.append('fit')
                plot_style_ls.append('fill_between')

                errorbar_fill_between_ls_plot(
                    x_ls, y_ls, yerr_ls, label_ls, plot_style_ls,
                    title=f'Ratio_fit_on_data_P{px}_{part}_b{b}_z{z}', save=True
                )

            # Plot real part
            plot_part('real', ra_re_avg, ra_re_fcn, 'pdf_re')

            # Plot imaginary part
            plot_part('imag', ra_im_avg, ra_im_fcn, 'pdf_im')


        px = py = 5
        pz = 0
        b = 0
        z = 5
        err_tsep_ls = [6, 8, 10]
        fill_tsep_ls = [8, 10, 12]


        plot_ra_fit_on_data(px, py, pz, b, z, ss_fit_res, err_tsep_ls, fill_tsep_ls, Ls=48, err_tau_cut=1, fill_tau_cut=3)

# %%

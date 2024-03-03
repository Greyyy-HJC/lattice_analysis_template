# %%
import logging

# set logging config
logging.basicConfig(filename='../log/bad_fit.log', level=logging.INFO, format='%(asctime)s %(message)s')

import numpy as np
import gvar as gv
import lsqfit as lsf

from read_data import get_sum_data
from liblattice.preprocess.resampling import bs_ls_avg
from liblattice.gs_fit.fit_funcs import sum_re_fcn, sum_im_fcn
from liblattice.gs_fit.prior_setting import summation_fit
from liblattice.general.general_plot_funcs import errorbar_ls_plot

# b range: 0 - 24
# z range: 0 - 24


def single_sum_fit(sum_re_avg, sum_im_avg, b, z, tsep_ls, tau_cut, priors, ifplot=False):
    """
    Fits the sum of real and imaginary parts of a function to the given data.

    Parameters:
    sum_re_avg (array-like): The average of the real part of the sum.
    sum_im_avg (array-like): The average of the imaginary part of the sum.
    b (float): The value of b.
    z (float): The value of z.
    tsep_ls (array-like): The list of tsep values.
    tau_cut (float): The value of tau_cut.
    priors (dict): The dictionary of prior values for the fit parameters.
    ifplot (bool, optional): Whether to plot the fit results. Defaults to False.

    Returns:
    fit_sum_res (object): The result of the fit.

    """
    # * fit function
    def fcn(x, p):
        t = x['re']
        re = sum_re_fcn(t, tau_cut, p)
        im = sum_im_fcn(t, tau_cut, p)
        val = {'re': re, 'im': im}
        return val

    x_dic = {'re': np.array(tsep_ls), 'im': np.array(tsep_ls)}
    y_dic = {'re': sum_re_avg, 'im': sum_im_avg}
    fit_sum_res = lsf.nonlinear_fit(data=(x_dic, y_dic), prior=priors, fcn=fcn, maxit=10000)

    if fit_sum_res.Q < 0.05:
        logging.info(f'>>> Bad sum fit for z = {z}, b = {b} with Q = {fit_sum_res.Q}')

    # Plotting should be handled outside or after the condition checks to streamline the fitting process
    # if ifplot:
    #     plot_fit_results(tsep_ls, sum_re_avg, fit_sum_res, fcn, title=f"sum_re_fit_b{b}_z{z}", save=False)

    return fit_sum_res

# %%
if __name__ == "__main__":

    #todo To adjust
    # Constants and parameters
    px = py = 4
    pz = 0
    b_array = np.arange(2, 12, 2)
    z_array = np.arange(25)
    tau_cut = 3
    tsep_ls = [8, 10, 12]
    tsep_ls_str = ''.join([str(tsep) for tsep in tsep_ls])
    ifdump = True
    #todo


    # * prior setting
    priors = summation_fit()

    # Initialize lists to store real and imaginary parts
    re_bz_ls = []
    im_bz_ls = []

    # Precompute z_array for all b values to avoid redundant calculations
    x_ls = [z_array for _ in b_array]

    # Loop over b_array only once and compute mean and standard deviation on the fly
    y_re_ls = []
    yerr_re_ls = []
    y_im_ls = []
    yerr_im_ls = []
    label_ls = []

    for i, b in enumerate(b_array):
        re_ls = []
        im_ls = []
        for z in z_array:
            # * data
            sum_real, sum_imag = get_sum_data(px, py, pz, b, z, tsep_ls, jk_bs="bs", tau_cut=tau_cut)

            sum_re_avg = bs_ls_avg(sum_real)
            sum_im_avg = bs_ls_avg(sum_imag)

            fit_sum_res = single_sum_fit(sum_re_avg, sum_im_avg, b, z, tsep_ls, tau_cut, priors, ifplot=False)
            re_ls.append(fit_sum_res.p['pdf_re'])
            im_ls.append(fit_sum_res.p['pdf_im'])
        
        # Add computed lists to the main list
        re_bz_ls.append(re_ls)
        im_bz_ls.append(im_ls)
        
        # Calculate mean and standard deviation for both real and imaginary parts
        y_re_ls.append(gv.mean(re_ls))
        yerr_re_ls.append(gv.sdev(re_ls))
        y_im_ls.append(gv.mean(im_ls))
        yerr_im_ls.append(gv.sdev(im_ls))
        
        # Create labels
        label_ls.append(f'b = {b}')

    # Plotting the real part
    errorbar_ls_plot(x_ls, y_re_ls, yerr_re_ls, label_ls,
                    title=f'Sum_fit_mixb_{tsep_ls_str}_P{px}_cut{tau_cut}_real', save=True)

    # Plotting the imaginary part
    errorbar_ls_plot(x_ls, y_im_ls, yerr_im_ls, label_ls,
                    title=f'Sum_fit_mixb_{tsep_ls_str}_P{px}_cut{tau_cut}_imag', save=True)

    # Dump data if required
    if ifdump:
        dump_dic = {}
        for i, b in enumerate(b_array):
            dump_dic[f'b={b}_z_array'] = z_array
            dump_dic[f'b={b}_re'] = np.array(re_bz_ls[i])
            dump_dic[f'b={b}_im'] = np.array(im_bz_ls[i])

        gv.dump(dump_dic, f'../output/dump/z_dep_sum_fit_{tsep_ls_str}_P{px}_cut{tau_cut}_mixb.dat')




# %%

# %%
import numpy as np
import lsqfit as lsf

from liblattice.gs_fit.log import log_count_fit
from liblattice.gs_fit.fit_funcs import sum_re_fcn, sum_im_fcn
from liblattice.gs_fit.prior_setting import summation_fit


def single_sum_fit(sum_re_avg, sum_im_avg, tsep_ls, tau_cut, pbz_label):
    """
    Perform a single sum fit.

    Args:
        sum_re_avg (array-like): The real part of the sum to be fitted.
        sum_im_avg (array-like): The imaginary part of the sum to be fitted.
        tsep_ls (array-like): The list of time separations.
        tau_cut (float): The cutoff value for tau.
        pbz_label (dict): A dictionary containing the labels for px, py, pz, b, and z.

    Returns:
        sum_fit_res (object): The result of the sum fit.

    Raises:
        None

    """
    priors = summation_fit()

    px = pbz_label["px"]
    py = pbz_label["py"]
    pz = pbz_label["pz"]
    b = pbz_label["b"]
    z = pbz_label["z"]

    # * fit function
    def fcn(x, p):
        t = x["re"]
        re = sum_re_fcn(t, tau_cut, p)
        im = sum_im_fcn(t, tau_cut, p)
        val = {"re": re, "im": im}
        return val

    x_dic = {"re": np.array(tsep_ls), "im": np.array(tsep_ls)}
    y_dic = {"re": sum_re_avg, "im": sum_im_avg}
    sum_fit_res = lsf.nonlinear_fit(
        data=(x_dic, y_dic), prior=priors, fcn=fcn, maxit=10000
    )

    if sum_fit_res.Q < 0.05:
        log_count_fit(
            f">>> Bad sum fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {sum_fit_res.Q}"
        )
    else:
        log_count_fit()

    return sum_fit_res

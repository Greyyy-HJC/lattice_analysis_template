import numpy as np
import lsqfit as lsf

from liblattice.gs_fit.log import log_count_fit
from liblattice.gs_fit.fit_funcs import pt2_re_fcn
from liblattice.gs_fit.prior_setting import two_state_fit


def single_2pt_fit(pt2_avg, tmin, tmax, Ls, label=None):
    """
    Perform a single 2-point fit on the given data.

    Args:
        pt2_avg (gvar list): The averaged 2-point data.
        tmin (int): The minimum time value for the fit range.
        tmax (int): The maximum time value for the fit range.
        Ls (int): The size of the lattice.
        label (str, optional): A label for the fit. Defaults to None.

    Returns:
        FitResult: The result of the fit.

    Raises:
        None

    """

    priors = two_state_fit()

    def fcn(t, p):
        return pt2_re_fcn(t, p, Ls)

    # Compute the range only once, outside of the loop
    t_range = np.arange(tmin, tmax)

    # Normalize the 2pt data only once for each dataset
    normalization_factor = pt2_avg[0]
    fit_pt2 = pt2_avg[tmin:tmax] / normalization_factor

    fit_res = lsf.nonlinear_fit(
        data=(t_range, fit_pt2), prior=priors, fcn=fcn, maxit=10000
    )

    if fit_res.Q < 0.05:
        log_count_fit(f">>> Bad 2pt {label} fit with Q = {fit_res.Q}")
    else:
        log_count_fit()

    return fit_res

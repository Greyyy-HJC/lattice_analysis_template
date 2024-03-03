# %%
import logging

# set logging config
logging.basicConfig(filename='../log/bad_fit.log', level=logging.INFO, format='%(asctime)s %(message)s')

import numpy as np
import gvar as gv
import lsqfit as lsf

from read_data import get_2pt_data
from liblattice.preprocess.resampling import bs_ls_avg
from liblattice.gs_fit.fit_funcs import pt2_re_fcn
from liblattice.gs_fit.prior_setting import two_state_fit
from liblattice.general.general_plot_funcs import errorbar_ls_plot


# b range: 0 - 24
# z range: 0 - 24

Ls = 48
priors = two_state_fit()


def single_2pt_fit(pt2_ss_avg, pt2_sp_avg, Ls, priors, tmin, tmax):
    """
    Perform a single 2-point fit for the given data.

    Args:
        pt2_ss_avg (1D numpy.ndarray): The averaged 2-point data for the ss dataset.
        pt2_sp_avg (1D numpy.ndarray): The averaged 2-point data for the sp dataset.
        Ls (float): The size of the lattice in the spacial direction.
        priors (list): The prior values for the fit parameters.
        tmin (int): The minimum time value for the fit range.
        tmax (int): The maximum time value for the fit range.

    Returns:
        tuple: A tuple containing the fit results for the ss and sp datasets.
    """

    def fcn(t, p):
        return pt2_re_fcn(t, p, Ls)

    # Compute the range only once, outside of the loop
    t_range = np.arange(tmin, tmax)

    # Normalize the 2pt data only once for each dataset
    normalization_factor_ss = pt2_ss_avg[0]
    normalization_factor_sp = pt2_sp_avg[0]
    fit_pt2_ss = pt2_ss_avg[tmin:tmax] / normalization_factor_ss
    fit_pt2_sp = pt2_sp_avg[tmin:tmax] / normalization_factor_sp

    # Define a common function to perform the fit and check the result
    def perform_fit_and_check(data, label):
        fit_res = lsf.nonlinear_fit(
            data=(t_range, data), prior=priors, fcn=fcn, maxit=10000
        )
        if fit_res.Q < 0.05:
            logging.info(f">>> Bad 2pt {label} fit with Q = {fit_res.Q}")

        return fit_res

    # Perform fits for ss and sp datasets
    ss_fit_res = perform_fit_and_check(fit_pt2_ss, "ss")
    sp_fit_res = perform_fit_and_check(fit_pt2_sp, "sp")

    return ss_fit_res, sp_fit_res


# %%
#! 2pt tmin stability
if __name__ == "__main__":
    from liblattice.preprocess.dispersion import disp_relation_plot

    if True:
        px = py = 5
        pz = 0
        tmin_ls = np.arange(1, 12)
        tmax = 13

        # Retrieve and average the data once outside the loop
        pt2_ss_re, _ = get_2pt_data("SS", px, py, pz, jk_bs="bs")
        pt2_sp_re, _ = get_2pt_data("SP", px, py, pz, jk_bs="bs")
        pt2_ss_avg = bs_ls_avg(pt2_ss_re)
        pt2_sp_avg = bs_ls_avg(pt2_sp_re)

        # Initialize lists to store fit results
        fit_results = {
            "ss": {"e0": [], "e1": [], "Q": []},
            "sp": {"e0": [], "e1": [], "Q": []},
        }

        # Perform fits for each tmin value
        for tmin in tmin_ls:
            ss_fit_res, sp_fit_res = single_2pt_fit(
                pt2_ss_avg, pt2_sp_avg, Ls, priors, tmin, tmax
            )

            # Append results to corresponding lists
            fit_results["ss"]["e0"].append(ss_fit_res.p["E0"])
            fit_results["ss"]["e1"].append(ss_fit_res.p["E0"] + ss_fit_res.p["dE1"])
            fit_results["ss"]["Q"].append(ss_fit_res.Q)

            fit_results["sp"]["e0"].append(sp_fit_res.p["E0"])
            fit_results["sp"]["e1"].append(sp_fit_res.p["E0"] + sp_fit_res.p["dE1"])
            fit_results["sp"]["Q"].append(sp_fit_res.Q)

        # Define a function to plot the results
        def plot_energy_levels(include_e1=False):
            """
            Plot the energy levels.

            Args:
                include_e1 (bool, optional): Whether to include the E1 energy level. Defaults to False.
            """
            x_ls = [tmin_ls] * (4 if include_e1 else 2)
            y_ls = [gv.mean(fit_results["ss"]["e0"]), gv.mean(fit_results["sp"]["e0"])]
            yerr_ls = [
                gv.sdev(fit_results["ss"]["e0"]),
                gv.sdev(fit_results["sp"]["e0"]),
            ]
            label_ls = [f"E0 SS", f"E0 SP"]

            if include_e1:
                y_ls.append(gv.mean(fit_results["ss"]["e1"]))
                y_ls.append(gv.mean(fit_results["sp"]["e1"]))
                yerr_ls.append(gv.sdev(fit_results["ss"]["e1"]))
                yerr_ls.append(gv.sdev(fit_results["sp"]["e1"]))
                label_ls.append(f"E1 SS")
                label_ls.append(f"E1 SP")

            ylim = [0.25, 1.2] if not include_e1 else [0.25, 2]
            title = f"E{'0_and_E1' if include_e1 else '0'}_P{px}"
            errorbar_ls_plot(
                x_ls, y_ls, yerr_ls, label_ls, title=title, save=True, ylim=ylim
            )

        # Plot energy levels
        plot_energy_levels(include_e1=True)
        plot_energy_levels(include_e1=False)

    #! 2pt dispersion relation
    if False:
        a = 0.06  # lattice spacing in fm
        Ls = 48  # lattice size in spatial direction

        tmin = 3
        tmax = 13
        pz = 0

        # Initialize lists to store momentum and energy levels
        mom_ls = []
        e0_ss_ls = []
        e0_sp_ls = []

        # Perform data retrieval and fitting in a single loop
        for px in range(2, 7):
            py = px
            pnorm = np.sqrt(px**2 + py**2)

            pt2_ss_re, _ = get_2pt_data("SS", px, py, pz, jk_bs="bs")
            pt2_sp_re, _ = get_2pt_data("SP", px, py, pz, jk_bs="bs")
            pt2_ss_avg = bs_ls_avg(pt2_ss_re)
            pt2_sp_avg = bs_ls_avg(pt2_sp_re)

            ss_fit_res, sp_fit_res = single_2pt_fit(
                pt2_ss_avg, pt2_sp_avg, Ls, priors, tmin, tmax
            )

            e0_ss_ls.append(ss_fit_res.p["E0"])
            e0_sp_ls.append(sp_fit_res.p["E0"])
            mom_ls.append(pnorm)

        disp_relation_plot(
            a,
            Ls,
            mom_ls,
            e0_ss_ls,
            title="Dispersion relation SS",
            save=True,
            m0=0.3,
            ylim=[0, 4],
        )

        # disp_relation_plot(
        #     a, Ls, mom_ls, e0_sp_ls, title="Dispersion relation SP", save=True, m0=0.3, ylim=[0, 4]
        # )
# %%

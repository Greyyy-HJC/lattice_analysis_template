# %%
import gvar as gv

from liblattice.coor_to_mom.extrapolation_fit import *

ready_for_extrapolation = gv.load("debug/ready_for_extrapolation.pkl")

for b in range(1, 6):
    temp = ready_for_extrapolation['b{}'.format(b)]

    lam_ls = z_ls_to_lam_ls(temp['z'], a=0.12, Ls=48, mom=8)
    re_gv = temp['re']
    im_gv = temp['im']

    fit_idx_range = [8, 13]

    extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_result = extrapolation_quasi(lam_ls, re_gv, im_gv, fit_idx_range)

    plot_lam_ls = extrapolated_lam_ls[fit_idx_range[0]:20]
    plot_re_gv = extrapolated_re_gv[fit_idx_range[0]:20]
    plot_im_gv = extrapolated_im_gv[fit_idx_range[0]:20]

    bf_aft_extrapolation_plot(lam_ls, re_gv, im_gv, plot_lam_ls, plot_re_gv, plot_im_gv, fit_idx_range, title="220t_mom8_b{}_extrapolation".format(b), ylim=[-0.5, 1.5])
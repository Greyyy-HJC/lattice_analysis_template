# %%
import gvar as gv

from liblattice.coor_to_mom.extrapolation_fit import *

ready_for_extrapolation = gv.load("debug/ready_for_extrapolation.pkl")


aft_ex_collect = gv.BufferDict()
for b in range(1, 6):
    temp = ready_for_extrapolation['b{}'.format(b)]

    lam_ls = z_ls_to_lam_ls(temp['z'], a=0.12, Ls=48, mom=8)
    re_gv = temp['re']
    im_gv = temp['im']

    fit_idx_range = [8, 13]

    extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_result = extrapolate_quasi(lam_ls, re_gv, im_gv, fit_idx_range)

    plot_lam_ls = extrapolated_lam_ls[fit_idx_range[0]:20]
    plot_re_gv = extrapolated_re_gv[fit_idx_range[0]:20]
    plot_im_gv = extrapolated_im_gv[fit_idx_range[0]:20]

    # bf_aft_extrapolation_plot(lam_ls, re_gv, im_gv, plot_lam_ls, plot_re_gv, plot_im_gv, fit_idx_range, title="220t_mom8_b{}_extrapolation".format(b), ylim=[-0.5, 1.5])

    # fill the lambda < 0 part with symmetry
    aft_ex_lam_ls = np.concatenate(([-l for l in extrapolated_lam_ls[:0:-1]], extrapolated_lam_ls[:]))
    aft_ex_re_gv = np.concatenate((extrapolated_re_gv[:0:-1], extrapolated_re_gv[:]))
    aft_ex_im_gv = np.concatenate(([v for v in extrapolated_im_gv[:0:-1]], [-v for v in extrapolated_im_gv[:]]))

    temp = {'lam': aft_ex_lam_ls, 're': aft_ex_re_gv, 'im': aft_ex_im_gv}

    aft_ex_collect['b{}'.format(b)] = temp

# plt.show()




# do the FT
from liblattice.coor_to_mom.fourier_transform import *

x_ls = np.linspace(-1.5, 1.5, 100)

mom_beam_re = {}
mom_beam_im = {}

for b in range(1, 6):
    lam_ls = aft_ex_collect['b{}'.format(b)]['lam']
    fx_re_ls = aft_ex_collect['b{}'.format(b)]['re']
    fx_im_ls = aft_ex_collect['b{}'.format(b)]['im']
    delta_lam = lam_ls[1] - lam_ls[0]

    temp_re = []
    temp_im = []
    for x in x_ls:
        val_re, val_im = sum_ft_re_im(lam_ls, fx_re_ls, fx_im_ls, delta_lam, x)
        temp_re.append(val_re)
        temp_im.append(val_im)

    mom_beam_re['b{}'.format(b)] = temp_re
    mom_beam_im['b{}'.format(b)] = temp_im

from liblattice.general.general_plot_funcs import *
fill_between_ls_plot([x_ls for b in range(1,6)], [gv.mean(mom_beam_re['b{}'.format(b)]) for b in range(1,6)], [gv.sdev(mom_beam_re['b{}'.format(b)]) for b in range(1,6)], label_ls=['b={}'.format(b) for b in range(1, 6)], title="220t_mom8_beam_bmix_FT_re", ylim=[-0.5, 2.5])


#! beam function to quasi-TMDPDF, dividing soft function

from soft_class import *

b_ls = np.arange(1, 6)
soft_dic = read_soft_dic(b_ls, soft_mom=2)

# print(soft_dic)

mom_quasi_re = {}

for b in range(1, 6):
    temp = np.array( mom_beam_re['b{}'.format(b)] )
    mom_quasi_re['b{}'.format(b)] = list( temp * np.sqrt( soft_dic['b_{}'.format(b)] ) )


fill_between_ls_plot([x_ls for b in range(1,6)], [gv.mean(mom_quasi_re['b{}'.format(b)]) for b in range(1,6)], [gv.sdev(mom_quasi_re['b{}'.format(b)]) for b in range(1,6)], label_ls=['b={}'.format(b) for b in range(1, 6)], title="220t_mom8_quasi_bmix_FT_re", ylim=[-0.5, 2])

plt.show()
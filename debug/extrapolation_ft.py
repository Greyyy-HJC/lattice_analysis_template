# %%
import gvar as gv

from liblattice.coor_to_mom.extrapolation_fit import *

#! extrapolation

ready_for_extrapolation = gv.load("debug/ready_for_extrapolation.pkl") #* 220t_mom8


fix_idx_range_dic = {}
fix_idx_range_dic['b1'] = [10, 13]
fix_idx_range_dic['b2'] = [8, 13]
fix_idx_range_dic['b3'] = [8, 13]
fix_idx_range_dic['b4'] = [8, 13]
fix_idx_range_dic['b5'] = [7, 13]


aft_ex_collect = gv.BufferDict()
for b in range(1, 6):
    temp = ready_for_extrapolation['b{}'.format(b)]

    lam_ls = z_ls_to_lam_ls(temp['z'], a=0.12, Ls=48, mom=8)
    re_gv = temp['re']
    im_gv = temp['im']

    fit_idx_range = fix_idx_range_dic['b{}'.format(b)]

    extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_result = extrapolate_quasi(lam_ls, re_gv, im_gv, fit_idx_range)

    # print('>>> b{}:'.format(b))
    # print(fit_result.format(100))

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


#! do the FT
from liblattice.coor_to_mom.fourier_transform import *

x_ls = np.linspace(-1.5, 1.5, 200)

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

import h5py as h5

#* new soft factor
def sf_from_mh(mom_sf=0, curr=1): # mom_sf=0 表示做外推后的结果
    if(mom_sf!=0):
        mom_310 = str(int(mom_sf/2)) # 310的动量是130的一半
    else:
        mom_310 = 'inf'

    filename = "debug/soft_whole.hdf5"
    data = h5.File(filename, 'r')['p='+mom_310][:, curr, 1:] # data_all.shape=(n_cfg, curr, b), cur=0表示I-gamma5，cur=1表示gamma^perp+gamma^perp gamma_5

    return gv.dataset.avg_data(data, bstrap=True)

b_ls = np.arange(1, 6)
temp = sf_from_mh() 
soft_dic = {}
for b in b_ls:
    soft_dic['b_{}'.format(b)] = temp[b-1] #* temp starts from b=1

# print(soft_dic)

mom_quasi_re = {}

for b in range(1, 6):
    temp = np.array( mom_beam_re['b{}'.format(b)] )
    mom_quasi_re['b{}'.format(b)] = list( temp * np.sqrt( soft_dic['b_{}'.format(b)] ) )


fill_between_ls_plot([x_ls for b in range(1,6)], [gv.mean(mom_quasi_re['b{}'.format(b)]) for b in range(1,6)], [gv.sdev(mom_quasi_re['b{}'.format(b)]) for b in range(1,6)], label_ls=['b={}'.format(b) for b in range(1, 6)], title="220t_mom8_quasi_bmix_FT_re", ylim=[-0.5, 2])

plt.show()


#! quasi-TMDPDF to light-cone TMDPDF

from liblattice.general.constants import *
pz = lat_unit_convert(8, 0.12, 48, 'P')
# print(pz)

def matching_f(quasi, hard, cs_kernel, x_ls, pz, zeta):
    #* return a list of light-cone TMDPDF xf
    lc = []
    for j in range(len(x_ls)):
        x = x_ls[j]
        zeta_z = (2 * x * pz)**2
        lc.append( quasi[j] / hard[j] * np.exp(-0.5 * cs_kernel * np.log(zeta_z / zeta)) * x )

    return lc #* shape = x_ls

from liblattice.general.constants import *

def hard_kernel(x, pz): #* 1 loop, from Xiangdong
    mu = 2
    alphasCF = 0.30444983991952645 * 4 / 3 # mu = 2 GeV #* for old ms-bar

    zeta = (2 * x * pz)**2
    temp = -2 + (np.pi**2)/12 + np.log( zeta / (mu**2) ) - 1/2 * (np.log(zeta / (mu**2)))**2
    h = alphasCF / (2*np.pi) * temp
    return np.exp(h)

def hard_kernel_Yong(x, pz): #* 1 loop, from Yong
    mu = 2
    alphasCF = 0.30444983991952645 * 4 / 3 # mu = 2 GeV #* for old ms-bar

    temp = - (np.log( (2*x*pz)**2 / mu**2 ))**2 + 2 * np.log( (2*x*pz)**2 / mu**2 ) - 4 + np.pi**2 / 6
    h = 1 + alphasCF / (4*np.pi) * temp
    return h

# cs kernel
cs_kernel_mh = gv.load('debug/cs_kernel_b_ls_mh.pkl')
hard = [hard_kernel(x, pz) for x in x_ls]
zeta = 4



light_cone = {}
for b in b_ls:
    cs_kernel = cs_kernel_mh[b-1] #* cs_kernel starts from b=1

    temp = matching_f(mom_quasi_re['b{}'.format(b)], hard, cs_kernel, x_ls, pz, zeta)
    light_cone['b{}'.format(b)] = temp

print(len(x_ls))



fill_between_ls_plot([x_ls[110:170] for b in range(1,6)], [gv.mean(light_cone['b{}'.format(b)][110:170]) for b in range(1,6)], [gv.sdev(light_cone['b{}'.format(b)][110:170]) for b in range(1,6)], label_ls=['b={}'.format(b) for b in range(1, 6)], title="220t_mom8_lc_bmix_re", ylim=[-0.2, 0.6])

plt.show()

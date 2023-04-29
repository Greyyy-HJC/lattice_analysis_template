import numpy as np
import h5py as h5

from liblattice.general.general_plot_funcs import *
from liblattice.preprocess.resampling import *


# test dispersion relation plot
from liblattice.preprocess.dispersion import *
from liblattice.preprocess.read_raw import pt2_to_meff

file = h5.File("test_data/a12m130p_tmdpdf_m220_2pt.h5", "r")['hadron_121050']

meff_avg_dic = {}
meff_ls = []
for mom in range(0, 14, 2):
    data_real = file["mom_{}".format(mom)][:, 1:, 1] 

    data_real_bs = bootstrap(data_real, 500, axis=0)

    data_real_bs_avg = bs_ls_avg(data_real_bs)

    meff_avg = pt2_to_meff(data_real_bs_avg)

    meff_fit_res = meff_fit(np.arange(4, 8), meff_avg[4:8])

    meff_avg_dic[str(mom)] = meff_avg
    meff_ls.append(meff_fit_res)

print(meff_avg_dic)
print(meff_ls)

fit_res = disp_relation_plot(a=0.12, Ls=48, mom_ls=np.arange(0, 14, 2), meff_ls=meff_ls, title='disp_test', save=True)

'''
# generate a 2 dimensional x list to test bootstrap function
x = np.random.rand(100, 3, 3)

bs = bootstrap(x, 50, axis=0)
print(np.shape(bs))

gv_ls_1 = gv.dataset.avg_data(bs, bstrap=True)

gv_ls_2 = bs_ls_avg(bs)

distribution = gv_ls_to_samples_corr(gv_ls_2, 100)

gv_ls_3 = gv.dataset.avg_data(distribution, bstrap=True)


# make a errorbar list plot with three lists
x_ls = [np.arange(10), np.arange(10), np.arange(10)]
y_ls = [gv.mean(gv_ls_1), gv.mean(gv_ls_2), gv.mean(gv_ls_3)]
yerr_ls = [gv.sdev(gv_ls_1), gv.sdev(gv_ls_2), gv.sdev(gv_ls_3)]

errorbar_ls_plot(x_ls, y_ls, yerr_ls, label_ls=['1','2','3'], title="test", ylim=None, save=True)


'''
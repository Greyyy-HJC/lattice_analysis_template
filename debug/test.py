import numpy as np
import h5py as h5

from liblattice.general.general_plot_funcs import *
from liblattice.preprocess.resampling import *

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



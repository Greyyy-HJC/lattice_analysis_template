'''
All those functions that you are not sure where to put them.
Basically, these functions are not related to any specific process.
'''

import h5py as h5
import numpy as np

from liblattice.preprocess.resampling import gv_ls_to_samples_corr

def gv_dic_save_to_h5(gv_dic, N_samp, file_path):
    """convert each key of a gvar dictionary to samples, then save the dict to a h5 file

    Args:
        gv_dic (dict): gvar dictionary
        N_samp (int): number of samples
        file_path (str): the path to save the h5 file
    """
    f = h5.File(file_path, 'w')
    for key in gv_dic:
        gv_ls = gv_dic[key]
        dist = gv_ls_to_samples_corr(gv_ls, N_samp)
        temp = f.create_dataset(key, data=dist)
    f.close()
    return

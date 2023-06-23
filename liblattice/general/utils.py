'''
All those functions that you are not sure where to put them.
Basically, these functions are not related to any specific process.
'''

import h5py as h5
import numpy as np
import gvar as gv
import lsqfit as lsf

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

def find_key(dict, key_words):
    """find the keys that contains the key words in a dictionary

    Args:
        dict (dict): the dictionary to search
        key_words (string): the key words to search

    Returns:
        list: a list of keys that contains the key words
    """

    key_ls = []
    for key in dict:
        if key_words in key:
            key_ls.append(key)

    return key_ls


def constant_fit(data):
    """do a constant fit to the data

    Args:
        data (list): a list of data to do the constant fit

    Returns:
        gvar: the result of the constant fit
    """
    def fcn(x, p):
        return x * 0 + p['const']
    
    priors = gv.BufferDict()
    priors['const'] = gv.gvar(0, 100)
    
    x = np.arange(len(data))
    y = list(data)
    
    fit_res = lsf.nonlinear_fit(data=(x, y), prior=priors, fcn=fcn, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')
    
    return fit_res.p['const']
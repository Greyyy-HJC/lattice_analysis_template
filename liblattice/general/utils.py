'''
All those functions that you are not sure where to put them.
Basically, these functions are not related to any specific process.
'''
import io
import h5py as h5
import numpy as np
import gvar as gv
import lsqfit as lsf
from scipy import interpolate

from liblattice.preprocess.resampling import gv_ls_to_samples_corr
from liblattice.preprocess.resampling import bs_ls_avg

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


def add_error_to_sample(sample_ls):
    """
    Add error to each sample in the sample list by combining the sample with the correlation matrix.

    Args:
        sample_ls (list): List of bootstrap samples, where each sample is a 1D array-like object.

    Returns:
        list: List of samples with errors, where each sample is a gvar object representing the sample with error.

    """
    avg = bs_ls_avg(sample_ls)
    cov = gv.evalcov(avg)
    # sdev = gv.sdev(avg) # * use sdev will cause the error to be larger
    with_err_ls = []
    for sample in sample_ls:
        with_err_ls.append(gv.gvar(sample, cov))

    return np.array(with_err_ls)


def gv_ls_interpolate(x_ls, gv_ls, x_new, N_samp=100, method="cubic"):
    """
    Interpolate a list of gvar objects to a new x list.

    Args:
        x_ls (list): List of x values.
        gv_ls (list): List of gvar objects.
        x_new (list): New x values to interpolate to.
        N_samp (int, optional): Number of samples. Defaults to 100.
        method (str, optional): Interpolation method. Defaults to "cubic".

    Returns:
        gvar: Interpolated gvar object.

    """
    y_ls_samp = gv_ls_to_samples_corr(gv_ls, N_samp)
    y_new_samp = []
    for y_ls in y_ls_samp:
        x_array = np.array(x_ls)
        y_array = np.array(y_ls)
        y_new = interpolate.interp1d(x_array, y_array, kind='cubic')(x_new)
        y_new_samp.append(y_new)

    return bs_ls_avg(y_new_samp)


# 为了解决 joblib 在处理 GVar 对象时丢失相关性的问题，你可以采用手动序列化和反序列化的方法。

def serialize_gvar(gvar_obj):
    buffer = io.BytesIO()
    gv.dump(gvar_obj, buffer)
    buffer.seek(0)  # 重置缓冲区的位置
    return buffer.getvalue()

def deserialize_gvar(serialized_gvar):
    buffer = io.BytesIO(serialized_gvar)
    buffer.seek(0)  # 重置缓冲区的位置
    return gv.load(buffer)

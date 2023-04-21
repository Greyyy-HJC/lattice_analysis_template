'''
Here are functions related to resampling, including bootstrap and jackknife.
'''

import numpy as np

def bootstrap(data, samp_times, axis=0):
    """Do bootstrap resampling on the data.

    Args:
        data (list): data to be resampled
        samp_times (int): how many times to sample, i.e. how many bootstrap samples to generate
        axis (int, optional): which axis to resample on. Defaults to 0.

    Returns:
        list: bootstrap samples
    """

    N_conf = data.shape[axis]
    conf_bs = np.random.choice(N_conf, (samp_times, N_conf), replace=True)
    bs_ls = np.take(data, conf_bs, axis=axis)
    bs_ls = np.mean(bs_ls, axis=axis+1)

    return bs_ls
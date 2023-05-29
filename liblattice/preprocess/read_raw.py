'''
Functions used to read from the raw data and return numpy array / gvar list of 2pt, 3pt, ratio, meff, etc. 
'''

import numpy as np

from liblattice.general.utils import find_key


def pt2_to_meff(pt2_ls):
    """convert the 2pt correlator list to the effective mass list

    Args:
        pt2_ls (array): array of 2pt correlator

    Returns:
        array: array of effective mass
    """

    meff_ls = []
    for i in range(len(pt2_ls) - 1):
        val = np.log(pt2_ls[i]) - np.log(pt2_ls[i + 1])
        meff_ls.append(val)
    meff_ls = np.array(meff_ls)

    return meff_ls


def pt2_pt3_to_ratio(pt2_ls, pt3_dic, tsep_ls):
    """convert the 2pt correlator list and 3pt correlator dictionary to the ratio dictionary

    Args:
        pt2_ls (list): list of 2pt correlator start from t = 0
        pt3_dic (dict): dictionary of 3pt correlator with keys indicating the tsep, tau from 0 to tsep(included)
        tsep_ls (list): list of tsep

    Returns:
        dict: dictionary of ratio with keys the same as the pt3_dic, tau from 0 to tsep(included)
    """    

    ratio_dic = {}
    for tsep in tsep_ls:
        key_ls = find_key(pt3_dic, 'tsep_' + str(tsep))
        for key in key_ls:
            pt3_ls = pt3_dic[key]
            ratio_ls = pt3_ls / pt2_ls[tsep]
            ratio_dic[key] = ratio_ls

    return ratio_dic


def ratio_to_fh(ratio_dic, tsep_ls, tau_cut, gap):
    """convert the ratio dictionary to the Feynman-Hellmann dict

    Args:
        ratio_dic (dict): dictionary of ratio with keys indicating the tsep, tau from 0 to tsep(included)
        tsep_ls (list): list of tsep, should be continuous
        tau_cut (int): the tau cut for the Feynman-Hellmann, i.e. sum from tau_cut to tsep-tau_cut(included)
        gap (int): the tsep gap between two summation when constructing the Feynman-Hellmann

    Returns:
        dict: dict of Feynman-Hellmann data with keys the same as the ratio_dic(but surely with less keys because of the gap)
    """    

    fh_dic = {}
    for tsep in tsep_ls[:-gap]:
        key_ls = find_key(ratio_dic, 'tsep_' + str(tsep))
        key_bigger_ls = find_key(ratio_dic, 'tsep_' + str(tsep + gap)) # the key for the bigger tsep = tsep + gap

        for key, key_bigger in zip(key_ls, key_bigger_ls):
            ratio_ls = ratio_dic[key]
            ratio_bigger_ls = ratio_dic[key_bigger]

            summation = np.sum(ratio_ls[tau_cut:tsep - tau_cut + 1])
            summation_bigger = np.sum(ratio_bigger_ls[tau_cut:tsep + gap - tau_cut + 1])

            fh_dic[key] = (summation_bigger - summation) / gap

    return fh_dic


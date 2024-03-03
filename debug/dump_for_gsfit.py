# %%
import logging
import gvar as gv

from read_data import get_2pt_data, get_ratio_data

a = 0.06
Ls = 48
tsep_ls = [6, 8, 10, 12]
N_samp = 800


def read_and_fit_single(px, b, z, tsep_ls, fit_tsep_ls, Ls, N_samp):
    logging.info(f">>> PARA: px={px}, b={b}, z={z}: ")

    data_collection = {}
    if px == 3 or px == 4:
        tau_cut = 2
    elif px == 5:
        tau_cut = 2

    # read 2pt data
    ss_data = get_2pt_data("SS", px=px, py=px, pz=0, jk_bs="bs")[0]
    sp_data = get_2pt_data("SP", px=px, py=px, pz=0, jk_bs="bs")[0]

    data_collection[f"pt2_ss_p{px}"] = ss_data
    data_collection[f"pt2_sp_p{px}"] = sp_data

    # read ratio data
    print(f">>> Reading ratio data for px={px}, b={b}, z={z} ...")
    for tsep in tsep_ls:
        temp_re, temp_im = get_ratio_data(
            px=px, py=px, pz=0, b=b, z=z, tsep_ls=tsep_ls, jk_bs="bs"
        )
        data_collection[f"ra_re_p{px}_b{b}_z{z}_tsep{tsep}"] = temp_re[
            :, tsep_ls.index(tsep), :
        ]
        data_collection[f"ra_im_p{px}_b{b}_z{z}_tsep{tsep}"] = temp_im[
            :, tsep_ls.index(tsep), :
        ]

    print([keys for keys in data_collection.keys()])

    return data_collection


# %%
if __name__ == "__main__":
    bmax = 16  # todo
    zmax = 21  # * means zmax * sqrt(2)
    tsep_ls = [6, 8, 10, 12]
    fit_tsep_ls = [6, 8, 10, 12]
    p_ls = [3, 4, 5]

    for px in p_ls:
        for b in range(bmax):
            for z in range(zmax):
                data_collection = read_and_fit_single(
                    px, b, z, tsep_ls, fit_tsep_ls, Ls=Ls, N_samp=N_samp
                )

                gv.dump(
                    data_collection,
                    f"../output/dump/ready_for_gsfit/pxy{px}_b{b}_z{z}.dat",
                )


# %%

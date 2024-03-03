# %%
import h5py as h5
import numpy as np
import gvar as gv
from liblattice.preprocess.resampling import (
    jackknife,
    bootstrap,
    bootstrap_with_seed,
    jk_ls_avg,
    bs_ls_avg,
)

N_conf = 553
tsep_ls = [6, 8, 10, 12]

N_bs_samp = 800
# bs_seed = np.random.randint(0, N_conf, size=(N_bs_samp, N_conf))
bs_seed = gv.load("../cache/bs_seed.pkl")

def bad_point_filter(data, threshold=1):
    """
    Filter out the bad points in the data.

    Args:
        data (ndarray): The input data.
        threshold (float, optional): The threshold value. Defaults to 1.

    Returns:
        ndarray: The filtered data.
    """
    mask = np.abs(data) > threshold
    bad_loc = np.argwhere(mask)

    for loc in bad_loc:
        data[tuple(loc)] = np.random.choice([-1, 1])

    return data


def get_2pt_data(ss_sp, px, py, pz, jk_bs=None):
    """
    Get the 2-point data for a given set of parameters.

    Parameters:
    - ss_sp (str): The value of ss_sp parameter.
    - px (int): The value of px parameter.
    - py (int): The value of py parameter.
    - pz (int): The value of pz parameter.
    - jk_bs (str or None): The type of analysis to perform. Options are "jk" for jackknife, "bs" for bootstrap, or None for no analysis.

    Returns:
    - pt2_real (ndarray): The real part of the 2-point data.
    - pt2_imag (ndarray): The imaginary part of the 2-point data.

    """
    if px == 4 or px == 5:
        pt2_file = f"../data/c2pt_comb/c2pt.CG52bxyp30_CG52bxyp30.{ss_sp}.meson_g15.PX{px}_PY{py}_PZ{pz}"
    elif px == 3:
        pt2_file = f"../data/c2pt_comb/c2pt.CG52bxyp20_CG52bxyp20.{ss_sp}.meson_g15.PX{px}_PY{py}_PZ{pz}"
    else:
        pt2_file = f"../data/c2pt_comb/c2pt.CG52bxyp30_CG52bxyp30.{ss_sp}.meson_g15.PX{px}_PY{py}_PZ{pz}"

    # read csv file
    pt2_real = np.loadtxt(pt2_file + ".real", skiprows=1, delimiter=",")
    pt2_imag = np.loadtxt(pt2_file + ".imag", skiprows=1, delimiter=",")

    # the idx 0 on axis 1 is the time slice, so we remove it and swap axes to make configurations on axis 0
    pt2_real = np.swapaxes(pt2_real[:, 1:], 0, 1)
    pt2_imag = np.swapaxes(pt2_imag[:, 1:], 0, 1)
    pt2_real = bad_point_filter(pt2_real)
    pt2_imag = bad_point_filter(pt2_imag)

    if jk_bs == None:
        return pt2_real, pt2_imag
    
    elif jk_bs == "jk":
        pt2_real_jk = jackknife(pt2_real)
        pt2_imag_jk = jackknife(pt2_imag)

        return pt2_real_jk, pt2_imag_jk
    
    elif jk_bs == "bs":
        pt2_real_bs = bootstrap_with_seed(pt2_real, bs_seed)
        pt2_imag_bs = bootstrap_with_seed(pt2_imag, bs_seed)

        return pt2_real_bs, pt2_imag_bs

def get_3pt_data(px, py, pz, b, z, tsep_ls, jk_bs=None):
    """
    Retrieve 3-point correlation function data from a file.

    Args:
        px (int): Momentum component in the x-direction.
        py (int): Momentum component in the y-direction.
        pz (int): Momentum component in the z-direction.
        b (int): Binning index.
        z (int): Z index.
        tsep_ls (list): List of time separations.
        jk_bs (str, optional): Jackknife or bootstrap method. Defaults to None.

    Returns:
        tuple: A tuple containing the real and imaginary parts of the 3-point correlation function data.
            If jk_bs is None, returns (pt3_real, pt3_imag).
            If jk_bs is "jk", returns (pt3_real_jk, pt3_imag_jk).
            If jk_bs is "bs", returns (pt3_real_bs, pt3_imag_bs).
    """
    
    if px == 4 or px == 5:
        pt3_file = f"../data/c3pt_h5/qpdf.SS.meson.ama.CG52bxyp30_CG52bxyp30.PX{px}_PY{py}_PZ{pz}.Z0-24.XY0-24.g8.qx0_qy0_qz0.h5"
    elif px == 3:
        pt3_file = f"../data/c3pt_h5/qpdf.SS.meson.ama.CG52bxyp20_CG52bxyp20.PX{px}_PY{py}_PZ{pz}.Z0-24.XY0-24.g8.qx0_qy0_qz0.h5"

    pt3_data = []
    for tsep in tsep_ls:
        data = h5.File(pt3_file, "r")[f"dt{tsep}"][f"Z{b}"][f"XY{z}"][:] # * Note here z = 1 means x = y = 1, i.e. the real separation is np.sqrt(2)
        pt3_data.append(data)

    pt3_data = np.array(pt3_data)
    pt3_data = np.swapaxes(pt3_data, 0, 2)
    pt3_data = np.swapaxes(pt3_data, 1, 2)

    pt3_real = np.real(pt3_data)
    pt3_imag = np.imag(pt3_data)
    pt3_real = bad_point_filter(pt3_real)
    pt3_imag = bad_point_filter(pt3_imag)

    if jk_bs == None:
        return pt3_real, pt3_imag
    
    elif jk_bs == "jk":
        pt3_real_jk = jackknife(pt3_real)
        pt3_imag_jk = jackknife(pt3_imag)

        return pt3_real_jk, pt3_imag_jk
    
    elif jk_bs == "bs":
        pt3_real_bs = bootstrap_with_seed(pt3_real, bs_seed)
        pt3_imag_bs = bootstrap_with_seed(pt3_imag, bs_seed)

        return pt3_real_bs, pt3_imag_bs


def get_ratio_data(px, py, pz, b, z, tsep_ls, jk_bs="jk"):
    """
    Calculate the ratio of 3pt correlators to 2pt correlators.

    Parameters:
    px (float): Momentum component x.
    py (float): Momentum component y.
    pz (float): Momentum component z.
    b (float): Impact parameter.
    z (float): Light-cone momentum fraction.
    tsep_ls (list): List of time separations.
    jk_bs (str, optional): Jackknife or bootstrap method. Defaults to "jk".

    Returns:
    np.array: Array of real parts of the ratio.
    np.array: Array of imaginary parts of the ratio.
    """
    # * take 2pt_ss as the denominator, do the ratio on each sample
    if jk_bs == "jk":
        pt2_real, pt2_imag = get_2pt_data("SS", px, py, pz, jk_bs="jk")
        pt3_real, pt3_imag = get_3pt_data(px, py, pz, b, z, tsep_ls, jk_bs="jk")
    elif jk_bs == "bs":
        pt2_real, pt2_imag = get_2pt_data("SS", px, py, pz, jk_bs="bs")
        pt3_real, pt3_imag = get_3pt_data(px, py, pz, b, z, tsep_ls, jk_bs="bs")

    ra_real = []
    ra_imag = []
    N_samp = len(pt2_real)

    for n in range(N_samp):
        ra_real.append([])
        ra_imag.append([])
        for id in range(len(tsep_ls)):
            tsep = tsep_ls[id]

            pt2_complex = pt2_real[n][tsep] + 1j * pt2_imag[n][tsep]
            pt3_complex = pt3_real[n][id] + 1j * pt3_imag[n][id]
            # * use complex divide complex to get ratio
            ra_complex = pt3_complex / pt2_complex

            ra_real[n].append(np.real(ra_complex))
            ra_imag[n].append(np.imag(ra_complex))
            # * here includes all 16 tau values from 0 to 15

    return np.array(ra_real), np.array(ra_imag)
    # * shape = ( N_samp, len(tsep_ls), 16 )


def get_sum_data(px, py, pz, b, z, tsep_ls, jk_bs="jk", tau_cut=1):
    """
    Calculate the sum of the ratio data over the tau axis.

    Args:
        px (array-like): x-component of momentum.
        py (array-like): y-component of momentum.
        pz (array-like): z-component of momentum.
        b (array-like): impact parameter.
        z (array-like): light-cone momentum fraction.
        tsep_ls (array-like): list of time separations.
        jk_bs (str, optional): jackknife binning scheme. Defaults to "jk".
        tau_cut (int, optional): number of contact points to cut. Defaults to 1.

    Returns:
        tuple: A tuple containing the sum of the real and imaginary parts of the ratio data.
               The shape of each element in the tuple is (N_samp, len(tsep_ls)).
    """
    ra_real, ra_imag = get_ratio_data(px, py, pz, b, z, tsep_ls, jk_bs=jk_bs)

    sum_real, sum_imag = [], []

    for id in range(len(tsep_ls)):
        tsep = tsep_ls[id]
        cutted_real = ra_real[:, id, tau_cut : tsep - tau_cut + 1]
        cutted_imag = ra_imag[:, id, tau_cut : tsep - tau_cut + 1]

        sum_real.append(np.sum(cutted_real, axis=1))
        sum_imag.append(np.sum(cutted_imag, axis=1))

    sum_real = np.swapaxes(
        np.array(sum_real), 0, 1
    )  # * swap the sample axis to the 0-th axis
    sum_imag = np.swapaxes(np.array(sum_imag), 0, 1)

    return sum_real, sum_imag


# %%
###############################
### Plot raw data for check ###
###############################
if __name__ == "__main__":
    from liblattice.preprocess.read_raw import *
    from liblattice.general.plot_settings import *
    from liblattice.general.general_plot_funcs import errorbar_ls_plot
    #! Plot meff to check
    if True:
        px = py = 3
        pz = 0

        # * meff
        pt2_ss_real, pt2_ss_imag = get_2pt_data("SS", px, py, pz, jk_bs="bs")

        meff_ss = pt2_to_meff(bs_ls_avg(pt2_ss_real), boundary="periodic")[:20]

        pt2_sp_real, pt2_sp_imag = get_2pt_data("SP", px, py, pz, jk_bs="bs")

        meff_sp = pt2_to_meff(bs_ls_avg(pt2_sp_real), boundary="periodic")[:20]

        x_ls = [np.arange(20), np.arange(20)]
        y_ls = [gv.mean(meff_ss), gv.mean(meff_sp)]
        yerr_ls = [gv.sdev(meff_ss), gv.sdev(meff_sp)]

        errorbar_ls_plot(
            x_ls,
            y_ls,
            yerr_ls,
            label_ls=["SS", "SP"],
            title="meff comparison, P{}".format(px),
            ylim=[0, 2],
            save=False,
        )

    # %%
    #! Dispersion relation
    if True:  # * plot meff
        from liblattice.preprocess.dispersion import *

        a = 0.06  # lattice spacing in fm
        Ls = 48  # lattice size in spatial direction

        pz = 0
        meff_ss_ls = []
        meff_sp_ls = []
        pnorm_ls = []
        label_ls = []

        tmin = 0
        tmax = 13

        for px in range(2, 8):
            py = px
            pnorm = np.sqrt(px**2 + py**2)

            pt2_ss_real, pt2_ss_imag = get_2pt_data("SS", px, py, pz, jk_bs="bs")
            meff_ss_avg = pt2_to_meff(bs_ls_avg(pt2_ss_real), boundary="periodic")

            pt2_sp_real, pt2_sp_imag = get_2pt_data("SP", px, py, pz, jk_bs="bs")
            meff_sp_avg = pt2_to_meff(bs_ls_avg(pt2_sp_real), boundary="periodic")

            meff_ss_ls.append(meff_ss_avg[tmin:tmax])
            meff_sp_ls.append(meff_sp_avg[tmin:tmax])

            pnorm_ls.append(pnorm)

            label_ls.append(f"PX{px}PY{py}")

        # Define a function to handle the common plotting logic.
        def plot_meff(meff_ls, title, save_name=None):
            fig, ax = plt.subplots(figsize=fig_size)

            # Use enumerate instead of manually incrementing i
            for i, (meff_avg, label, pnorm) in enumerate(zip(meff_ls, label_ls, pnorm_ls)):
                ax.errorbar(
                    np.arange(tmin, tmax),
                    gv.mean(meff_avg),
                    gv.sdev(meff_avg),
                    marker=marker_ls[i],
                    label=label,
                    color=color_ls[i],
                    **errorb,
                )

                m0 = 0.3 * 0.06 / GEV_FM
                mom = pnorm * 2 * np.pi / Ls
                en = np.sqrt(m0**2 + mom**2)

                print(pnorm)

                ax.plot(
                    np.arange(tmax - 3 - i, tmax),
                    np.full(3 + i, en),
                    color=color_ls[i],
                    linestyle="dashed",
                )

            ax.tick_params(direction="in", top="on", right="on", **ls_p)
            ax.grid(linestyle=":")
            ax.set_ylim(0, 2.5)
            plt.title(title, **fs_p)
            plt.legend(ncol=3, loc="upper center")
            if save_name is not None:
                plt.savefig(save_name, transparent=True)
            plt.show()

        plot_meff(
            meff_ss_ls, "Effective mass mix P, SS", "../output/plots/meff_mix_P_SS.pdf"
        )

        plot_meff(
            meff_sp_ls, "Effective mass mix P, SP", "../output/plots/meff_mix_P_SP.pdf"
        )


    # %%
    #! Plot ratio to check
    if True:
        px = py = 3
        pz = 0
        b = 0
        z = 0
        tau_cut_plot = 2
        tsep_ls_plot = [6, 8, 10]

        # * ratio
        ra_real, ra_imag = get_ratio_data(px, py, pz, b, z, tsep_ls, jk_bs="bs")
        ra_real_avg = gv.dataset.avg_data(ra_real, bstrap=True)
        ra_imag_avg = gv.dataset.avg_data(ra_imag, bstrap=True)

        # * real part
        x_ls = []
        y_ls = []
        yerr_ls = []
        label_ls = []

        for id in range(len(tsep_ls_plot)):
            tsep = tsep_ls_plot[id]
            x_ls.append(np.arange(tau_cut_plot, tsep + 1 - tau_cut_plot) - tsep / 2)
            y_ls.append(gv.mean(ra_real_avg)[id, tau_cut_plot : tsep + 1 - tau_cut_plot])
            yerr_ls.append(gv.sdev(ra_real_avg)[id, tau_cut_plot : tsep + 1 - tau_cut_plot])
            label_ls.append(f"tsep={tsep}")

        errorbar_ls_plot(
            x_ls,
            y_ls,
            yerr_ls,
            label_ls=label_ls,
            title="ratio tsep comparison, real, P{}, b={}, z={}".format(px, b, z),
            save=False,
        )

        # * imag part
        x_ls = []
        y_ls = []
        yerr_ls = []
        label_ls = []

        for id in range(len(tsep_ls_plot)):
            tsep = tsep_ls_plot[id]
            x_ls.append(np.arange(tau_cut_plot, tsep + 1 - tau_cut_plot) - tsep / 2)
            y_ls.append(gv.mean(ra_imag_avg)[id, tau_cut_plot : tsep + 1 - tau_cut_plot])
            yerr_ls.append(gv.sdev(ra_imag_avg)[id, tau_cut_plot : tsep + 1 - tau_cut_plot])
            label_ls.append(f"tsep={tsep}")

        errorbar_ls_plot(
            x_ls,
            y_ls,
            yerr_ls,
            label_ls=label_ls,
            title="ratio tsep comparison, imag, P{}, b={}, z={}".format(px, b, z),
            save=False,
        )

    # %%
    #! Plot sum to check
    if True:
        px = py = 4
        pz = 0
        b = 0
        z = 0
        tau_cut_plot = 3

        # * real part
        x_ls, y_ls, yerr_ls, label_ls = [], [], [], []

        for z in range(5):
            # * sum
            sum_real, sum_imag = get_sum_data(px, py, pz, b, z, tsep_ls, jk_bs="bs", tau_cut=tau_cut_plot)
            sum_real_avg = gv.dataset.avg_data(sum_real, bstrap=True)

            x_ls.append(tsep_ls)
            y_ls.append(gv.mean(sum_real_avg))
            yerr_ls.append(gv.sdev(sum_real_avg))
            label_ls.append(f"z={z}")

        errorbar_ls_plot(
            x_ls,
            y_ls,
            yerr_ls,
            label_ls=label_ls,
            title="sum tsep comparison, real, P{}, b={}, mixz".format(px, b),
            save=False,
        )

        # * imag part
        x_ls, y_ls, yerr_ls, label_ls = [], [], [], []

        for z in range(5):
            # * sum
            sum_real, sum_imag = get_sum_data(px, py, pz, b, z, tsep_ls, jk_bs="bs", tau_cut=tau_cut_plot)
            sum_imag_avg = gv.dataset.avg_data(sum_imag, bstrap=True)

            x_ls.append(tsep_ls)
            y_ls.append(gv.mean(sum_imag_avg))
            yerr_ls.append(gv.sdev(sum_imag_avg))
            label_ls.append(f"z={z}")

        errorbar_ls_plot(
            x_ls,
            y_ls,
            yerr_ls,
            label_ls=label_ls,
            title="sum tsep comparison, imag, P{}, b={}, mixz".format(px, b),
            save=False,
        )

# %%

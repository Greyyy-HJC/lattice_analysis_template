# %%
import logging

# set logging config
logging.basicConfig(filename='../log/bad_fit.log', level=logging.INFO, format='%(asctime)s %(message)s')

import numpy as np
import gvar as gv
import lsqfit as lsf
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from read_data import get_2pt_data, get_ratio_data
from gs_fit_2pt import single_2pt_fit
from gs_fit_ratio import single_ra_fit
from ft import interpolate_and_ft_single_sample
from cs_kernel import cal_cs_kernel_single_sample
from liblattice.preprocess.resampling import bs_dic_avg, bs_ls_avg
from liblattice.gs_fit.prior_setting import two_state_fit
from liblattice.general.utils import add_error_to_sample, constant_fit
from liblattice.general.general_plot_funcs import fill_between_ls_plot, errorbar_ls_plot, fill_between_plot

a = 0.06
Ls = 48
tsep_ls = [6, 8, 10, 12]
N_samp = 800

# %%
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
        temp_re, temp_im = get_ratio_data(px=px, py=px, pz=0, b=b, z=z, tsep_ls=tsep_ls, jk_bs="bs")
        data_collection[f'ra_re_p{px}_b{b}_z{z}_tsep{tsep}'] = temp_re[:, tsep_ls.index(tsep), :]
        data_collection[f'ra_im_p{px}_b{b}_z{z}_tsep{tsep}'] = temp_im[:, tsep_ls.index(tsep), :]

    # prepare for fit
    ss_ls = data_collection[f"pt2_ss_p{px}"]
    sp_ls = data_collection[f"pt2_sp_p{px}"]
    ss_gv_array = add_error_to_sample(ss_ls)
    sp_gv_array = add_error_to_sample(sp_ls)

    ra_re_gv_dic = {f'tsep_{tsep}': add_error_to_sample(data_collection[f'ra_re_p{px}_b{b}_z{z}_tsep{tsep}']) for tsep in fit_tsep_ls}
    ra_im_gv_dic = {f'tsep_{tsep}': add_error_to_sample(data_collection[f'ra_im_p{px}_b{b}_z{z}_tsep{tsep}']) for tsep in fit_tsep_ls}


    # do the fit
    temp_re = []
    temp_im = []

    # Worker function that will be executed in parallel
    for n in tqdm( range(N_samp), desc=f"Loop in samples for gs fit of P{px}, b={b}, z={z}" ):
        logging.info(">>> 2pt Config: n = %d", n)

        ss_gv = ss_gv_array[n]
        sp_gv = sp_gv_array[n]
        
        ss_fit_res, _ = single_2pt_fit(ss_gv, sp_gv, Ls, priors=two_state_fit(), tmin=3, tmax=13)

        logging.info(">>> ratio Config: n = %d", n)

        ra_re_avg_dic = {f'tsep_{tsep}': ra_re_gv_dic[f'tsep_{tsep}'][n] for tsep in fit_tsep_ls}
        ra_im_avg_dic = {f'tsep_{tsep}': ra_im_gv_dic[f'tsep_{tsep}'][n] for tsep in fit_tsep_ls}

        ra_fit_res = single_ra_fit(ra_re_avg_dic, ra_im_avg_dic, b, z, fit_tsep_ls, tau_cut, ss_fit_res, Ls)
        temp_re.append(gv.mean(ra_fit_res.p['pdf_re']))
        temp_im.append(gv.mean(ra_fit_res.p['pdf_im']))


    z_dep_result = {'re': temp_re, 'im': temp_im}

    return z_dep_result


def extrapolate_and_ft(z_dep_ls, zmax, b, x_ls, px, py, pz, a, Ls, fit_z_min, fit_z_max, tail_end, zmax_keep, ifextrapolate=True):
    # * z_dep_ls is real part only

    z_array = np.arange(zmax) * np.sqrt(2) #* for z = (x, y)

    priors = gv.BufferDict()
    priors["b"] = gv.gvar(1, 10)
    priors["c"] = gv.gvar(0, 10)
    priors["d"] = gv.gvar(0, 10)
    priors["log(n)"] = gv.gvar(0, 10)
    priors["log(m)"] = gv.gvar(0, 10)

    def fcn(x, p):
        #todo poly * exp * power
        return ( p["b"] + p["c"] * x + p["d"] * x**2 ) * np.exp(-x * p["m"]) / (x ** p["n"])
    
    z_dep_gv = add_error_to_sample(z_dep_ls)
    # to fit in extrapolation
    fit_mask = (z_array >= fit_z_min) & (z_array <= fit_z_max)
    fit_z_array = z_array[fit_mask]

    # to keep the original data
    #* keep the data points smaller than zmax_keep * a
    keep_mask = z_array < zmax_keep

    z_dep_ext_ls = []
    x_dep_re_ls = []
    x_dep_im_ls = []

    if ifextrapolate:
        for n in tqdm( range(N_samp), desc=f"Loop in samples for extra and FT of P{px}, b={b}" ):
            z_dep = z_dep_gv[n]
            fit_gv = z_dep[fit_mask]

            fit_res = lsf.nonlinear_fit(
                data=(fit_z_array, fit_gv), fcn=fcn, prior=priors, maxit=10000
            )

            if fit_res.Q < 0.05:
                logging.info(f">>> Bad extrapolation with Q = {fit_res.Q}")

            # complete the tail
            z_gap = abs(z_array[1] - z_array[0])
            # * start to apply extrapolation from the first point larger than zmax_keep
            tail_array = np.arange(z_array[keep_mask][-1] + z_gap, tail_end, z_gap) 
            z_dep_tail = fcn(tail_array, fit_res.p)

            # concatenate the original part and the tail part
            z_array_ext = np.concatenate((z_array[keep_mask], tail_array))
            z_dep_ext = np.concatenate((z_dep[keep_mask], z_dep_tail))
        
            z_dep_ext = gv.mean(z_dep_ext)
            z_array_int = np.linspace(-tail_end, tail_end, 100)

            x_dep_re_samp, x_dep_im_samp = interpolate_and_ft_single_sample(z_dep_ext, z_array_ext, z_array_int, x_ls, px, py, pz, a, Ls)

            z_dep_ext_ls.append(z_dep_ext)
            x_dep_re_ls.append(x_dep_re_samp)
            x_dep_im_ls.append(x_dep_im_samp)

        z_dep_ext_gv = bs_ls_avg(z_dep_ext_ls)

        fill_between_plot(z_array_ext, gv.mean(z_dep_ext_gv), gv.sdev(z_dep_ext_gv), title=f"z_dep_extrapolated_P{px}_b{b}", save=True)

        return x_dep_re_ls, x_dep_im_ls

    elif ifextrapolate == False:
        for n in tqdm( range(N_samp), desc=f"Loop in samples for FT of P{px}, b={b}" ):
            z_dep = gv.mean( z_dep_gv[n] )
            z_array_int = np.linspace(-tail_end, tail_end, 100)

            x_dep_re_samp, x_dep_im_samp = interpolate_and_ft_single_sample(z_dep, z_array, z_array_int, x_ls, px, py, pz, a, Ls)

            x_dep_re_ls.append(x_dep_re_samp)
            x_dep_im_ls.append(x_dep_im_samp)

        return x_dep_re_ls, x_dep_im_ls



# %%
if __name__ == "__main__":
    bmax = 16 #todo
    zmax = 21 # * means zmax * sqrt(2) 
    tsep_ls = [6, 8, 10, 12]
    fit_tsep_ls = [6, 8, 10, 12]
    p_ls = [3, 4, 5]
    ifplot = False


    #! read and fit data for z dependence
    if False:
        # define wrapper function
        def read_and_fit_wrapper(px, b, z):
            result = read_and_fit_single(px, b, z, tsep_ls, fit_tsep_ls, Ls=Ls, N_samp=N_samp)
        
            return result

        # para list
        loop_params = [(px, b, z) for px in p_ls for b in range(bmax) for z in range(zmax)]

        # use joblib to parallel
        results = Parallel(n_jobs=8)(delayed(read_and_fit_wrapper)(px, b, z) for px, b, z in tqdm(loop_params, desc="Loop in px, b, z"))

        # collect z_dep_collection
        z_dep_collection = {}
        temp_dic = {}
        for (px, b, z), result in zip(loop_params, results):
            temp_dic[f're_p{px}_b{b}_z{z}'] = result['re']
            temp_dic[f'im_p{px}_b{b}_z{z}'] = result['im']

        for px in p_ls:
            for b in range(bmax):
                z_dep_collection[f're_p{px}_b{b}'] = np.array([temp_dic[f're_p{px}_b{b}_z{z}'] for z in range(zmax)]).swapaxes(0, 1)
                z_dep_collection[f'im_p{px}_b{b}'] = np.array([temp_dic[f'im_p{px}_b{b}_z{z}'] for z in range(zmax)]).swapaxes(0, 1)

        # dump
        gv.dump(z_dep_collection, "../output/dump/z_dep_collection_cut_222.dat") #todo

    else:
        z_dep_collection = gv.load("../cache/z_dep_collection_cut_222.dat")

    #! do normalization on z dependence: f(b, P, z) / f(b, P=3, 0)
    if False:
        z_dep_collection_normalized = {}
        for px in p_ls:
            for b in range(bmax):
                divisor = z_dep_collection[f're_p3_b{b}'][:, 0][:, np.newaxis]  # Reshape to (800, 1)
                z_dep_collection_normalized[f're_p{px}_b{b}'] = z_dep_collection[f're_p{px}_b{b}'] / divisor
                z_dep_collection_normalized[f'im_p{px}_b{b}'] = z_dep_collection[f'im_p{px}_b{b}'] / divisor

        z_dep_collection = z_dep_collection_normalized
        

    if ifplot:
        x_ls_ls = []
        p3_y_ls_ls = []
        p4_y_ls_ls = []
        p5_y_ls_ls = []
        p3_yerr_ls_ls = []
        p4_yerr_ls_ls = []
        p5_yerr_ls_ls = []
        label_ls = []

        for b in range(2, 12, 2):
            x_ls = np.arange(zmax) * np.sqrt(2) #* for z = (x, y)
            temp_p3 = bs_ls_avg( z_dep_collection[f're_p{3}_b{b}'] )
            temp_p4 = bs_ls_avg( z_dep_collection[f're_p{4}_b{b}'] )
            temp_p5 = bs_ls_avg( z_dep_collection[f're_p{5}_b{b}'] )
            
            x_ls_ls.append(x_ls)
            p3_y_ls_ls.append(gv.mean(temp_p3))
            p4_y_ls_ls.append(gv.mean(temp_p4))
            p5_y_ls_ls.append(gv.mean(temp_p5))
            p3_yerr_ls_ls.append(gv.sdev(temp_p3))
            p4_yerr_ls_ls.append(gv.sdev(temp_p4))
            p5_yerr_ls_ls.append(gv.sdev(temp_p5))
            label_ls.append(f"b={b}")

        title = f"z_dep_P{3}_real_bmix"
        errorbar_ls_plot(x_ls_ls, p3_y_ls_ls, p3_yerr_ls_ls, label_ls, title=title, save=True)

        title = f"z_dep_P{4}_real_bmix"
        errorbar_ls_plot(x_ls_ls, p4_y_ls_ls, p4_yerr_ls_ls, label_ls, title=title, save=True)

        title = f"z_dep_P{5}_real_bmix"
        errorbar_ls_plot(x_ls_ls, p5_y_ls_ls, p5_yerr_ls_ls, label_ls, title=title, save=True)

        plt.close('all')


    #! interpolate and FT to x dependence
    if False:
        p_ls = [3, 4, 5]
        b_ls = np.arange(2, 12, 1)
        ifdump = True

        x_ls = np.arange(-2, 2, 0.005)
        fit_z_min = 5 * np.sqrt(2)
        fit_z_max = 20 * np.sqrt(2)
        tail_end = 30 * np.sqrt(2)
        zmax_keep = 8
        ifextrapolate = True

        x_dep_collection = {}
        x_dep_collection["x_ls"] = x_ls

        def ext_ft_wrapper(px, b):
            py = px
            pz = 0
            z_dep_ls = z_dep_collection[f're_p{px}_b{b}']
            
            x_dep_re_ls, x_dep_im_ls = extrapolate_and_ft(z_dep_ls, zmax, b, x_ls, px, py, pz, a, Ls, fit_z_min, fit_z_max, tail_end, zmax_keep, ifextrapolate)

            result = {'re': np.array(x_dep_re_ls), 'im': np.array(x_dep_im_ls)}

            return result
        
        # para list
        loop_params = [(px, b) for px in p_ls for b in b_ls]

        # use joblib to parallel
        results = Parallel(n_jobs=8)(delayed(ext_ft_wrapper)(px, b) for px, b in tqdm(loop_params, desc="Loop in px, b"))

        # collect x_dep_collection
        for (px, b), result in zip(loop_params, results):
            x_dep_collection[f're_p{px}_b{b}'] = result['re']
            x_dep_collection[f'im_p{px}_b{b}'] = result['im']

        if ifdump:
            gv.dump(x_dep_collection, f"../output/dump/x_dep_collection.dat")
    else:
        x_dep_collection = gv.load(f"../output/dump/x_dep_collection.dat")


    #! bmix x-depenence
    if ifplot:
        b_ls = np.arange(2, 12, 2)

        x_ls_ls = []
        p3_y_ls_ls = []
        p4_y_ls_ls = []
        p5_y_ls_ls = []
        p3_yerr_ls_ls = []
        p4_yerr_ls_ls = []
        p5_yerr_ls_ls = []
        label_ls = []

        for b in b_ls:
            x_ls = x_dep_collection['x_ls']
            temp_p3 = bs_ls_avg( x_dep_collection[f're_p{3}_b{b}'] )
            temp_p4 = bs_ls_avg( x_dep_collection[f're_p{4}_b{b}'] )
            temp_p5 = bs_ls_avg( x_dep_collection[f're_p{5}_b{b}'] )
            
            x_ls_ls.append(x_ls)
            p3_y_ls_ls.append(gv.mean(temp_p3))
            p4_y_ls_ls.append(gv.mean(temp_p4))
            p5_y_ls_ls.append(gv.mean(temp_p5))
            p3_yerr_ls_ls.append(gv.sdev(temp_p3))
            p4_yerr_ls_ls.append(gv.sdev(temp_p4))
            p5_yerr_ls_ls.append(gv.sdev(temp_p5))
            label_ls.append(f"b={b}")


        title = f"x_dep_P{3}_real_bmix"
        fill_between_ls_plot(x_ls_ls, p3_y_ls_ls, p3_yerr_ls_ls, label_ls, title=title, save=True, ylim=[-0.1,1.0])

        title = f"x_dep_P{4}_real_bmix"
        fill_between_ls_plot(x_ls_ls, p4_y_ls_ls, p4_yerr_ls_ls, label_ls, title=title, save=True, ylim=[-0.1,1.0])

        title = f"x_dep_P{5}_real_bmix"
        fill_between_ls_plot(x_ls_ls, p5_y_ls_ls, p5_yerr_ls_ls, label_ls, title=title, save=True, ylim=[-0.1,1.0])

        plt.close('all')

    #! Pmix x-depenence
    if ifplot:
        b_ls = np.arange(2, 12, 2)

        for b in b_ls:
            x_ls = x_dep_collection['x_ls']
            temp_p3 = bs_ls_avg( x_dep_collection[f're_p{3}_b{b}'] )
            temp_p4 = bs_ls_avg( x_dep_collection[f're_p{4}_b{b}'] )
            temp_p5 = bs_ls_avg( x_dep_collection[f're_p{5}_b{b}'] )

            x_ls_ls = [x_ls, x_ls, x_ls]
            y_ls_ls = [gv.mean(temp_p5), gv.mean(temp_p4), gv.mean(temp_p3)]
            yerr_ls_ls = [gv.sdev(temp_p5), gv.sdev(temp_p4), gv.sdev(temp_p3)]
            label_ls = ["P5", "P4", "P3"]

            title = f"x_dep_b{b}_real_Pmix"

            fill_between_ls_plot(x_ls_ls, y_ls_ls, yerr_ls_ls, label_ls, title=title, save=True, ylim=[-0.1,1.0])

        plt.close('all')


    #! calculate cs kernel
    if True:
        b_ls = np.arange(2, 12, 1)
        kernel = "fixed"
        ifsubtract = False

        def cs_kernel_wrapper(b):
            x_ls = x_dep_collection['x_ls']
            temp_p3 = x_dep_collection[f're_p{3}_b{b}']
            temp_p4 = x_dep_collection[f're_p{4}_b{b}']
            temp_p5 = x_dep_collection[f're_p{5}_b{b}']
            
            cs_kernel_p5p4_ls = []
            cs_kernel_p5p3_ls = []
            cs_kernel_p4p3_ls = []
            for n in tqdm(range(N_samp), desc=f"Loop in samples for cs kernel of b={b}"):
                cs_x_ls, cs_kernel = cal_cs_kernel_single_sample(x_ls, temp_p5[n], temp_p4[n], 5, 4, b, a, Ls, kernel, ifsubtract)
                cs_kernel_p5p4_ls.append(cs_kernel)
                # check whether it is nan
                if np.isnan(gv.mean(cs_kernel)).any():
                    print(f">>> cs_kernel_p5p4_ls is nan for b={b}, n={n}")

                cs_x_ls, cs_kernel = cal_cs_kernel_single_sample(x_ls, temp_p5[n], temp_p3[n], 5, 3, b, a, Ls, kernel, ifsubtract)
                cs_kernel_p5p3_ls.append(cs_kernel)

                cs_x_ls, cs_kernel = cal_cs_kernel_single_sample(x_ls, temp_p4[n], temp_p3[n], 4, 3, b, a, Ls, kernel, ifsubtract)
                cs_kernel_p4p3_ls.append(cs_kernel)

            cs_kernel_p5p4_avg = bs_ls_avg(cs_kernel_p5p4_ls)
            cs_kernel_p5p3_avg = bs_ls_avg(cs_kernel_p5p3_ls)
            cs_kernel_p4p3_avg = bs_ls_avg(cs_kernel_p4p3_ls)

            x_ls_ls = [cs_x_ls, cs_x_ls, cs_x_ls]
            y_ls_ls = [gv.mean(cs_kernel_p5p4_avg), gv.mean(cs_kernel_p5p3_avg), gv.mean(cs_kernel_p4p3_avg)]
            yerr_ls_ls = [gv.sdev(cs_kernel_p5p4_avg), gv.sdev(cs_kernel_p5p3_avg), gv.sdev(cs_kernel_p4p3_avg)]
            label_ls = ["P5/P4", "P5/P3", "P4/P3"]

            title = f"cs_kernel_b{b}_{kernel}"

            fill_between_ls_plot(x_ls_ls, y_ls_ls, yerr_ls_ls, label_ls, title=title, save=True, ylim=[-3,3])

            return cs_x_ls, cs_kernel_p5p4_avg, cs_kernel_p5p3_avg, cs_kernel_p4p3_avg, cs_kernel_p5p4_ls, cs_kernel_p5p3_ls, cs_kernel_p4p3_ls
        
        # para list
        loop_params = [(b) for b in b_ls]

        # use joblib to parallel
        results = Parallel(n_jobs=8)(delayed(cs_kernel_wrapper)(b) for b in tqdm(loop_params, desc="Loop in b"))

        # collect cs_kernel_collection
        cs_kernel_collection = {}
        for b, result in zip(loop_params, results):
            cs_x_ls, cs_kernel_p5p4_avg, cs_kernel_p5p3_avg, cs_kernel_p4p3_avg, cs_kernel_p5p4_ls, cs_kernel_p5p3_ls, cs_kernel_p4p3_ls = result

            cs_kernel_collection[f'cs_x_ls_b{b}'] = cs_x_ls
            cs_kernel_collection[f'cs_kernel_p5p4_b{b}'] = cs_kernel_p5p4_avg
            cs_kernel_collection[f'cs_kernel_p5p4_b{b}_bs'] = cs_kernel_p5p4_ls
            cs_kernel_collection[f'cs_kernel_p5p3_b{b}'] = cs_kernel_p5p3_avg
            cs_kernel_collection[f'cs_kernel_p5p3_b{b}_bs'] = cs_kernel_p5p3_ls
            cs_kernel_collection[f'cs_kernel_p4p3_b{b}'] = cs_kernel_p4p3_avg
            cs_kernel_collection[f'cs_kernel_p4p3_b{b}_bs'] = cs_kernel_p4p3_ls


    #! check b-dep of cs kernel
    if True:
        b_ls = np.arange(2, 12, 1)

        csk_p5p4_ls = []
        csk_p5p3_ls = []
        csk_p4p3_ls = []

        for b in b_ls:
            csk_array_bs = cs_kernel_collection[f'cs_kernel_p5p4_b{b}_bs']

            # * 0.4 < x < 0.6
            idx_ini = 40
            idx_end = 81


            csk_mean_bs = [np.mean(sample[idx_ini:idx_end]) for sample in csk_array_bs] 
            csk_sdev_bs = [np.std(sample[idx_ini:idx_end]) for sample in csk_array_bs]

            csk_p5p4_mean = np.mean(csk_mean_bs)
            csk_p5p4_stat = np.std(csk_mean_bs)
            csk_p5p4_sys = np.mean(csk_sdev_bs)
            csk_p5p4_avg = gv.gvar(csk_p5p4_mean, np.sqrt(csk_p5p4_stat**2 + csk_p5p4_sys**2))
            # csk_p5p4_avg = gv.gvar(csk_p5p4_mean, csk_p5p4_stat)




            csk_array_bs = cs_kernel_collection[f'cs_kernel_p5p3_b{b}_bs']

            csk_mean_bs = [np.mean(sample[idx_ini:idx_end]) for sample in csk_array_bs]
            csk_sdev_bs = [np.std(sample[idx_ini:idx_end]) for sample in csk_array_bs]

            csk_p5p3_mean = np.mean(csk_mean_bs)
            csk_p5p3_stat = np.std(csk_mean_bs)
            csk_p5p3_sys = np.mean(csk_sdev_bs)
            csk_p5p3_avg = gv.gvar(csk_p5p3_mean, np.sqrt(csk_p5p3_stat**2 + csk_p5p3_sys**2))
            # csk_p5p3_avg = gv.gvar(csk_p5p3_mean, csk_p5p3_stat)



            csk_array_bs = cs_kernel_collection[f'cs_kernel_p4p3_b{b}_bs']

            csk_mean_bs = [np.mean(sample[idx_ini:idx_end]) for sample in csk_array_bs]
            csk_sdev_bs = [np.std(sample[idx_ini:idx_end]) for sample in csk_array_bs]

            csk_p4p3_mean = np.mean(csk_mean_bs)
            csk_p4p3_stat = np.std(csk_mean_bs)
            csk_p4p3_sys = np.mean(csk_sdev_bs)
            csk_p4p3_avg = gv.gvar(csk_p4p3_mean, np.sqrt(csk_p4p3_stat**2 + csk_p4p3_sys**2))
            # csk_p4p3_avg = gv.gvar(csk_p4p3_mean, csk_p4p3_stat)



            csk_p5p4_ls.append(csk_p5p4_avg)
            csk_p5p3_ls.append(csk_p5p3_avg)
            csk_p4p3_ls.append(csk_p4p3_avg)

        x_ls_ls = [b_ls-0.1, b_ls, b_ls+0.1]
        y_ls_ls = [gv.mean(csk_p5p4_ls), gv.mean(csk_p5p3_ls), gv.mean(csk_p4p3_ls)]
        yerr_ls_ls = [gv.sdev(csk_p5p4_ls), gv.sdev(csk_p5p3_ls), gv.sdev(csk_p4p3_ls)]
        label_ls = ["P5/P4", "P5/P3", "P4/P3"]

        title = f"cs_kernel_b_dep_mean_x_0.4_0.6"

        errorbar_ls_plot(x_ls_ls, y_ls_ls, yerr_ls_ls, label_ls, title=title, save=True, ylim=[-3,1])

        cs_kernel_res = {}
        cs_kernel_res['b_ls'] = b_ls
        cs_kernel_res['csk_p5p4_ls'] = csk_p5p4_ls
        cs_kernel_res['csk_p5p3_ls'] = csk_p5p3_ls
        cs_kernel_res['csk_p4p3_ls'] = csk_p4p3_ls

        gv.dump(cs_kernel_res, "../output/dump/cs_kernel_res.dat")

    plt.close('all')


# %%

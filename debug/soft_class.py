# %%
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate

from bisect import bisect

#* soft function
def read_soft_dic(b_ls, soft_mom):
    soft_f = Soft_Function()
    b_list, soft_factor = soft_f.SF_interpolate(soft_mom) # mom_list = [2, 3, 4]

    a = 0.12

    b_list_div_a = [b/a for b in b_list]

    soft_dic = {}
    for b in b_ls:
        check_index = bisect(b_list_div_a, b)
        print(b_list_div_a[check_index])
        print(soft_factor[check_index])
        soft_dic['b_%d'%(b)] = soft_factor[check_index]
    
    return soft_dic

class Soft_Function(object):
    ## soft function generated on A654, with Pz=1.05, 1.58 and 2.11GeV
    ## b_{\perp} range in [1, 8]a with a=0.098fm

    def ori_SF(self):
        b_list = np.array(list(range(1, 8)))
        mom_list = [2, 3, 4]

        sf = gv.load('debug/soft_factor_A654.pkl')
        print(sf.keys())

        #data = np.array(sf['mom_2'])
        #data_rescale = data / data[0]
        #print(data_rescale)
        #quit()

        fig, ax = plt.subplots(1,1, figsize=(5, 3.09))
        colors = Analysis_Tools().colors()
        count = 0

        for mom in mom_list:
            data_ori = np.array(sf['mom_'+str(mom)])
            data = data_ori / data_ori[0]
            ax.errorbar(b_list*0.098+0.005*count, [v.mean for v in data], [v.sdev for v in data], fmt='x', markerfacecolor='none', capsize=4.5, ms=4.5, color=colors[count], label='Pz='+str('%.2f'%(mom*0.525))+'GeV')
            count += 1

        ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, borderpad=0.3, \
            ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)

        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        plt.yscale('log')
        plt.xlabel('$b_{\perp}$')
        plt.show()

    def SF_interpolate(self, mom):
        sf = gv.load('debug/soft_factor_A654.pkl')
        data_ori = np.array(sf['mom_'+str(mom)])
        data = data_ori / data_ori[0]

        n_resample = 100; n_interp = 100
        ## turn data to distributions, then do the interpolate
        data_dis = Correlated_distributions().gv_to_samples_corr(data, n_resample) # shape=(n_resample, b=7)
        #print(data_dis.shape)

        x_in = np.arange(1, 8)
        x_out = np.arange(1, 7, 0.01)

        ## loop for interpolate
        #data_interp_dis = []
        #for i in range(n_resample):
        #    f = interpolate.interp1d(x_in, data_dis[i], kind='linear')
        #    data_interp = f(x_out)
        #    data_interp_dis.append(data_interp)
        #data_interp_dis = np.array(data_interp_dis)

        f = interpolate.interp1d(x_in, data_dis, kind='linear')
        data_interp = f(x_out)  # shape=(n_resample, n_interp*(b-1))
        #print((data_interp_dis == data_interp).all())

        new_data = gv.gvar(np.mean(data_interp, axis=0), np.cov(data_interp, rowvar=False)) # shape=( n_interp*(b-1), )
        b_list = x_out * 0.098
        return b_list, new_data
    
    def plot_interpolated_SF(self):
        mom_list = [2, 3, 4]
        fig, ax = plt.subplots(1,1, figsize=(5, 3.09))
        colors = Analysis_Tools().colors()
        count = 0

        b_list = np.array(list(range(1, 8)))
        mom_list = [4, 3, 2]
        sf = gv.load('debug/soft_factor_A654.pkl')

        for mom in mom_list:
            ## plot ori data
            data_ori = np.array(sf['mom_'+str(mom)])
            data = data_ori / data_ori[0]
            ax.errorbar(b_list*0.098-0.005*count, [v.mean for v in data], [v.sdev for v in data], fmt='x', markerfacecolor='none', capsize=4.5, ms=4.5, color=colors[count], label='Pz='+str('%.2f'%(mom*0.525))+'GeV')
            

            ## plot extrapolated data
            xx, data_ext = self.SF_interpolate(mom) # shape=( n_interp*(b-1), )
            ax.plot(xx, gv.mean(data_ext), color=colors[count], alpha=0.5)
            ax.fill_between(xx, gv.mean(data_ext)-gv.sdev(data_ext), gv.mean(data_ext)+gv.sdev(data_ext), facecolor=colors[count], alpha=0.2)

            count += 1
        
        ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, borderpad=0.3, \
            ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

        plt.ylim([0.03, 1.2])
        plt.yscale('log')
        plt.xlabel('$b_{\perp}$')
        plt.show()


#=============================================================#
#                    Analysis Tools                           # 
#=============================================================#

class Analysis_Tools(object):
    
    def colors(self):
        colors = ['orange','dodgerblue','blueviolet','deeppink','royalblue','rosybrown', \
            'fuchsia','red','green','cyan', 'orange','dodgerblue',\
            'blueviolet','deeppink','indigo','rosybrown','greenyellow','cyan','fuchsia',\
            'royalblue','red','green']
        return colors

    def reAv(self, data): # 画图的时候用的
        cen =[]; err= []
        for i in range(0, len(data)):
            cen.append(data[i].mean)
            err.append(data[i].sdev)
        cen = np.array(cen)
        err = np.array(err)
        return cen, err

    def confmean(self, a): # 对cfgs求平均，cfg在第0轴
        return np.mean(a,axis=0)

    ## 这里bootstrap只考虑输入resample list的情况
    def bootstrap(self, data, N_resample, resamp_list):
        nf, nt = data.shape
        if(resamp_list.shape[0]!=data.shape[0]*N_resample):
            raise IndexError("resamp_list的数目必须等于组态数乘以N_resample！")
        #list = np.random.randint(nf, size = N_resample * nf)
        dataMat = data[resamp_list,:]
        dataMat = np.reshape(dataMat, (N_resample, nf, nt))
        return np.mean(dataMat, axis=1)

    def jackknife(self, data): #data shape: (n_conf * n_t)
        nf, nt = data.shape #data shape: (n nf * n_t) 
        cv = np.mean(data, 0) #将所有组态平均，cv shape: (1 * nt) 
        cv = np.broadcast_to(cv, (nf, nt)) #广播将 cv 重新扩充为 data shape(n_cnf* n_t) 
        jac = (nf * cv - data) / nf #(mean[N,:]*n_conf - data[N,:])/n_conf 等价于每次抽出 1 个数做平均 
        return jac #jac shape: (n_conf * n_t)

    def covmatrix(self, data): # 求协方差矩阵, cfg 在第 0 轴 
        n_measure = data.shape[0] # 测量数就是组态的数目 
        xavg = self.confmean(data) 
        cov = data - np.array([xavg] * n_measure, float) 
        cov = np.matmul(cov.T, cov) 
        return cov
    
class Correlated_distributions(object):
    ## based on Jinchen's work

    def bootstrap(self, conf_ls, N_re):
        N_conf = len(conf_ls)
        conf_re = []
        for times in range(N_re):
            idx_ls = np.random.randint(N_conf, size=N_conf)
            temp = []
            for idx in idx_ls:
                temp.append(conf_ls[idx])
            conf_re.append( np.average(temp, axis=0) )

        return np.array(conf_re)

    def gv_to_samples(self, gv_ls, N_samp):
        '''
        transform gvar to bs samples
        '''
        samp_ls = []
        for var in gv_ls:
            samp = np.random.normal(loc=var.mean, scale=var.sdev, size=N_samp)
            samp_ls.append(samp)

        samp_ls = np.array(samp_ls).T

        return samp_ls

    def gv_to_samples_corr(self, gv_ls, N_samp):
        '''
        transform gvar to bs samples with correlation
        '''
        mean = [v.mean for v in gv_ls]
        cov_m = gv.evalcov(gv_ls)
        rng = np.random.default_rng()

        samp_ls = rng.multivariate_normal(mean, cov_m, size=N_samp)

        return samp_ls

    def add_corr(self, conf_ls, gv_ls):
        corr = gv.evalcov(gv_ls)
        conf_gv = gv.gvar(conf_ls, corr)

        return conf_gv


if __name__ == '__main__':
    soft_f = Soft_Function()
    soft_f.ori_SF()
    soft_f.plot_interpolated_SF()

    b_list, new_data = soft_f.SF_interpolate(mom=2)

    bls = b_list/0.12
    for i in range(1, 6):
        check_index = bisect(bls, i)
        print( bls[check_index] )
        print( new_data[check_index] )

# %%
# %%
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate


class INV_MATCHING_MOM:
    def __init__(self, x_ls, x_ls_mat, y_ls_mat):
        self.x_ls = x_ls  # for original quasi
        self.x_ls_mat = x_ls_mat  # for quasi in matching
        self.y_ls_mat = y_ls_mat  # for light-cone in matching

        # parameters setting
        cf = 4 / 3
        nf = 3
        b0 = 11 - 2 / 3 * nf
        mu_f = 2  # GeV, for factorization
        lms = 0.24451721864451428  # Lambda_MS
        alphas = 2 * np.pi / (b0 * np.log(mu_f / lms))

        self.mom_to_pz = (
            0.215  # mom=8 corresponding to pz=1.72, pz=2pi / (0.09*64) * mom=8 * 0.197
        )
        self.alphas_cf_div_2pi = alphas * cf / (2 * np.pi)
        self.mu_f = mu_f

    def interp_1d(self, x_in, y_in, x_out, method="linear"):  # interpolation
        f = interpolate.interp1d(x_in, y_in, kind=method)
        y_out = f(x_out)

        return y_out

    def main(self, pz, quasi_mom_ls):
        self.pz = pz
        kernel = self.matching_kernel()

        lc_mom_ls = []
        print(
            ">>> matching in the momentum space of mom " + str(self.pz / self.mom_to_pz)
        )
        for n_conf in tqdm(range(len(quasi_mom_ls))):
            quasi = np.array(quasi_mom_ls[n_conf])
            quasi = self.interp_1d(
                self.x_ls, quasi, self.x_ls_mat, method="cubic"
            )  # interpolation to match a big matrix
            dot = np.dot(kernel, quasi)  ###
            lc = self.interp_1d(
                self.y_ls_mat, dot, self.x_ls, method="cubic"
            )  # interpolation back to previous x_ls
            lc_mom_ls.append(lc)

        return lc_mom_ls

    def matching_kernel(self):
        x_ls = self.x_ls_mat  # for quasi
        y_ls = self.y_ls_mat  # for light-cone

        def H1(x, y):
            return (1 + x - y) / (y - x) * (1 - x) / (1 - y) * np.log(
                (y - x) / (1 - x)
            ) + (1 + y - x) / (y - x) * x / y * np.log((y - x) / (-x))

        def H2(x, y):
            return (1 + y - x) / (y - x) * x / y * np.log(
                4 * x * (y - x) * self.pz**2 / self.mu_f**2
            ) + (1 + x - y) / (y - x) * (
                (1 - x) / (1 - y) * np.log((y - x) / (1 - x)) - x / y
            )

        ### CB_matrix ###
        #################
        CB_matrix = np.zeros([len(x_ls), len(y_ls)])
        for idx1 in range(len(x_ls)):
            for idx2 in range(len(y_ls)):
                x = x_ls[idx1]
                y = y_ls[idx2]  #!#
                if abs(x - y) > 0.0001:
                    if x < 0 and y > 0 and y < 1:
                        CB_matrix[idx1][idx2] = H1(x, y)
                    elif x > 0 and y > x and y < 1:
                        CB_matrix[idx1][idx2] = H2(x, y)
                    elif y > 0 and y < x and x < 1:
                        CB_matrix[idx1][idx2] = H2(1 - x, 1 - y)
                    elif y > 0 and y < 1 and x > 1:
                        CB_matrix[idx1][idx2] = H1(1 - x, 1 - y)

        CB_matrix = CB_matrix * self.alphas_cf_div_2pi

        for idx in range(len(x_ls)):  # diagnoal input
            if CB_matrix[idx][idx] != 0:
                print("CB matrix diagnoal error")
            CB_matrix[idx][idx] = -np.sum([CB_matrix[i][idx] for i in range(len(x_ls))])

        ### extra term related to the modified hybrid method ###
        #################################################
        extra_term = np.zeros([len(x_ls), len(y_ls)])
        for idx1 in range(len(x_ls)):
            for idx2 in range(len(y_ls)):
                x = x_ls[idx1]
                y = y_ls[idx2]
                if y > 0 and y < 1 and abs(x - y) > 0.0001:
                    extra_term[idx1][idx2] = 3 / 2 * (1 / abs(x - y))

        for idx in range(len(x_ls)):  # diagnoal input
            if extra_term[idx][idx] != 0:
                print("extra term matrix diagnoal error")
            extra_term[idx][idx] = -np.sum(
                [extra_term[i][idx] for i in range(len(x_ls))]
            )

        extra_term = extra_term * self.alphas_cf_div_2pi

        ### delta(x-y) ###
        ##################
        identity = np.zeros([len(x_ls), len(y_ls)])

        for idx in range(len(x_ls)):
            identity[idx][idx] = 1

        C_matrix = (CB_matrix + extra_term) * (
            x_ls[1] - x_ls[0]
        ) + identity  # multiply by da to represent integral

        C_matrix_inverse = np.linalg.inv(C_matrix)

        return C_matrix_inverse


# an example of realization
x_ls = np.arange(-2 - 0.01, 3.02, 0.01)  # x after ft, for quasi before matching

delta = 0.00001
x_ls_mat = np.linspace(x_ls[0] + delta, x_ls[-1] - delta, 500)
y_ls_mat = np.linspace(x_ls[0], x_ls[-1], 500)

inv_matching_mom = INV_MATCHING_MOM(x_ls, x_ls_mat, y_ls_mat)

mom = 10
pz = 2.15  # GeV

# * read quasi data
quasi_mom_ls = gv.load("quasi_mom_ls.pkl")
print(np.shape(quasi_mom_ls))  # (n_conf, n_x)

# * inverse match to light-cone
lc_mom_ls = inv_matching_mom.main(pz, quasi_mom_ls)
print(np.shape(lc_mom_ls))


# * sample average
quasi_mom_avg = gv.dataset.avg_data(quasi_mom_ls, bstrap=True)
lc_mom_avg = gv.dataset.avg_data(lc_mom_ls, bstrap=True)


# * plot quasi and light-cone
plt_axes = [0.12, 0.12, 0.8, 0.8]


fig = plt.figure()
ax = plt.axes(plt_axes)
ax.fill_between(
    x_ls,
    [v.mean + v.sdev for v in lc_mom_avg],
    [v.mean - v.sdev for v in lc_mom_avg],
    label="light-cone",
    alpha=0.4,
)
ax.fill_between(
    x_ls,
    [v.mean + v.sdev for v in quasi_mom_avg],
    [v.mean - v.sdev for v in quasi_mom_avg],
    label="quasi",
    alpha=0.4,
)
ax.tick_params(direction="in", top="on", right="on")
ax.grid(linestyle=":")
plt.xlim(-0.5, 1.5)
plt.legend()
plt.show()

# %%

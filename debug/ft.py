# %%
import numpy as np
from scipy import interpolate

from liblattice.coor_to_mom.fourier_transform import sum_ft_re_im
from liblattice.general.constants import *


def interpolate_and_ft_single_sample(
    z_dep, z_array, z_array_int, x_ls, px, py, pz, a, Ls
):
    """
    Apply on single sample
    Interpolates the z-dependent data, performs Fourier transform, and returns the real and imaginary parts for each x value.

    Args:
        z_dep (ndarray): Array of z-dependent data.
        z_array (ndarray): Array of z values corresponding to z_dep.
        z_array_int (ndarray): Array of interpolated z values.
        x_ls (list): List of x values.
        px (float): Momentum component in the x-direction.
        py (float): Momentum component in the y-direction.
        pz (float): Momentum component in the z-direction.
        a (float): Lattice spacing.
        Ls (float): Lattice size.

    Returns:
        tuple: A tuple containing two lists: x_dep_re_ls and x_dep_im_ls.
            - x_dep_re_ls (list): List of real parts of the Fourier transformed data for each x value.
            - x_dep_im_ls (list): List of imaginary parts of the Fourier transformed data for each x value.
    """

    # complete the z<0 part
    z_array = np.concatenate((-z_array[::-1][:-1], z_array))
    z_dep = np.concatenate((z_dep[::-1][:-1], z_dep))

    # interpolate
    f = interpolate.interp1d(z_array, z_dep, kind="quadratic", fill_value="extrapolate")
    z_dep_int = f(z_array_int)

    # FT
    lambda_ls = (
        z_array_int * a * lat_unit_convert(np.sqrt(px**2 + py**2), a, Ls, "P") / GEV_FM
    )

    delta_lambda = lambda_ls[1] - lambda_ls[0]

    re_ls = z_dep_int
    im_ls = np.zeros_like(re_ls)  # set imag to zero because of the u-d unpolarization

    x_dep_re_ls = []
    x_dep_im_ls = []

    for x in x_ls:
        x_dep_re, x_dep_im = sum_ft_re_im(lambda_ls, re_ls, im_ls, delta_lambda, x)
        x_dep_re_ls.append(x_dep_re)
        x_dep_im_ls.append(x_dep_im)

    return x_dep_re_ls, x_dep_im_ls


# %%
if __name__ == "__main__":

    def test_interpolate_and_ft_single_sample():
        # Test case 1
        z_dep = np.array([1, 2, 3, 4, 5])
        z_array = np.array([0, 1, 2, 3, 4])
        z_array_int = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
        x_ls = [0, 1, 2]
        px = 0.5
        py = 0.5
        pz = 0.5
        a = 0.1
        Ls = 1.0

        x_dep_re_ls, x_dep_im_ls = interpolate_and_ft_single_sample(
            z_dep, z_array, z_array_int, x_ls, px, py, pz, a, Ls
        )

        print(x_dep_re_ls)

    # Test interpolate_and_ft_single_sample function
    test_interpolate_and_ft_single_sample()
# %%

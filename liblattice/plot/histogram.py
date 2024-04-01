# %%
import matplotlib.pyplot as plt
import numpy as np
import gvar as gv
from liblattice.general.plot_settings import *
from liblattice.preprocess.resampling import bs_ls_avg

def hist_comparison(data_ls, label_ls, title, bins, scale, save=False):
    """
    Plot a histogram comparing multiple datasets.

    Parameters:
    - data_ls (list): A list of numpy arrays representing the datasets.
    - label_ls (list): A list of labels for each dataset.
    - title (str): The title of the plot.
    - bins (int or sequence): The number of bins or a sequence of bin edges.
    - scale (float): The scaling factor to adjust the Gaussian curve.
    - save (bool, optional): Whether to save the plot as a PDF file. Default is False.

    Returns:
    None
    """
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    for id, (data, label) in enumerate(zip(data_ls, label_ls)):
        plt.hist(
            data,
            bins=bins,
            label=f"{label}",
            alpha=0.5,
            color=color_ls[id],
        )
        # Add a curve to represent a Gaussian distribution with the mean and sdev of the data sets
        data_avg = bs_ls_avg(data)
        mean = gv.mean(data_avg)
        sdev = gv.sdev(data_avg)
        curve_x = np.linspace(mean - 3 * sdev, mean + 3 * sdev, 100)
        curve_y = np.exp(-0.5 * ((curve_x - mean) / sdev) ** 2) / (sdev * np.sqrt(2 * np.pi)) * scale
        plt.plot(curve_x, curve_y * len(data) * (curve_x[1] - curve_x[0]), color=color_ls[id])

    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    plt.legend()
    plt.title(title, **fs_p)
    if save:
        plt.savefig(f"figs/{title}.pdf")
    plt.show()
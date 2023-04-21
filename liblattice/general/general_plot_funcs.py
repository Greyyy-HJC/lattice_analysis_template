"""
Here are general plot functions for liblattice.
"""

import matplotlib.pyplot as plt
from .plot_settings import *


# * default plot axes for general plots
plt_axes = [0.12, 0.12, 0.8, 0.8]  # left, bottom, width, height
fs_p = {"fontsize": 13}  # font size of text, label, ticks
ls_p = {"labelsize": 13}


def errorbar_plot(x, y, yerr, title, ylim=None, save=True):
    """Make a general errorbar plot, default save to `output/plots/`.

    Args:
        x (list): list of float x values
        y (list): list of float y values
        yerr (list): list of float yerr values
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
    """

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    ax.errorbar(x, y, yerr, **errorb)
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("output/plots/" + title + "_err.pdf", transparent=True)


def fill_between_plot(x, y, yerr, title, ylim=None, save=True):
    """Make a general fill_between plot, default save to `output/plots/`.

    Args:
        x (list): list of float x values
        y (list): list of float y values
        yerr (list): list of float yerr values
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
    """

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    ax.fill_between(
        x,
        [y[i] + yerr[i] for i in range(len(y))],
        [y[i] - yerr[i] for i in range(len(y))],
        alpha=0.4
    )
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("output/plots/" + title + "_fill.pdf", transparent=True)





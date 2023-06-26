"""
Here are general plot functions for liblattice.
"""
import numpy as np
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
        alpha=0.4,
    )
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("output/plots/" + title + "_fill.pdf", transparent=True)


def errorbar_ls_plot(x_ls, y_ls, yerr_ls, label_ls, title, ylim=None, save=True):
    """Make a general errorbar plot with multiple lines, default save to `output/plots/`.

    Args:
        x_ls (list): list of list of float x values
        y_ls (list): list of list of float y values
        yerr_ls (list): list of list of float yerr values
        label_ls (list): list of str labels
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
    """

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    for x_ls, y_ls, yerr_ls, label_ls in zip(x_ls, y_ls, yerr_ls, label_ls):
        ax.errorbar(x_ls, y_ls, yerr_ls, label=label_ls, **errorb)
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("output/plots/" + title + "_err_ls.pdf", transparent=True)


def fill_between_ls_plot(x_ls, y_ls, yerr_ls, label_ls, title, ylim=None, save=True):
    """Make a general fill_between plot with multiple lines, default save to `output/plots/`.

    Args:
        x_ls (list): list of list of float x values
        y_ls (list): list of list of float y values
        yerr_ls (list): list of list of float yerr values
        label_ls (list): list of str labels
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
    """

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    for x_ls, y_ls, yerr_ls, label_ls in zip(x_ls, y_ls, yerr_ls, label_ls):
        ax.fill_between(
            x_ls,
            [y_ls[i] + yerr_ls[i] for i in range(len(y_ls))],
            [y_ls[i] - yerr_ls[i] for i in range(len(y_ls))],
            alpha=0.4,
            label=label_ls,
        )
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("output/plots/" + title + "_fill_ls.pdf", transparent=True)


def errorbar_fill_between_ls_plot(
    x_ls, y_ls, yerr_ls, label_ls, plot_style_ls, title, ylim=None, save=True
):
    """Make a general errorbar & fill_between plot with multiple lines, default save to `output/plots/`.

    Args:
        x_ls (list): list of list of float x values
        y_ls (list): list of list of float y values
        yerr_ls (list): list of list of float yerr values
        label_ls (list): list of str labels
        plot_style_ls (list): list of str plot styles, 'errorbar' or 'fill_between'
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
    """

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    for x_ls, y_ls, yerr_ls, label_ls, plot_style in zip(
        x_ls, y_ls, yerr_ls, label_ls, plot_style_ls
    ):
        if plot_style == "errorbar":
            ax.errorbar(x_ls, y_ls, yerr_ls, label=label_ls, **errorb)
        elif plot_style == "fill_between":
            ax.fill_between(
                x_ls,
                [y_ls[i] + yerr_ls[i] for i in range(len(y_ls))],
                [y_ls[i] - yerr_ls[i] for i in range(len(y_ls))],
                alpha=0.4,
                label=label_ls,
            )
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("output/plots/" + title + "_err_fill_ls.pdf", transparent=True)


def stability_plot(x_ls, gv_y_ls, Q_ls, logGBF_ls, title, chose_idx, save=True):
    """
    This is a general stability plot function, with three subplots: matrix elements, Q, logGBF.
    The input should be x list, gvar y list, Q list, logGBF list, chose_idx.
    chose_idx is the index in the x list, which indicates the fit that you choose to use.
    """

    # * Define the height ratios for each subplot
    heights = [3, 1, 1]

    # * Create the subplots and set the height ratios
    fig, axs = plt.subplots(
        3, 1, sharex=True, figsize=fig_size, gridspec_kw={"height_ratios": heights}
    )

    # * Plot the data on each subplot
    axs[0].errorbar(
        x_ls, [v.mean for v in gv_y_ls], [v.sdev for v in gv_y_ls], **errorb
    )

    # * Plot the chosen fit
    upper = gv_y_ls[chose_idx].mean + gv_y_ls[chose_idx].sdev
    lower = gv_y_ls[chose_idx].mean - gv_y_ls[chose_idx].sdev

    axs[0].fill_between(
        x_ls,
        np.ones_like(x_ls) * upper,
        np.ones_like(x_ls) * lower,
        color=grey,
        alpha=0.4,
    )

    axs[1].scatter(x_ls, Q_ls, marker="X", facecolors="none", edgecolors="k", s=20)
    axs[1].plot(x_ls, 0.1 * np.ones_like(x_ls), "r--", linewidth=1)
    axs[2].scatter(x_ls, logGBF_ls, marker="o", facecolors="none", edgecolors="k", s=20)

    # Add labels to the x- and y-axes
    # axs[0].set_ylabel(r'$\Delta_{GMO}$', font)
    axs[1].set_ylabel(r"$Q$", font_config)
    axs[2].set_ylabel(r"$logGBF$", font_config)

    for i in range(3):
        axs[i].tick_params(direction="in", top="on", right="on", **ls_p)
        axs[i].grid(linestyle=":")

    plt.subplots_adjust(hspace=0)
    axs[0].set_title(title, font_config)

    if save == True:
        plt.savefig("fig/" + title + ".pdf", transparent=True)
    # Display the plot
    plt.show()

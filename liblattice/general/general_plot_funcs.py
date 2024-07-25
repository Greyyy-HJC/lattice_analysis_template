"""
Here are general plot functions for liblattice.
"""
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
from .plot_settings import *


# * default plot axes for general plots
plt_axes = [0.12, 0.12, 0.8, 0.8]  # left, bottom, width, height
fs_p = {"fontsize": 13}  # font size of text, label, ticks
ls_p = {"labelsize": 13}


def errorbar_plot(x, y, yerr, title, ylim=None, save=True, head=None):
    """Make a general errorbar plot, default save to `output/plots/`.

    Args:
        x (list): list of float x values
        y (list): list of float y values
        yerr (list): list of float yerr values
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
    ax.errorbar(x, y, yerr, marker='x', **errorb)
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)

    if save == True:
        plt.savefig("../output/plots/" + title + "_err.pdf", transparent=True)


def fill_between_plot(x, y, yerr, title, ylim=None, save=True, head=None):
    """Make a general fill_between plot, default save to `output/plots/`.

    Args:
        x (list): list of float x values
        y (list): list of float y values
        yerr (list): list of float yerr values
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
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

    if save == True:
        plt.savefig("../output/plots/" + title + "_fill.pdf", transparent=True)


def errorbar_ls_plot(x_ls, y_ls, yerr_ls, label_ls, title, ylim=None, save=True, head=None):
    """Make a general errorbar plot with multiple lines, default save to `output/plots/`.

    Args:
        x_ls (list): list of list of float x values
        y_ls (list): list of list of float y values
        yerr_ls (list): list of list of float yerr values
        label_ls (list): list of str labels
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
    for x_ls, y_ls, yerr_ls, label_ls in zip(x_ls, y_ls, yerr_ls, label_ls):
        ax.errorbar(x_ls, y_ls, yerr_ls, marker='x', label=label_ls, **errorb)
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("../output/plots/" + title + "_err_ls.pdf", transparent=True)


def fill_between_ls_plot(x_ls, y_ls, yerr_ls, label_ls, title, ylim=None, save=True, head=None):
    """Make a general fill_between plot with multiple lines, default save to `output/plots/`.

    Args:
        x_ls (list): list of list of float x values
        y_ls (list): list of list of float y values
        yerr_ls (list): list of list of float yerr values
        label_ls (list): list of str labels
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
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
        plt.savefig("../output/plots/" + title + "_fill_ls.pdf", transparent=True)


def errorbar_fill_between_ls_plot(
    x_ls, y_ls, yerr_ls, label_ls, plot_style_ls, title, ylim=None, save=True, head=None
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
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
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
        plt.savefig("../output/plots/" + title + "_err_fill_ls.pdf", transparent=True)



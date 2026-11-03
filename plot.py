import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import paths
from constants import TNG_H

# Set graph preferences
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


def tpcf_plot(rbin_centers, xi, errors, title, set_color):
    """Creates plot of tpcf versus radial bin centers
    Args:
        rbin_centers: radial bin centers
        xi: tpcf computed for mass sample
        errors: errors from compute_jackknife_errors
        title: 'your title' that includes mass sample
        set_color: color choice for plot, i.e. 'CO'
    Return:
        fig: figure object containing the plot
    """
    fig, ax = plt.subplots()
    ax.scatter(rbin_centers, xi, color=set_color)
    ax.plot(rbin_centers, xi, color=set_color)
    ax.fill_between(rbin_centers, xi-errors, xi+errors, alpha=0.4, color=set_color)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(r'$r $  $\rm{[Mpc/h]}$')
    ax.set_ylabel(r'$\xi(r)$')
    return fig


def tpcf_sidebyside_plots(rbin_centers, xi, errors, title1, title2, color_list, label_list):
    """Creates figure with side-by-side tpcf plots with and without jackknife errors
    Args: 
        rbin_centers: radial bin centers
        xi: tpcf computed for mass samples
        errors: errors from compute_jackknife_errors
        title1: string with title for plot without errors
        title2: string with title for plot with errors
        color_list: list of colors for different samples
        label_list: list of labels for different samples
     Return:
        fig: figure object containing the two plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = color_list
    labels = label_list
    for ii in range(len(xi)):
        axes[0].loglog(rbin_centers, xi[ii], color=colors[ii], label=labels[ii])
    
    axes[0].set_xlabel(r'$r$  $\rm{[Mpc/h]}$')
    axes[0].set_ylabel(r'$\xi(r)$')
    axes[0].set_title(title1)
    axes[0].legend(loc='best')
    
    for ii in range(len(xi)):
        axes[1].loglog(rbin_centers, xi[ii], color=colors[ii], label=labels[ii])
        axes[1].scatter(rbin_centers, xi[ii], color=colors[ii])
        axes[1].fill_between(rbin_centers, xi[ii]-errors[ii], xi[ii]+errors[ii], 
                             color=colors[ii],  alpha = 0.4)
        
    axes[1].set_xlabel(r'$r$  $\rm{[Mpc/h]}$')
    axes[1].set_title(title2)
    axes[1].legend(loc='best')
    return fig


def tpcf_ratio_grid(n_cols):
    """Creates a 2-row grid for xi(r) (top) and its ratio to the truth (bottom), one column per mass bin
    Args:
        n_cols: number of columns, i.e. len(mass_bins)
    Return:
        fig, axes: figure and (2, n_cols) array of axes, drawn into with tpcf_ratio_panel
    """
    return plt.subplots(2, n_cols, figsize=(19, 9), sharex=True, sharey='row',
                        gridspec_kw={'height_ratios': [3, 1]})


def tpcf_ratio_panel(ax, rax, rbin_centers, xi_truth, err_truth, truth_label, truth_color, curves,
                     truth_alpha=0.2):
    """Draws the truth xi(r) and predicted curves on one tpcf_ratio_grid column; can be called repeatedly
    Args:
        ax, rax: xi(r) axis and ratio axis of the column
        rbin_centers: radial bin centers
        xi_truth: truth tpcf, drawn as a line with a jackknife error band
        err_truth: errors from compute_tpcf_jackknife_errors for xi_truth
        truth_label: legend label for the truth
        truth_color: color of the truth line and bands
        curves: list of dicts drawn in order, each either a jackknife prediction
            {'r', 'xi', 'err', 'color', 'marker', 'linestyle', 'label'} drawn as error bars, or a sampling
            prediction {'median', 'lo', 'hi', 'color', 'label'} drawn as a dash-dot line with a band
        truth_alpha: alpha of the truth error bands
    """
    ax.plot(rbin_centers, xi_truth, marker='o', color=truth_color, label=truth_label)
    ax.fill_between(rbin_centers, xi_truth - err_truth, xi_truth + err_truth, color=truth_color,
                    alpha=truth_alpha)
    rax.fill_between(rbin_centers, 1 - err_truth / xi_truth, 1 + err_truth / xi_truth, color=truth_color,
                     alpha=truth_alpha)

    for c in curves:
        if 'xi' in c:
            ax.errorbar(c['r'], c['xi'], yerr=c['err'], color=c['color'], capsize=2, marker=c['marker'],
                        linestyle=c['linestyle'], label=c['label'])
            rax.errorbar(c['r'], c['xi'] / xi_truth, yerr=c['err'] / xi_truth, color=c['color'], capsize=2,
                         marker=c['marker'], linestyle=c['linestyle'])
        else:
            ax.plot(rbin_centers, c['median'], color=c['color'], linestyle='-.', label=c['label'])
            ax.fill_between(rbin_centers, c['lo'], c['hi'], color=c['color'], alpha=0.2)
            rax.plot(rbin_centers, c['median'] / xi_truth, color=c['color'], linestyle='-.')
            rax.fill_between(rbin_centers, c['lo'] / xi_truth, c['hi'] / xi_truth, color=c['color'], alpha=0.2)


def finish_tpcf_ratio_grid(fig, axes, mass_bins, title_prefix=''):
    """Adds scales, labels, titles, legend and spacing to a filled tpcf_ratio_grid
    Args:
        fig, axes: figure and axes from tpcf_ratio_grid
        mass_bins: dict of bin key -> (low, high) log stellar mass edges, one per column
        title_prefix: string placed before each bin title, i.e. 'Red: '
    """
    for i, (low, high) in enumerate(mass_bins.values()):
        ax, rax = axes[0, i], axes[1, i]
        rax.axhline(1, color='k', linewidth=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(fr'{title_prefix}${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
        rax.set_xlabel(r'$r \, [\mathrm{Mpc}/h]$')
        if i == 0:
            ax.set_ylabel(r'$\xi(r)$')
            rax.set_ylabel(r'$\xi_{\rm pred} / \xi_{\rm hydro}$')
        if i == len(mass_bins) - 1:
            ax.legend(fontsize=8)

    axes[1, 0].set_ylim(0.5, 1.5)
    fig.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95, bottom=0.1, top=0.9)


def color_hist_by_bin(mass_bins, truth, preds, color_cut, bins):
    """Creates one g-i histogram panel per mass bin, true color against predicted colors
    Args:
        mass_bins: dict of bin key -> (low, high) log stellar mass edges
        truth: dict of bin key -> true colors in that bin
        preds: list of (label, dict of bin key -> predicted colors, dict of extra hist kwargs) tuples
        color_cut: red/blue g-i cut, drawn as a vertical line
        bins: histogram bin edges shared by all panels
    Return:
        fig: figure object containing the plots
    """
    fig, axes = plt.subplots(1, len(mass_bins), figsize=(18, 4))
    for i, (key, (low, high)) in enumerate(mass_bins.items()):
        ax = axes[i]
        ax.hist(truth[key], bins=bins, density=True, color='gray', alpha=0.4, label='True color (Hydro)')
        for label, colors, kwargs in preds:
            ax.hist(colors[key], bins=bins, density=True, histtype='step', linewidth=2, label=label, **kwargs)
        ax.axvline(color_cut, color='k', linestyle='--')
        ax.set_title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
        ax.set_xlabel(r'$g - i$ color')
        ax.set_ylabel('Normalized frequency')
        if i == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_named_figures(figures, directory=paths.FIGURES_DIR):
    """Saves figures as PNG files named after their keys in given directory
    Args:
        figures: dict of name -> figure, saved as <name>.png
        directory: creates default path paths.FIGURES_DIR
    Return:
        filenames: list that stores the paths of the saved PNG files
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    filenames = []
    for name, fig in figures.items():
        filename = os.path.join(directory, f'{name}.png')
        fig.savefig(filename, format='png', bbox_inches = 'tight', pad_inches = .25)
        plt.close(fig)
        filenames.append(filename)
    return filenames


def save_figures_as_png(figures, number, directory=paths.FIGURES_DIR):
    """Saves figures as PNG files in given directory
    Args:
        figures: list of figures to be saved
        number: number corresponding to starting figure, ie 1
        directory: creates default path paths.FIGURES_DIR
    Return:
        filenames: list that stores the paths of the saved PNG files
    """
    return save_named_figures({f'figure{i}': fig for i, fig in enumerate(figures, start=number)}, directory)


def hist_plot(data, title, xlabel):
    """Creates histogram plot for distributions
    Args:
        data: data for plotting, i.e. t['StellarMass']
        title: title, i.e. Stellar Mass Distribution
        xlabel: label for x-axis
    Return:
        fig: figure object containing plot
    """
    fig, ax = plt.subplots()
    ax.hist(data, bins=55, color='skyblue', edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    return fig


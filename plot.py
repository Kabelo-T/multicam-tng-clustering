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


def save_figures_as_png(figures, number, directory=paths.FIGURES_DIR):
    """Saves figures as PNG files in given directory
    Args:
        figures: list of figures to be saved
        number: number corresponding to starting figure, ie 1
        directory: creates default path paths.FIGURES_DIR
    Return:
        filenames: list that stores the paths of the saved PNG files
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)  
    filenames = []
    for i, fig in enumerate(figures, start=number):
        filename = os.path.join(directory, f'figure{i}.png')
        fig.savefig(filename, format='png', bbox_inches = 'tight', pad_inches = .25)
        plt.close(fig)
        filenames.append(filename) 
    return filenames


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


import math
import itertools

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp

from utils import add_mask

HETDEX_LON_RANGE = [158, 234]
HETDEX_LAT_RANGE = [43, 60]


def plot_many_data_correlations(experiment_dict, correlation_symbol, x_min=0, x_max=None, y_min=None, y_max=None,
                                x_scale='linear', y_scale='linear', legend_loc='upper right'):
    # Assuming the same theory across experiments
    experiment = list(experiment_dict.values())[0]
    data_to_plot = experiment.theory_correlations[correlation_symbol] - experiment.noise_curves[correlation_symbol]
    plt.plot(experiment.l_arr, data_to_plot, 'r', label='theory', markersize=2)

    # Iterate experiments on data
    marker = itertools.cycle(('o', 'v', 's', 'p', '*'))
    for experiment_name, experiment in experiment_dict.items():
        ell_arr = experiment.binnings[correlation_symbol].get_effective_ells()
        noise = experiment.noise_curves[correlation_symbol][0] if correlation_symbol == 'gg' else 0
        data_to_plot = experiment.data_correlations[correlation_symbol] - noise
        plt.errorbar(ell_arr, data_to_plot, marker=next(marker), linestyle='', label=experiment_name)
        if noise:
            plt.axhline(y=noise, color='grey', label='noise')

    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$C_\\ell^{{{}}}$'.format(correlation_symbol), fontsize=16)
    plt.legend(loc=legend_loc, ncol=2, labelspacing=0.1)
    plt.grid()
    plt.show()


def plot_correlation(experiment, correlation_symbol, x_min=0, x_max=None, y_min=None, y_max=None, x_scale='linear',
                     y_scale='linear', title=None, with_error=True):
    # Data error bars
    y_err = None
    if with_error:
        covariance_symbol = '-'.join([correlation_symbol, correlation_symbol])
        if covariance_symbol in experiment.covariance_matrices:
            y_err = np.sqrt(np.diag(experiment.covariance_matrices[covariance_symbol]))

    # Data
    if correlation_symbol in experiment.data_correlations:
        ell_arr = experiment.binnings[correlation_symbol].get_effective_ells()
        noise = experiment.noise_curves[correlation_symbol][0] if correlation_symbol == 'gg' else 0
        data_to_plot = experiment.data_correlations[correlation_symbol] - noise
        plt.errorbar(ell_arr, data_to_plot, yerr=y_err, fmt='ob', label='data', markersize=2)
        if noise:
            plt.axhline(y=noise, color='grey', label='noise')

    # Theory
    if correlation_symbol in experiment.theory_correlations:
        data_to_plot = experiment.theory_correlations[correlation_symbol] - experiment.noise_curves[correlation_symbol]
        plt.plot(experiment.l_arr, data_to_plot, 'r', label='theory', markersize=2)

    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$C_\\ell^{{{}}}$'.format(correlation_symbol), fontsize=16)
    plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
    plt.grid()
    plt.title(title)
    plt.show()


def plot_correlation_matrix(experiment):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(experiment.inference_correlation, interpolation=None)
    fig.colorbar(cax)

    half_ticks = []
    lines = []
    next_start = -0.5
    for correlation_symbol in experiment.correlation_symbols:
        n_ells = experiment.n_ells[correlation_symbol]
        half_ticks.append(next_start + n_ells / 2)
        next_start += n_ells
        lines.append(next_start)
    lines = lines[:-1]

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks(half_ticks)
    ax.set_yticks(half_ticks)
    for x in lines:
        plt.axvline(x=x, color='black')
        plt.axhline(y=x, color='black')
    correlation_symbols = [pretty_print_corr_symbol(corr_symbol) for corr_symbol in experiment.correlation_symbols]
    ax.set_xticklabels(correlation_symbols)
    ax.set_yticklabels(correlation_symbols)


def pretty_print_corr_symbol(correlation_symbol):
    math_symbols = {'g': 'g', 'k': '\kappa', 't': 'T'}
    symbol_a = correlation_symbol[0]
    symbol_b = correlation_symbol[1]
    return r'${} \times {}$'.format(math_symbols[symbol_a], math_symbols[symbol_b])


def my_mollview(map, fwhm=0, unit=None, cmap='jet', zoom=False):
    if fwhm > 0:
        map = hp.sphtfunc.smoothing(map, fwhm=math.radians(fwhm))
    view_func = hp.zoomtool.mollzoom if zoom else hp.mollview
    view_func(map, cmap=cmap, unit=unit)
    hp.graticule()


def plot_hetdex_image(map, additional_mask=None, title=None, cmap='viridis', fwhm=0.0, norm=None):
    if fwhm > 0:
        map = hp.sphtfunc.smoothing(map, fwhm=fwhm)

    if additional_mask is not None:
        map = add_mask(map, additional_mask)

    hp.visufunc.cartview(map=map, xsize=1000, lonra=HETDEX_LON_RANGE, latra=HETDEX_LAT_RANGE, title=title,
                         cmap=cmap, badcolor='gray', bgcolor='white', cbar=False, coord='C', norm=norm)
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    fig.colorbar(image, orientation='horizontal', aspect=40, pad=0.08, ax=ax)
    plt.show()

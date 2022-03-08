import math
import itertools

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp

from utils import add_mask

HETDEX_LON_RANGE = [158, 234]
HETDEX_LAT_RANGE = [43, 60]


def plot_correlation_comparison(correlations_a, correlations_b, correlation_symbols, correlation_names,
                                is_raw=[False, False], error_method='gauss', x_min=0, x_max=None, y_min=None,
                                y_max=None, x_scale='linear',
                                y_scale='linear', title=None, with_error=True):
    # Data
    ell_arr = correlations_a['l']
    corr_a = correlations_a['Cl_{}'.format(correlation_symbols[0])]
    noise_a = correlations_a['nl_{}'.format(correlation_symbols[0])]
    corr_b = correlations_b['Cl_{}'.format(correlation_symbols[1])]
    noise_b = correlations_b['nl_{}'.format(correlation_symbols[1])]

    # Remove noise for all calculations
    corr_a = corr_a - noise_a
    corr_b = corr_b - noise_b
    if is_raw[0]:
        corr_a += correlations_a['nl_{}_multicomp'.format(correlation_symbols[0])]
    if is_raw[1]:
        corr_b += correlations_b['nl_{}_multicomp'.format(correlation_symbols[1])]

    # Error bars
    error_a = correlations_a['error_{}_{}'.format(correlation_symbols[0], error_method)]
    error_b = correlations_b['error_{}_{}'.format(correlation_symbols[1], error_method)]

    # Upper plot, two correlation functions
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [2, 1]})
    name_a = '$(C_\\ell^{{{}}})_{{{}}}$'.format(correlation_symbols[0][:2], correlation_names[0])
    name_b = '$(C_\\ell^{{{}}})_{{{}}}$'.format(correlation_symbols[1][:2], correlation_names[1])
    axs[0].errorbar(ell_arr, corr_a, yerr=error_a, fmt='ob', markersize=2, label=name_a)
    axs[0].errorbar(ell_arr, corr_b, yerr=error_b, fmt='og', markersize=2, label=name_b)

    axs[0].set_xlim(left=x_min, right=x_max)
    # plt.ylim(ymin=y_min, ymax=y_max)
    axs[0].set_xscale('linear')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('$C_\\ell$', fontsize=16)
    axs[0].grid()
    axs[0].legend(loc='upper right')

    # Lower plot, ratio of the correlations
    ratio = corr_a / corr_b
    error = np.sqrt(((error_a / corr_a) ** 2 + (error_b / corr_b) ** 2)) * corr_a / corr_b if with_error else None

    axs[1].errorbar(ell_arr, ratio, yerr=error, fmt='or', markersize=2)
    axs[1].axhline(1, color='green')

    axs[1].set_xlim(left=x_min, right=x_max)
    axs[1].set_ylim(bottom=y_min, top=y_max)
    axs[1].set_xscale(x_scale)
    axs[1].set_yscale(y_scale)
    axs[1].set_xlabel('$\\ell$', fontsize=16)
    y_label = '$(C_\\ell^{{{}}})_{{{}}} / (C_\\ell^{{{}}})_{{{}}}$'.format(
        correlation_symbols[0][:2], correlation_names[0], correlation_symbols[1][:2], correlation_names[1]
    )
    axs[1].set_ylabel(y_label, fontsize=16)
    axs[1].grid()
    plt.title(title)
    plt.show()


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
        data_to_plot = experiment.data_correlations[correlation_symbol] - experiment.noise_curves[correlation_symbol]
        plt.errorbar(ell_arr, data_to_plot, marker=next(marker), linestyle='', label=experiment_name)

    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$C_\\ell^{{{}}}$'.format(correlation_symbol), fontsize=16)
    plt.legend(loc=legend_loc, ncol=2, labelspacing=0.1)
    plt.grid()
    plt.show()


def plot_correlation(experiment, correlation_symbol, x_min=0, x_max=None, y_min=None, y_max=None, x_scale='linear',
                     y_scale='linear', title=None, with_error=True, is_raw=False, error_method='jackknife'):
    # Data error bars
    y_err = None
    if with_error and correlation_symbol in experiment.errors[error_method]:
        y_err = experiment.errors[error_method][correlation_symbol]

    # Data
    if correlation_symbol in experiment.data_correlations:
        ell_arr = experiment.binnings[correlation_symbol].get_effective_ells()
        noise = experiment.noise_decoupled[correlation_symbol]
        if is_raw:
            noise = noise - experiment.multicomp_noise
        correlation_dict = experiment.data_correlations
        data_to_plot = correlation_dict[correlation_symbol] - noise
        plt.errorbar(ell_arr, data_to_plot, yerr=y_err, fmt='ob', label='data', markersize=2)
        # Shot noise
        if correlation_symbol == 'gg':
            plt.plot(ell_arr, noise, color='grey', marker='o', label='noise', markersize=2)
        # ell range lines
        l_range = experiment.config.l_range[correlation_symbol]
        plt.axvline(l_range[0], label='ell range', color='green')
        plt.axvline(l_range[1], color='green')

    # TODO: change for decoupling and adding decoupled noise
    # Theory
    if correlation_symbol in experiment.theory_correlations:
        # TODO: refactor, effective ells used in line 48 too
        data_to_plot = experiment.theory_correlations[correlation_symbol] - experiment.noise_curves[correlation_symbol]
        dense_l_arr = experiment.l_arr
        eff_l_arr = experiment.binnings[correlation_symbol].get_effective_ells()
        ell_arr = dense_l_arr if dense_l_arr.shape[0] == data_to_plot.shape[0] else eff_l_arr
        plt.plot(ell_arr, data_to_plot, 'r', label='theory', markersize=2)

    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
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


# TODO: regions should be just positive values
def plot_jackknife_regions(experiment, regions):
    hp.visufunc.cartview(map=experiment.masks['g'], xsize=1000, badcolor='gray', bgcolor='white', cbar=False, norm=None,
                         cmap='viridis')
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    fig.colorbar(image, orientation='horizontal', aspect=40, pad=0.08, ax=ax)

    for region in regions:
        lon = region['lon']
        lat = region['lat']
        # direction = region['dir']

        # if direction == 'left':
        #     lon = [l if l < 180 else l - 360 for l in lon]

        # Create arrays of line anchors
        lon_arr = np.append(np.arange(lon[0], lon[1], (lon[1] - lon[0]) / lon[2]), lon[1])
        lat_arr = np.append(np.arange(lat[0], lat[1], (lat[1] - lat[0]) / lat[2]), lat[1])

        # if direction == 'right':
        lon_arr = [lon if lon < 180 else lon - 360 for lon in lon_arr]

        # Plot lonigtude lines
        lat_range = np.arange(lat_arr[0], lat_arr[-1])
        for lon in lon_arr:
            plt.plot([lon] * len(lat_range), lat_range, 'b')

        # Plot latitude lines
        lon = region['lon']
        # if direction == 'left':
        #     lon = [l if l < 180 else l - 360 for l in lon]

        lon_range = np.append(np.arange(lon[0], lon[1], (lon[1] - lon[0]) / 100), lon[1])

        # TODO: additional if on right transformation missing here
        lon_range_a = [lon for lon in lon_range if lon < 180]
        lon_range_b = [lon - 360 for lon in lon_range if lon > 180]

        for lat in lat_arr:
            plt.plot(lon_range_a, [lat] * len(lon_range_a), 'b')
            plt.plot(lon_range_b, [lat] * len(lon_range_b), 'b')


def my_mollview(map, fwhm=0, unit=None, cmap='jet', zoom=False, rot=None):
    if fwhm > 0:
        map = hp.sphtfunc.smoothing(map, fwhm=math.radians(fwhm))
    view_func = hp.zoomtool.mollzoom if zoom else hp.mollview
    view_func(map, cmap=cmap, unit=unit, rot=rot)
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

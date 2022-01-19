import os
import copy

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
from scipy.integrate import simps

from env_config import PROJECT_PATH
from scipy.optimize import curve_fit


def plot_redshift_distributions(redshift_distributions, styles, ylabel='dp/dz', bias_scales=None, xscale='linear',
                                legend_size=None):
    for flux_cut in [2, 1, 0.5]:
        for i, dist_name in enumerate(styles):
            if flux_cut in redshift_distributions[dist_name]:
                dist = redshift_distributions[dist_name][flux_cut]
                z_arr = dist['z']
                p_arr = dist['pz']

                label = dist_name
                scale = 1
                if bias_scales is not None and dist_name in bias_scales:
                    scale = bias_scales[dist_name][0]
                    scale_name = bias_scales[dist_name][1]
                    label = '{} * {}'.format(scale_name, dist_name)

                plt.plot(z_arr, p_arr * scale, styles[dist_name], label=label)

                if dist.get('pz_min') is not None:
                    pz_min = dist.get('pz_min')
                    pz_max = dist.get('pz_max')
                    plt.fill_between(z_arr, pz_min * scale, pz_max * scale, color=styles[dist_name][0], alpha=0.2)

                if dist.get('pz_sfg') is not None:
                    plt.plot(z_arr, dist.get('pz_sfg') * scale, styles[dist_name][0] + '--', label=label + ' SFG')
                    plt.plot(z_arr, dist.get('pz_agn') * scale, styles[dist_name][0] + '--', label=label + ' AGN')

        plt.title('{} mJy'.format(flux_cut))
        plt.xlabel('z')
        plt.ylabel(ylabel)
        plt.xscale(xscale)
        plt.legend(prop={'size': legend_size})
        plt.show()


def scale_with_bias(redshift_distributions, inverse=False):
    redshift_distributions = copy.deepcopy(redshift_distributions)
    for dist_name in redshift_distributions:
        for flux_cut in redshift_distributions[dist_name]:

            dist = redshift_distributions[dist_name][flux_cut]
            z_arr = dist['z']

            bias_arr = np.ones(len(z_arr))
            with open(os.path.join(PROJECT_PATH, 'cosmologies.yml'), 'r') as cosmology_file:
                cosmology_params = yaml.full_load(cosmology_file)['planck']
            cosmology = ccl.Cosmology(**cosmology_params)
            bias_arr = bias_arr / ccl.growth_factor(cosmology, 1. / (1. + np.array(z_arr)))

            if inverse:
                bias_arr = 1 / bias_arr

            dist['pz'] *= bias_arr

            if dist.get('pz_min') is not None:
                dist['pz_min'] *= bias_arr
                dist['pz_max'] *= bias_arr

            if dist.get('pz_sfg') is not None:
                dist['pz_sfg'] *= bias_arr
                dist['pz_agn'] *= bias_arr

    return redshift_distributions


def normalize_dists(redshift_distributions):
    redshift_distributions = copy.deepcopy(redshift_distributions)
    for dist_name in redshift_distributions:
        for flux_cut in redshift_distributions[dist_name]:

            dist = redshift_distributions[dist_name][flux_cut]
            area = simps(dist['pz'], dist['z'])
            dist['pz'] /= area

            if dist.get('pz_min') is not None:
                dist['pz_min'] /= area
                dist['pz_max'] /= area

            if dist.get('pz_sfg') is not None:
                dist['pz_sfg'] /= area
                dist['pz_agn'] /= area

    return redshift_distributions


def make_tomographer_fit(tomographer, p_0):
    popt, pcov = curve_fit(get_powerlaw_redshift, tomographer['z'], tomographer['dNdz_b'],
                           sigma=tomographer['dNdz_b_err'], p0=p_0, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def get_powerlaw_redshift(z_arr, z_sfg, a, r, n):
    return n * (z_arr ** 2) / (1 + z_arr) * (np.exp((-z_arr / z_sfg)) + r ** 2 / (1 + z_arr) ** a)


def make_tomographer_plot(tomographer, popt, perr, func=get_powerlaw_redshift, xscale='linear', ylabel='N * b',
                          add_bias=False):
    # Plot results
    z_max = 6
    z_step = 0.01
    z_min = 0.01
    z_max = z_max + z_step
    z_arr = np.arange(z_min, z_max, z_step)

    with open(os.path.join(PROJECT_PATH, 'cosmologies.yml'), 'r') as cosmology_file:
        cosmology_params = yaml.full_load(cosmology_file)['planck']
    cosmology = ccl.Cosmology(**cosmology_params)

    y = copy.copy(tomographer['dNdz_b'])
    y_err = copy.copy(tomographer['dNdz_b_err'])
    b_arr = 1
    if add_bias:
        b_arr = 1 / ccl.growth_factor(cosmology, 1. / (1. + tomographer['z']))
        y *= b_arr
        y_err *= b_arr
    plt.errorbar(tomographer['z'], y, y_err, fmt='g.', label='tomographer')

    y = func(z_arr, *popt)
    if add_bias:
        b_arr = 1 / ccl.growth_factor(cosmology, 1. / (1. + z_arr))
        y *= b_arr
    plt.plot(z_arr, y, 'r-', label='fit')

    to_modify = perr / 2
    # Revert a so it also gives upper boundary when added
    to_modify[1] *= -1
    to_modify[3] = 0

    y_a = func(z_arr, *(popt - to_modify))
    y_b = func(z_arr, *(popt + to_modify))
    if add_bias:
        y_a *= b_arr
        y_b *= b_arr
    plt.plot(z_arr, y_a, 'r--')
    plt.plot(z_arr, y_b, 'r--')

    plt.axhline(y=0, color='gray', linestyle='-')
    plt.legend()
    plt.xscale(xscale)
    plt.xlabel('z')
    plt.ylabel(ylabel)

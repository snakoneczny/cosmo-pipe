import math

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp

from utils import add_mask

HETDEX_LON_RANGE = [158, 234]
HETDEX_LAT_RANGE = [43, 60]


def plot_correlation(experiment, correlation_symbol, x_min=0, x_max=None, y_min=None, y_max=None, x_scale='linear',
                     y_scale='linear'):
    y_err = None
    covariance_symbol = '-'.join([correlation_symbol, correlation_symbol])
    if covariance_symbol in experiment.covariance_matrices:
        y_err = np.sqrt(np.diag(experiment.covariance_matrices[covariance_symbol]))

    # Data
    if correlation_symbol in experiment.data_correlations:
        ell_arr = experiment.binnings[correlation_symbol].get_effective_ells()
        data_to_plot = experiment.data_correlations[correlation_symbol] - experiment.noise_decoupled[correlation_symbol]
        plt.errorbar(ell_arr, data_to_plot, yerr=y_err, fmt='ob', label='data', markersize=2)

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
    plt.show()


def plot_correlation_matrix(experiment):
    plt.matshow(experiment.inference_correlation)
    plt.colorbar()


def my_mollview(map, fwhm=0, unit=None, cmap='jet'):
    if fwhm > 0:
        map = hp.sphtfunc.smoothing(map, fwhm=math.radians(fwhm))
    hp.mollview(map, cmap=cmap, unit=unit)
    hp.graticule()


def plot_cmb_lensing_hetdex(experiment):
    plot_hetdex_image(experiment.original_maps['k'], experiment.masks['g'], title='k', fwhm=math.radians(0.6))
    plot_hetdex_image(experiment.masks['k'], title='CMB mask')


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

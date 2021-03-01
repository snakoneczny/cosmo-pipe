import math

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp

from utils import add_mask

HETDEX_LON_RANGE = [158, 234]
HETDEX_LAT_RANGE = [43, 60]


def plot_correlations(experiment, x_scale='linear', y_scale='linear'):
    for spectra_symbol in experiment.all_correlation_symbols:
        # noise = experiment.noise_curves[spectra_symbol] if spectra_symbol == 'gg' else 0
        plot_correlation(experiment.binning, experiment.data_correlations[spectra_symbol] - experiment.noise_decoupled[spectra_symbol],
                         model_correlation=experiment.theory_correlations[spectra_symbol] - experiment.noise_curves[spectra_symbol],
                         covariance_matrix=experiment.covariance_matrices.get(
                             '-'.join([spectra_symbol, spectra_symbol])),
                         x_max=500, x_scale=x_scale, y_scale=y_scale, label=spectra_symbol)


def plot_correlation(binning, correlation, model_correlation=None, covariance_matrix=None, x_max=None, y_min=None,
                     label='gg', x_scale='linear', y_scale='linear'):
    y_err = None
    if covariance_matrix is not None:
        y_err = np.sqrt(np.diag(covariance_matrix))

    ell_arr = binning.get_effective_ells()
    plt.errorbar(ell_arr, np.fabs(correlation), yerr=y_err, fmt='ob', label='data', markersize=2)

    if model_correlation is not None:
        plt.plot(model_correlation, 'r', label='theory', markersize=2)

    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.xlim(xmax=x_max)
    plt.ylim(ymin=y_min)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$C_\\ell^{{{}}}$'.format(label), fontsize=16)
    plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
    plt.show()


def plot_lotss_hetdex(experiment):
    plot_hetdex_image(experiment.original_maps['g'], title='LOFAR counts map', fwhm=math.radians(0.6))
    plot_hetdex_image(experiment.noise_maps['g'], title='LOFAR noise map', fwhm=math.radians(0.6))
    plot_hetdex_image(experiment.noise_weight_maps['g'], title='LOFAR noise weight map', fwhm=math.radians(0.6))
    plot_hetdex_image(experiment.processed_maps['g'], title='LOFAR overdensity map', fwhm=math.radians(0.6))
    plot_hetdex_image(experiment.masks['g'], title='LOFAR mask')


def plot_cmb_lensing_hetdex(experiment):
    plot_hetdex_image(experiment.original_maps['k'], experiment.masks['g'], title='k', fwhm=math.radians(0.6))
    plot_hetdex_image(experiment.masks['k'], title='CMB mask')


def plot_hetdex_image(map, additional_mask=None, title=None, cmap='viridis', fwhm=0.0, norm=None):
    if fwhm > 0:
        map = hp.sphtfunc.smoothing(map, fwhm=fwhm)

    if additional_mask is not None:
        map = add_mask(map, additional_mask)

    hp.visufunc.cartview(map=map, xsize=1000, lonra=HETDEX_LON_RANGE, latra=HETDEX_LAT_RANGE, title=title,
                         cmap=cmap, badcolor='gray', bgcolor='white', cbar=True, coord='C', norm=norm)
    # fig = plt.gcf()
    # ax = plt.gca()
    # image = ax.get_images()[0]
    # fig.colorbar(image, orientation='horizontal', aspect=40, pad=0.08, ax=ax)
    # plt.xlabel('R.A.')
    # plt.ylabel('dec.')
    plt.show()

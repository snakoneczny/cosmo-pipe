import numpy as np
from matplotlib import pyplot as plt
import healpy as hp

from utils import add_mask

HETDEX_LON_RANGE = [158, 234]
HETDEX_LAT_RANGE = [43, 60]


def plot_hetdex_image(map, additional_mask=None, title=None, cmap='viridis', fwhm=0, norm=None):
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


def plot_correlation(binning, correlation, model_correlation=None, covariance_matrix=None, x_max=None, y_min=None,
                     x_scale='linear', y_scale='linear'):
    ell_arr = binning.get_effective_ells()
    to_plot = np.fabs(correlation)

    if covariance_matrix is not None:
        y_err = [covariance_matrix[i, i] for i in range(covariance_matrix.shape[0])]
        plt.errorbar(ell_arr, to_plot, yerr=y_err, fmt='ob', label='GG', markersize=2)
    else:
        plt.plot(ell_arr, to_plot, 'ob', label='GG', markersize=2)

    if model_correlation is not None:
        plt.plot(ell_arr, model_correlation, 'r', label='theory', markersize=2)

    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.xlim(xmax=x_max)
    plt.ylim(ymin=y_min)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$C_\\ell$', fontsize=16)
    plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
    plt.show()

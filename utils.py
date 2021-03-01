import logging
import math

import numpy as np
from astropy.table import Table
import healpy as hp
import pymaster as nmt

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# def get_chi_squared(data, theory, covariance):
#     diff = data - theory
#     cov_inv = np.linalg.inv(covariance)
#     return diff.dot(cov_inv).dot(diff)


def compute_master(field_a, field_b, binning):
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field_a, field_b, binning)
    cl_coupled = nmt.compute_coupled_cell(field_a, field_b)
    cl_decoupled = workspace.decouple_cell(cl_coupled)
    return cl_decoupled[0], workspace


def get_correlation_matrix(covariance_matrix):
    correlation_matrix = covariance_matrix.copy()
    for i in range(covariance_matrix.shape[0]):
        for j in range(covariance_matrix.shape[1]):
            correlation_matrix[i, j] = covariance_matrix[i, j] / math.sqrt(
                covariance_matrix[i, i] * covariance_matrix[j, j])
    return correlation_matrix


# TODO: noise_weight_map = None
def get_overdensity_map(counts_map, mask, noise_weight_map):
    # sky_mean = np.mean(counts_map.compressed())
    # overdensity_map = (counts_map.data - sky_mean) / sky_mean
    # overdensity_map = get_masked_map(overdensity_map, mask)
    sky_mean = (counts_map / noise_weight_map).compressed().mean()
    overdensity_map = counts_map / noise_weight_map / sky_mean - 1
    overdensity_map = get_masked_map(overdensity_map, mask)
    return overdensity_map


def get_shot_noise(map, mask):
    sky_frac = np.sum(mask) / np.shape(mask)[0]
    # TODO: not only non zero but weighted mean (?)
    n_obj = np.sum(map[np.nonzero(mask)])
    shot_noise = 4.0 * math.pi * sky_frac / n_obj
    return shot_noise


def add_mask(map, additional_mask):
    map = map.copy()
    map.mask = np.logical_or(map.mask, np.logical_not(additional_mask))
    return map


def get_masked_map(map, mask):
    map = hp.ma(map)
    map.mask = np.logical_not(mask)
    return map


def tansform_map_and_mask_to_nside(map, mask, nside):
    if nside:
        map = hp.pixelfunc.ud_grade(map, nside)
        mask = hp.pixelfunc.ud_grade(mask, nside)
    return map, mask


# TODO: index on pixel indices
def get_mean_map(l, b, v, nside):
    thetas, phis = np.radians(-b + 90.), np.radians(l)
    npix = hp.nside2npix(nside)  # 12 * nside ^ 2
    n_obj_map = np.zeros(npix, dtype=np.float)
    mean_map = np.zeros(npix, dtype=np.float)

    indices = hp.ang2pix(nside, thetas, phis, nest=False)
    for i, j in enumerate(indices):
        # Add objects weight and store a count
        mean_map[j] += v[i]
        n_obj_map[j] += 1

    mean_map /= n_obj_map
    lon, lat = hp.pixelfunc.pix2ang(nside, range(npix), nest=False, lonlat=True)

    return mean_map, lon, lat


def get_redshift_distribution(data, n_bins=50, z_col='Z_PHOTO_QSO'):
    n_arr, z_arr = np.histogram(data[z_col], bins=n_bins)
    z_arr = [(z_arr[i + 1] + z_arr[i]) / 2 for i in range(len(z_arr) - 1)]
    return z_arr, n_arr


def get_map(l, b, v=None, nside=128):
    # TODO
    # ipix = hp.ang2pix(nside, np.radians(90 - cat['DEC']), np.radians(cat['RA']))
    # map_n = np.bincount(ipix, minlength=npix)
    phis, thetas = np.radians(l), np.radians(-b + 90.)
    npix = hp.nside2npix(nside)  # 12 * nside ^ 2
    n_obj_map = np.zeros(npix, dtype=np.float)

    indices = hp.ang2pix(nside, thetas, phis, nest=False)
    for i, j in enumerate(indices):
        # Add objects weight or store a count
        v_i = v[i] if v is not None else 1
        n_obj_map[j] += v_i

    lon, lat = hp.pixelfunc.pix2ang(nside, range(npix), nest=False, lonlat=True)

    return n_obj_map, lon, lat


def read_fits_to_pandas(filepath, columns=None, n=None):
    table = Table.read(filepath, format='fits')

    # Get first n rows if limit specified
    if n:
        table = table[0:n]

    # Get proper columns into a pandas data frame
    if columns:
        table = table[columns]
    table = table.to_pandas()

    # Astropy table assumes strings are byte arrays
    for col in ['ID', 'ID_1', 'CLASS', 'CLASS_PHOTO', 'id1']:
        if col in table and hasattr(table.loc[0, col], 'decode'):
            table.loc[:, col] = table[col].apply(lambda x: x.decode('UTF-8').strip())

    # Change type to work with it as with a bit map
    if 'IMAFLAGS_ISO' in table:
        table.loc[:, 'IMAFLAGS_ISO'] = table['IMAFLAGS_ISO'].astype(int)

    return table


def get_pairs(values_arr, join_with=''):
    return [join_with.join((a, b)) for i, a in enumerate(values_arr) for b in values_arr[i:]]

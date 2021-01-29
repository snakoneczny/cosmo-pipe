import logging

import numpy as np
from astropy.table import Table
import healpy as hp

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_masked_map(map, mask):
    map = hp.ma(map)
    map.mask = np.logical_not(mask)
    return map


def add_mask(map, additional_mask):
    map = map.copy()
    map.mask = np.logical_or(map.mask, np.logical_not(additional_mask))
    return map


def tansform_map_and_mask_to_nside(map, mask, nside):
    if nside:
        map = hp.pixelfunc.ud_grade(map, nside)
        mask = hp.pixelfunc.ud_grade(mask, nside)
    return map, mask


# TODO: noise_weight_map = None
def get_overdensity_map(counts_map, mask, noise_weight_map):
    # sky_mean = np.mean(counts_map.compressed())
    # overdensity_map = (counts_map.data - sky_mean) / sky_mean
    # overdensity_map = get_masked_map(overdensity_map, mask)
    sky_mean = (counts_map / noise_weight_map).compressed().mean()
    overdensity_map = counts_map / noise_weight_map / sky_mean - 1
    overdensity_map = get_masked_map(overdensity_map, mask)
    return overdensity_map


def get_map(l, b, v=None, nside=128):
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

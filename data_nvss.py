import os

import healpy as hp
import numpy as np

from env_config import DATA_PATH
from utils import tansform_map_and_mask_to_nside


def get_nvss_redshift_distribution():
    z_arr = np.array([0.01, 1.0, 2.0, 3.0, 4.0])
    # n_arr = 1.29 + 32.37 * z_arr - 32.89 * z_arr * z_arr + 11.13 * z_arr * z_arr * z_arr - 1.25 * z_arr * z_arr * z_arr * z_arr
    n_arr = np.array([0.01, 0.5, 0.5 * 7 / 8, 0.5 * 5 / 8, 0.5 * 9 / 16])
    return z_arr, n_arr


def get_nvss_map(nside=None):
    nvss_map_filepath = os.path.join(DATA_PATH, 'NVSS', 'nvss_map_n512.fits')
    nvss_mask_filepath = os.path.join(DATA_PATH, 'NVSS', 'nvss_mask_n0512.fits')
    nvss_map, nvss_mask = hp.read_map(nvss_map_filepath), hp.read_map(nvss_mask_filepath)
    nvss_map, nvss_mask = tansform_map_and_mask_to_nside(nvss_map, nvss_mask, nside=nside)
    return nvss_map, nvss_mask

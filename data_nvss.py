import os

import healpy as hp
import numpy as np
from healpy import Rotator

from env_config import DATA_PATH
from utils import tansform_map_and_mask_to_nside, get_masked_map


def get_nvss_redshift_distribution():
    z_max = 2.25
    z_min = 0.01
    z_step = 0.01
    z_arr = np.arange(z_min, z_max, z_step)
    n_arr = 1.29 + 32.37 * z_arr - 32.98 * np.power(z_arr, 2) + 11.13 * np.power(z_arr, 3) - 1.25 * np.power(z_arr, 4)
    return z_arr, n_arr


def get_nvss_map(nside=None):
    map_filepath = os.path.join(DATA_PATH, 'NVSS', 'nvss_map_n512.fits')
    mask_filepath = os.path.join(DATA_PATH, 'NVSS', 'nvss_mask_n0512.fits')
    map, mask = hp.read_map(map_filepath), hp.read_map(mask_filepath)

    rotator = Rotator(coord=['G', 'C'])
    map = rotator.rotate_map_pixel(map)
    mask = rotator.rotate_map_pixel(mask)

    map, mask = tansform_map_and_mask_to_nside(map, mask, nside=nside)
    map = get_masked_map(map, mask)
    return map, mask

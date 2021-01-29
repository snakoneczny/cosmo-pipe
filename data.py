import os

import numpy as np
import healpy as hp

from env_config import DATA_PATH
from utils import tansform_map_and_mask_to_nside, get_map, read_fits_to_pandas


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


def get_kids_qso_map(qsos, nside):
    # TODO: include skycoord frame and transform to galactic
    map, _, _ = get_map(qsos['RAJ2000'], qsos['DECJ2000'], nside=nside)

    # TODO: normal masking
    mask = map.copy()
    mask[map.nonzero()] = 1

    return map, mask


def get_kids_qsos():
    qso_candidates_filepath = os.path.join(DATA_PATH, 'KiDS/DR4/catalogs/published/KiDS_DR4_QSO_candidates.fits')
    qso_candidates = read_fits_to_pandas(qso_candidates_filepath)

    qsos = qso_candidates.loc[
        (qso_candidates['MAG_GAAP_r'] < 23.5) &
        (qso_candidates['QSO_PHOTO'] > 0.98)
        # (qso_candidates['DECJ2000'] > -10)
        ]

    return qsos

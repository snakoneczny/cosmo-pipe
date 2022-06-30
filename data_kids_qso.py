import os

import numpy as np
import healpy as hp

from env_config import DATA_PATH
from utils import read_fits_to_pandas, get_map


def get_kids_qso_map(qsos, nside):
    map = get_map(qsos['RAJ2000'].values, qsos['DECJ2000'].values, nside=nside)

    mask = hp.read_map(os.path.join(DATA_PATH, 'KiDS/DR4/masks/mask_nside=256.fits'))
    mask = mask.astype('float64')

    # mask = get_map(qsos['RAJ2000'].values, qsos['DECJ2000'].values, nside=256)
    # mask = hp.ud_grade(mask, nside)
    # mask[mask.nonzero()] = 1

    return map, mask


def get_kids_qsos(r_max=22, qso_min_proba=0.9):
    qso_candidates_filepath = os.path.join(DATA_PATH, 'KiDS/DR4/catalogs/published/KiDS_DR4_QSO_candidates.fits')
    qso_candidates = read_fits_to_pandas(qso_candidates_filepath)

    # proba_limits = [(21, 0.9), (22, 0.9), (23, 0.998), (23.5, 0.998)]
    # proba_limits = {21: 0.9, 22: 0.9, 23: 0.998, 23.5: 0.998}
    # proba_limits[r_max] = qso_min_proba
    # proba_limits = [(k, v) for k, v in proba_limits.items() if k <= r_max]
    # print(proba_limits)
    # idx = [(qso_candidates['MAG_GAAP_r'] < r_max) & (qso_candidates['QSO_PHOTO'] > proba_min) for r_max, proba_min in proba_limits]
    # idx = np.logical_or.reduce(idx)
    # qsos = qso_candidates.loc[idx]

    qsos = qso_candidates.loc[
        (qso_candidates['MAG_GAAP_r'] < r_max) &
        (qso_candidates['QSO_PHOTO'] > qso_min_proba)
        # (qso_candidates['DECJ2000'] < -10)
    ]

    return qsos

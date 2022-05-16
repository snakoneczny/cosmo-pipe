import os

import healpy as hp

from env_config import DATA_PATH
from utils import read_fits_to_pandas, get_map


def get_kids_qso_map(qsos, nside):
    map = get_map(qsos['RAJ2000'].values, qsos['DECJ2000'].values, nside=nside)

    mask = get_map(qsos['RAJ2000'].values, qsos['DECJ2000'].values, nside=256)
    mask = hp.ud_grade(mask, nside)
    mask[mask.nonzero()] = 1

    return map, mask


def get_kids_qsos(r_max=22, qso_min_proba=0.9):
    qso_candidates_filepath = os.path.join(DATA_PATH, 'KiDS/DR4/catalogs/published/KiDS_DR4_QSO_candidates.fits')
    qso_candidates = read_fits_to_pandas(qso_candidates_filepath)

    qsos = qso_candidates.loc[
        (qso_candidates['MAG_GAAP_r'] < r_max) &
        (qso_candidates['QSO_PHOTO'] > qso_min_proba)
        # (qso_candidates['DECJ2000'] < -10)
    ]

    return qsos

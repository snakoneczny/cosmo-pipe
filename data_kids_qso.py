import os

from env_config import DATA_PATH
from utils import get_map, read_fits_to_pandas


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
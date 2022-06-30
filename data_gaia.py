import os

import healpy as hp

from env_config import DATA_PATH
from utils import get_map, read_fits_to_pandas


def get_gaia_stars_map(stars, nside):
    map = get_map(stars['pmra'].values, stars['pmdec'].values, nside=nside)

    mask = get_map(stars['pmra'].values, stars['pmdec'].values, nside=256)
    mask = hp.ud_grade(mask, nside)
    mask[mask.nonzero()] = 1

    return map, mask


def get_gaia_stars():
    gaia_filepath = os.path.join(DATA_PATH, 'GAIA/GAIA.DR2.fits')
    gaia_stars = read_fits_to_pandas(gaia_filepath)

    # qsos = qso_candidates.loc[
    #     (qso_candidates['MAG_GAAP_r'] < r_max) &
    #     (qso_candidates['QSO_PHOTO'] > qso_min_proba)
    #     # (qso_candidates['DECJ2000'] < -10)
    # ]

    return gaia_stars

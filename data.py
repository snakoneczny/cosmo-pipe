import os

import numpy as np
from astropy.table import Table
import healpy as hp
from scipy.integrate import simps

from env_config import DATA_PATH


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


cmb_columns_idx = struct(
    I_STOKES=0,
    Q_STOKES=1,
    U_STOKES=2,
    TMASK=3,
    PMASK=4,
    I_STOKES_INP=5,
    Q_STOKES_INP=6,
    U_STOKES_INP=7,
    TMASK_INP=8,
    PMASKINP=9,
)


def get_nvss_redshift_distribution(map, mask, nside):
    # Redshift distribution according to [65] but not the one from fig. 1
    # z_min = 0.01
    # z_max = 3
    # z_step = 0.1
    # z_arr = np.arange(z_min, z_max, z_step)
    z_step = 1
    z_arr = np.array([0.01, 1.0, 2.0, 3.0, 4.0])

    # n_arr = 1.29 + 32.37 * z_arr - 32.89 * z_arr * z_arr + 11.13 * z_arr * z_arr * z_arr - 1.25 * z_arr * z_arr * z_arr * z_arr
    n_arr = np.array([0.01, 0.5, 0.5 * 7 / 8, 0.5 * 5 / 8, 0.5 * 9 / 16])

    n_arr = make_redshift_distribution_physical(n_arr, z_step, map, mask, nside)

    return z_arr, n_arr


def get_data_redshift_distribution(data, map, mask, nside, n_bins=50, z_col='Z_PHOTO_QSO'):
    n_arr, z_arr = np.histogram(data[z_col], bins=n_bins)
    z_arr = [(z_arr[i + 1] + z_arr[i]) / 2 for i in range(len(z_arr) - 1)]
    z_step = z_arr[1] - z_arr[0]

    n_arr = make_redshift_distribution_physical(n_arr, z_step, map, mask, nside)

    return z_arr, n_arr


def make_redshift_distribution_physical(n_arr, z_step, map, mask, nside):
    # Normalize to unit integral
    area = simps(n_arr, dx=z_step)
    n_arr = n_arr / area
    # Get dNdz in arcmin squared
    n_objects = map[np.nonzero(mask)].sum()
    pix_size = hp.pixelfunc.nside2resol(nside, arcmin=True)
    pix_area = pix_size ** 2
    sky_area = mask.sum() * pix_area
    objects_density = n_objects / sky_area
    n_arr = n_arr * objects_density
    return n_arr


def get_cmb_map(nside=None):
    cmb_filename = 'COM_CMB_IQU-smica_2048_R3.00_full.fits'
    cmb_map = hp.read_map(os.path.join(DATA_PATH, 'Planck2018', cmb_filename), field=cmb_columns_idx.I_STOKES)
    cmb_mask = hp.read_map(os.path.join(DATA_PATH, 'Planck2018', cmb_filename), field=cmb_columns_idx.TMASK)
    cmb_map, cmb_mask = tansform_map_and_mask_to_nside(cmb_map, cmb_mask, nside=nside)
    return cmb_map, cmb_mask


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


def tansform_map_and_mask_to_nside(map, mask, nside):
    if nside:
        map = hp.pixelfunc.ud_grade(map, nside)
        mask = hp.pixelfunc.ud_grade(mask, nside)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
    return map, mask


def get_map(l, b, v=None, nside=128):
    # Set the number of sources and the coordinates for the input
    npix = hp.nside2npix(nside)  # 12 * nside ^ 2

    # Coordinates and the density field f
    thetas, phis = np.radians(-b + 90.), np.radians(l)

    # Initate the map and fill it with the values
    hpxmap = np.zeros(npix, dtype=np.float)

    # Go from HEALPix coordinates to indices
    indices = hp.ang2pix(nside, thetas, phis, nest=False)
    for i, j in enumerate(indices):
        # Add objects weight or store a count
        v_i = v[i] if v is not None else 1
        hpxmap[j] += v_i

    lon, lat = hp.pixelfunc.pix2ang(nside, range(npix), nest=False, lonlat=True)

    return hpxmap, lon, lat


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

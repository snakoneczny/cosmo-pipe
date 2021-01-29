import os

import healpy as hp
import numpy as np
from healpy import Rotator

from env_config import DATA_PATH
from utils import struct, get_masked_map, tansform_map_and_mask_to_nside, read_fits_to_pandas

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


def get_cmb_map(nside=None):
    filename = 'COM_CMB_IQU-smica_2048_R3.00_full.fits'
    map = hp.read_map(os.path.join(DATA_PATH, 'Planck2018', filename), field=cmb_columns_idx.I_STOKES)
    mask = hp.read_map(os.path.join(DATA_PATH, 'Planck2018', filename), field=cmb_columns_idx.TMASK)
    map = get_masked_map(map, mask)
    map, mask = tansform_map_and_mask_to_nside(map, mask, nside=nside)
    return map, mask


def get_cmb_lensing_map(nside=None, fwhm=0):
    folder_path = os.path.join(DATA_PATH, 'Planck2018/COM_Lensing_4096_R3.00')
    map_path = os.path.join(folder_path, 'MV', 'dat_klm.fits')
    mask_path = os.path.join(folder_path, 'mask.fits')

    mask = hp.read_map(mask_path)

    klm = read_fits_to_pandas(map_path)

    # l_max = m_max = 4096  # 2048
    # n_max = int(m_max * (2 * l_max + 1 - m_max) / 2 + l_max + 1)
    # klm_max = klm.head(n_max)

    # l_min = m_min = 8
    # n_min = int(m_min * (2 * l_min + 1 - m_min) / 2 + l_min + 1)
    # klm_min = klm.head(n_min)

    # klm_max = np.array([complex(real, imag) for (real, imag) in zip(klm_max['real'], klm_max['imag'])])
    # map_max = hp.sphtfunc.alm2map(klm_max, nside=nside, lmax=None, mmax=None, pixwin=False, fwhm=fwhm, sigma=None,
    #                               pol=True, inplace=False, verbose=True)

    # klm_min = np.array([complex(real, imag) for (real, imag) in zip(klm_min['real'], klm_min['imag'])])
    # map_min = hp.sphtfunc.alm2map(klm_min, nside=nside, lmax=None, mmax=None, pixwin=False, fwhm=fwhm, sigma=None,
    #                                  pol=True, inplace=False, verbose=True)

    # map = map_max - map_min

    klm = np.array([complex(real, imag) for (real, imag) in zip(klm['real'], klm['imag'])])
    map = hp.sphtfunc.alm2map(klm, nside=nside, lmax=None, mmax=None, pixwin=False, fwhm=fwhm, sigma=None,
                              pol=True, inplace=False, verbose=True)

    # map = hp.sphtfunc.smoothing(map, fwhm=0.0174533, sigma=None, beam_window=None, pol=True, iter=3, lmax=None,
    #                             mmax=None, use_weights=False, use_pixel_weights=False, datapath=None, verbose=True)

    mask = hp.ud_grade(mask, nside_out=nside)

    rotator = Rotator(coord=['G', 'C'])
    map = rotator.rotate_map_pixel(map)
    mask = rotator.rotate_map_pixel(mask)

    map = get_masked_map(map, mask)

    return map, mask

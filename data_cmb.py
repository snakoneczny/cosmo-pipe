import os

import healpy as hp
import numpy as np
from healpy import Rotator
import pymaster as nmt
import pandas as pd

from env_config import DATA_PATH
from utils import struct, get_masked_map, tansform_map_and_mask_to_nside

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


def get_cmb_temperature_map(nside=None):
    filename = 'COM_CMB_IQU-smica_2048_R3.00_full.fits'

    # Read
    map = hp.read_map(os.path.join(DATA_PATH, 'Planck2018', filename), field=cmb_columns_idx.I_STOKES)
    mask = hp.read_map(os.path.join(DATA_PATH, 'Planck2018', filename), field=cmb_columns_idx.TMASK)

    # Rotate
    rotator = Rotator(coord=['G', 'C'])
    map = rotator.rotate_map_pixel(map)
    mask = rotator.rotate_map_pixel(mask)

    # Adjust
    map, mask = tansform_map_and_mask_to_nside(map, mask, nside=nside)
    mask[mask < 0.1] = 0  # Visualization purpose

    map = get_masked_map(map, mask)
    return map, mask


def get_cmb_lensing_map(nside=None):
    folder_path = os.path.join(DATA_PATH, 'Planck2018/COM_Lensing_2048_R2.00')
    map_path = os.path.join(folder_path, 'dat_klm.fits')
    mask_path = os.path.join(folder_path, 'mask.fits')

    # Read
    klm = hp.read_alm(map_path)
    map = hp.alm2map(klm, nside)
    mask = hp.read_map(mask_path)

    # Rotate
    rotator = Rotator(coord=['G', 'C'])
    map = rotator.rotate_map_pixel(map)
    mask = rotator.rotate_map_pixel(mask)

    # Adjust
    mask = nmt.mask_apodization(mask, 0.2, apotype='C1')
    mask = hp.ud_grade(mask, nside_out=nside)
    mask[mask < 0.1] = 0  # Visualization purpose

    map = get_masked_map(map, mask)
    return map, mask


def get_cmb_lensing_noise(nside):
    l_arr = np.arange(3 * nside)

    path = os.path.join(DATA_PATH, 'Planck2018/COM_Lensing_2048_R2.00/nlkk.dat')
    data = np.loadtxt(path, unpack=False)
    data = pd.DataFrame(data, columns=['l', 'nl', 'cl+nl'])

    l_min = int(data['l'][0])
    l_max = min(l_arr[-1], data['l'].values[-1])
    noise_l_arr = np.arange(l_min, l_max + 1)

    noise = np.zeros(len(l_arr))
    noise[noise_l_arr] += data['nl'].values[:l_max - l_min + 1]

    return noise

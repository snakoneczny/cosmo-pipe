import os
import math

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


def get_cmb_temperature_power_spectra(nside):
    assert 3 * nside <= 2508

    path = os.path.join(DATA_PATH, 'Planck2018/COM_PowerSpect_CMB-TT-full_R3.01.txt')
    data = np.loadtxt(path, unpack=False)
    data = pd.DataFrame(data, columns=['l', 'Dl', '-dDl', '+dDl'])

    # TODO: 1e-12 ??
    # cal_planck = 0.1000442E+01
    data['Cl'] = data['Dl'] / data['l'] / (data['l'] + 1) * 2 * math.pi * 1e-12  # / (cal_planck ** 2)

    l_arr = np.arange(3 * nside)
    l_min = int(data['l'][0])
    l_max = int(min(l_arr[-1], data['l'].values[-1]))
    cl_l_arr = np.arange(l_min, l_max + 1)
    cl = np.zeros(len(l_arr))
    cl[cl_l_arr] += data['Cl'].values[:l_max - l_min + 1]

    return cl


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

    map = get_masked_map(map, mask)
    mask = get_masked_map(mask, mask)
    return map, mask


def get_cmb_lensing_noise(nside):
    assert nside <= 2048

    path = os.path.join(DATA_PATH, 'Planck2018/COM_Lensing_4096_R3.00/MV/nlkk.dat')
    data = np.loadtxt(path, unpack=False)
    data = pd.DataFrame(data, columns=['l', 'nl', 'cl+nl'])

    l_arr = np.arange(3 * nside)
    l_min = int(data['l'][0])
    l_max = int(min(l_arr[-1], data['l'].values[-1]))
    noise_l_arr = np.arange(l_min, l_max + 1)
    noise = np.zeros(len(l_arr))
    noise[noise_l_arr] += data['nl'].values[:l_max - l_min + 1]

    return noise


def get_cmb_lensing_map(nside=None):
    folder_path = os.path.join(DATA_PATH, 'Planck2018/COM_Lensing_4096_R3.00')
    map_path = os.path.join(folder_path, 'MV/dat_klm.fits')
    mask_path = os.path.join(folder_path, 'mask.fits')

    # Read
    klm = hp.read_alm(map_path)

    fl = np.zeros(hp.Alm.getlmax(klm.shape[0]))
    fl[:3*nside] = 1
    klm = hp.sphtfunc.almxfl(klm, fl)

    map = hp.alm2map(klm, nside=nside)
    mask = hp.read_map(mask_path)

    # Rotate
    rotator = Rotator(coord=['G', 'C'])
    map = rotator.rotate_map_pixel(map)
    mask = rotator.rotate_map_pixel(mask)

    # Adjust
    mask = hp.ud_grade(mask, nside_out=nside)

    map = get_masked_map(map, mask)
    mask = get_masked_map(mask, mask)
    return map, mask

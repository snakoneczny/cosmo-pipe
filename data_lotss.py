import math
import os

import healpy as hp
import numpy as np
from scipy.special import erf
from scipy.integrate import simps
from tqdm import tqdm

from env_config import DATA_PATH
from utils import get_map, get_masked_map, get_aggregated_map, read_fits_to_pandas


def get_biggest_optical_region(data):
    return data.loc[((data['RA'] < 33) | (data['RA'] > 360 - 29)) & (18 < data['DEC']) & (data['DEC'] < 35)]


def get_lotss_redshift_distribution(z_tail=None, z_sfg=None, a=None, r=None, n=None, flux_cut=None, model='power_law',
                                    z_max=6):
    if model == 'deep_fields':
        deepfields_file = 'LoTSS/DR2/pz_deepfields/Pz_booterrors_wsum_deepfields_{}mJy.fits'.format(
            ''.join(str(flux_cut).split('.')))
        pz_deepfields = read_fits_to_pandas(os.path.join(DATA_PATH, deepfields_file))
        z_arr = pz_deepfields['zbins']
        n_arr = pz_deepfields['pz_boot_mean']

    else:
        z_step = 0.01
        z_min = 0
        z_max = z_max + z_step
        z_arr = np.arange(z_min, z_max, z_step)

        if model == 'power_law':
            # n_arr = (z_arr ** 2) / (1 + z_arr) * (np.exp((-z_arr / z_sfg)) + r * np.exp(-z_arr / z_agn))
            n_arr = (z_arr ** 2) / (1 + z_arr) * (np.exp((-z_arr / z_sfg)) + r ** 2 / (1 + z_arr) ** a)
            if n:
                n_arr *= n
        elif model == 'z_tail':
            z_0 = 0.1
            gamma = 3.5
            n_arr = ((z_arr / z_0) ** 2) / (1 + (z_arr / z_0) ** 2) / (1 + (z_arr / z_tail) ** gamma)
        else:
            raise Exception('Not known redshift distribution model: {}'.format(model))

        area = simps(n_arr, dx=z_arr[1] - z_arr[0])
        n_arr /= area

    return z_arr, n_arr


def read_lotss_noise_weight_map(nside, data_release, flux_min_cut, signal_to_noise):
    file_path = os.path.join(DATA_PATH,
                             'LoTSS/DR{}/weight_maps/weight_map__pointing-mean_minflux-{}_snr-{}.fits'.format(
                                 data_release, flux_min_cut, signal_to_noise))
    weight_map = hp.read_map(file_path)
    weight_map = hp.ud_grade(weight_map, nside)
    return weight_map


def get_lotss_noise_weight_map(noise_map, flux_cut=2, signal_to_noise=0):
    # Read flux distribution from SKADS' S3-SEX simulation
    file_path = os.path.join(DATA_PATH, 'SKADS/skads_flux_counts.result')
    log_flux, counts = np.loadtxt(file_path, unpack=True, delimiter=',', skiprows=1)
    log_flux += 3  # Use mJy instead of Jy
    # Assuming equal spacing
    log_flux = log_flux[counts >= 0]
    counts = counts[counts >= 0]
    probabilities = counts / np.sum(counts)

    # Transfer 1400 to 144
    alpha = -0.7
    log_flux = log_flux + alpha * np.log10(144. / 1400.)

    # Compute probability map
    proba_map = np.zeros(len(noise_map))
    for i, noise in tqdm(enumerate(noise_map)):
        if noise > 0:
            threshold = max(signal_to_noise * noise, flux_cut)
            x = (threshold - 10. ** log_flux) / (np.sqrt(2.) * noise)
            comp = 0.5 * (1 - erf(x))
            proba_map[i] = np.sum(probabilities * comp)

    return proba_map


# def get_lotss_noise_weight_map(lotss_noise_map, lotss_mask, flux_min_cut, nside_out):
#     n_bins = 1000
#     flux_max = 2000
#     skads = get_skads_sim_data()
#     fluxes = skads['S_144'].loc[skads['S_144'] < flux_max]
#     flux_arr, flux_proba_arr, d_flux = get_normalized_dist(fluxes, n_bins=n_bins)
#
#     npix = hp.nside2npix(nside=256)  # 12 * nside ^ 2
#     noise_weight_map = np.zeros(npix, dtype=np.float)
#
#     # For each pixel
#     a = d_flux * 0.5
#     for i in tqdm(range(len(lotss_noise_map))):
#         if not lotss_noise_map.mask[i]:
#             # Integrate d_flux * flux_proba * 0.5 * erfc((flux_true - flux_thr) / (sqrt(2) * sigma_n))
#             b = (math.sqrt(2) * lotss_noise_map[i])
#             for j in range(len(flux_proba_arr)):
#                 flux_thr = max(5 * lotss_noise_map[i], flux_min_cut)
#                 noise_weight_map[i] += a * flux_proba_arr[j] * erfc((flux_arr[j] - flux_thr) / b)
#
#     noise_weight_map = hp.ud_grade(noise_weight_map, nside_out=nside_out)
#     noise_weight_map /= np.max(noise_weight_map)
#     noise_weight_map = get_masked_map(noise_weight_map, lotss_mask)
#
#     return noise_weight_map


def get_skads_sim_data():
    skads = read_fits_to_pandas(os.path.join(DATA_PATH, 'SKADS/100sqdeg_5uJy_s1400_components_fixed.fits'))
    # Conversion from log(I) to I
    skads['S_151'] = skads['i_151'].apply(math.exp)
    # Conversion to mJy
    skads['S_151'] *= 10 ** 3
    # Extrapolation to 144MHz
    skads['S_144'] = skads['S_151'].apply(flux_151_to_144)
    return skads


def flux_151_to_144(s_151):
    # assumption: I_nu = I_1400 * (nu / 1400) ^ alpha
    spectral_index = -0.7
    s_144 = s_151 * (144 / 151) ** spectral_index
    return s_144


def get_lotss_map(lotss_data, data_release, mask_filename=None, nside=2048, cut_pixels=False, masked=True):
    counts_map = get_map(lotss_data['RA'].values, lotss_data['DEC'].values, nside=nside)
    if masked:
        if data_release == 1:
            mask = get_lotss_dr1_mask(nside)
        elif data_release == 2:
            mask = get_lotss_dr2_mask(nside, filename=mask_filename)
        else:
            raise Exception('Wrong LoTSS data release number')
    else:
        mask = None

    # Get noise in larger bins
    noise_map = get_aggregated_map(lotss_data['RA'].values, lotss_data['DEC'].values,
                                   lotss_data['Isl_rms'].values, nside=256, aggregation='mean')

    # Cut artificially high count pixels
    if cut_pixels:
        indices_max = np.argsort(counts_map)[::-1]
        pix_size = hp.pixelfunc.nside2resol(512)
        for ipix in indices_max:
            if counts_map[ipix] > 11:
                vec = hp.pixelfunc.pix2vec(nside, ipix)
                radius = pix_size * 2
                indices = hp.query_disc(nside, vec, radius)
                mask[indices] = 0
            else:
                break

    if masked:
        noise_map = get_masked_map(noise_map, hp.ud_grade(mask, nside_out=256))
        counts_map = get_masked_map(counts_map, mask)

    return counts_map, mask, noise_map


def get_lotss_dr2_mask(nside, filename=None):
    filename = 'Mask_default' if filename is None else filename
    masks_in_files = [
        'mask_coverage', 'mask_default', 'mask_noise_75percent', 'mask_noise_99_percent', 'mask_noise_median']
    if filename in masks_in_files:
        mask = hp.read_map(os.path.join(DATA_PATH, 'LoTSS/DR2/masks/{}.fits'.format(filename)))
        mask = hp.ud_grade(mask, nside)
    elif filename == 'mask_optical':
        mask = get_dr2_optical_region(nside)
    else:
        raise Exception('Mask doesn\'t exist: {}'.format(filename))

    # TODO: delete
    # mask = get_lotss_dr1_mask(nside)

    # for i in range(len(mask)):
    #     lon, lat = hp.pixelfunc.pix2ang(nside, i, lonlat=True)
    #     if 60 < lon < 300:
    #         mask[i] = 0

    # TODO: delete
    # import healpy as hp
    # tmp = hp.pixelfunc.ud_grade(hp.pixelfunc.ud_grade(self.base_maps['g'], nside_out=128), nside_out=self.nside)
    # self.masks['g'][tmp == 0] = 0

    # TODO: delete or permanently add to code / maps
    # TODO: refactor to some query instead of the for loop
    # for i in range(len(mask)):
    #     lon, lat = hp.pixelfunc.pix2ang(nside=nside, ipix=i, nest=False, lonlat=True)
    #     if lon < 100 or lon > 300:  # Big patch
    #     if 100 < lon < 300:  # Small patch
    #         mask[i] = 0

    #     if lat < 50:  # Big north part
    #     if lat > 50:  # Big south part
    #         mask[i] = 0

    #     if lon > 160:  # Big right part
    #     if lon < 160 or lon > 220:  # Big central part
    #     if lon < 220:  # Big left part
    #         mask[i] = 0

    return mask


def get_dr2_optical_region(nside):
    mask = np.zeros(hp.pixelfunc.nside2npix(nside))
    for i in range(len(mask)):
        lon, lat = hp.pixelfunc.pix2ang(nside, i, lonlat=True)
        inside_region = (
            # Big region left
            (33 < lon < 39 and 25 < lat < 35)
            # Big region right
            or ((lon < 33 or lon > 360 - 29) and 18 < lat < 35)
            # Top stripe
            or (123.5 < lon < 360 - 121.5 and 59.5 < lat < 65)
            # Bottom stripe left
            or (122 < lon < 360 - 84 and 40.2 < lat < 45.5)
            # Bottom stripe right - left
            or (130 < lon < 134 and 28.5 < lat < 40)
            # Bottom stripe right - right
            or (111 < lon < 130 and 26.5 < lat < 40)
        )
        if inside_region:
            mask[i] = 1
    return mask


def get_lotss_dr1_mask(nside):
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=np.float)
    pointings = np.loadtxt(os.path.join(DATA_PATH, 'LoTSS/DR1/pointings.txt'))
    pointings_to_skip = [[164.633, 54.685], [211.012, 49.912], [221.510, 47.461], [225.340, 47.483], [227.685, 52.515]]
    radius = math.radians(1.7)
    for ra, dec in pointings:
        if [ra, dec] in pointings_to_skip:
            continue
        theta, phi = np.radians(-dec + 90.), np.radians(ra)
        vec = hp.pixelfunc.ang2vec(theta, phi, lonlat=False)
        indices = hp.query_disc(nside, vec, radius)
        mask[indices] = 1

    return mask


def get_lotss_data(data_release, flux_min_cut=2, signal_to_noise=None, optical=True, columns=None):
    if data_release == 1:
        filename = 'LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2b_restframe.fits' if optical else \
            'LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits'
    elif data_release == 2:
        filename = 'combined-release-v0.1.fits' if optical else 'LoTSS_DR2_v100.srl.fits'
    data_path = os.path.join(DATA_PATH, 'LoTSS/DR{}'.format(data_release), filename)

    data = read_fits_to_pandas(data_path)
    print('Original LoTSS DR{} datashape: {}'.format(data_release, data.shape))

    # Flux cut
    if flux_min_cut:
        data = data.loc[data['Total_flux'] > flux_min_cut]
        print('Total flux of S > {} mJy: {}'.format(flux_min_cut, data.shape))

    # Signal to noise ratio cut
    if signal_to_noise:
        data = data.loc[data['Total_flux'] / data['E_Total_flux'] > signal_to_noise]
        print('Signal to noise > {}: {}'.format(signal_to_noise, data.shape))

    # TODO: delete or permanently add to code
    # Patches
    # data = data.loc[(data['RA'] > 100) & (data['RA'] < 300)]
    # print('Big patch: {}'.format(data.shape))
    # data = data.loc[(data['RA'] < 100) | (data['RA'] > 300)]
    # print('Small patch: {}'.format(data.shape))

    if columns is not None:
        data = data[columns]

    return data

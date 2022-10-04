import math
import os
from collections import defaultdict

import healpy as hp
import numpy as np
from scipy.special import erf
from scipy.integrate import simps
from tqdm import tqdm
import pandas as pd

from env_config import DATA_PATH
from utils import get_map, read_fits_to_pandas

# (8, 4) 32 in total, (4, 2) 16 in total
LOTSS_JACKKNIFE_REGIONS = [
    {'lon': (113, 260, 11), 'lat': (25, 68, 4)},
    {'lon': (37, -25, 5), 'lat': (19, 40, 2)},
]


# TODO: refactor, shorten, split into more functions (?)
def get_redshift_distributions(data_optical, data_skads):
    # Tomographer
    z_sfg_t = {2: (3.77263046e-02, 7.89948246e-03), 1: None, 0.5: None}
    a_t = {2: (4.44865395e+00, -2.66729855e-01), 1: None, 0.5: None}
    r_t = {2: (1.81466381e-01, 5.02055812e-02), 1: None, 0.5: None}

    # Fit to deep fields
    z_sfg_d = {2: 0.13, 1: 0.14, 0.5: 0.15}
    a_d = {2: 4.66, 1: 4.60, 0.5: 4.56}
    r_d = {2: 0.94, 1: 0.90, 0.5: 0.93}

    # Fit to DR2 correlations
    z_sfg_f = {2: 0.15, 1.5: 0.016, 1: 0.22, 0.5: 0.15}
    a_f = {2: 4.8, 1.5: 4.32, 1: 5.2, 0.5: 4.56}
    r_f = {2: 0.91, 1.5: 0.05, 1: 0.80, 0.5: 0.93}

    z_tail = {2: 1.3}
    z_tail_limit = {2: (-0.4, 0.27)}

    redshift_distributions = defaultdict(dict)
    for flux_cut in [2, 1.5, 1, 0.5]:

        # TRECS
        filepath = os.path.join(DATA_PATH, 'TRECS/pz/trecs{}wide_z_nz_{}mjy.dat')
        trecs_all_filepath = filepath.format('ALL', flux_cut)
        trecs_sfg_filepath = filepath.format('SFG', flux_cut)
        trecs_agn_filepath = filepath.format('AGN', flux_cut)

        trecs_all = pd.read_table(trecs_all_filepath, sep=' ', names=['z', 'nz'])
        trecs_sfg = pd.read_table(trecs_sfg_filepath, sep=' ', names=['z', 'nz'])
        trecs_agn = pd.read_table(trecs_agn_filepath, sep=' ', names=['z', 'nz'])

        redshift_distributions['TRECS'][flux_cut] = {'z': trecs_all['z'], 'pz': trecs_all['nz'],
                                                     'pz_sfg': trecs_sfg['nz'], 'pz_agn': trecs_agn['nz']}

        # Deep fields
        pz_deepfields = read_fits_to_pandas(os.path.join(DATA_PATH,
                                                         'LoTSS/DR2/pz_deepfields/Pz_booterrors_wsum_deepfields_{}mJy.fits'.format(
                                                             ''.join(str(flux_cut).split('.')))))

        key = 'deep fields'
        redshift_distributions[key][flux_cut] = {'z': pz_deepfields['zbins'], 'pz': pz_deepfields['pz']}

        key = 'deep fields, boot'
        redshift_distributions[key][flux_cut] = {'z': pz_deepfields['zbins'], 'pz': pz_deepfields['pz_boot_mean'],
                                                 'pz_min': pz_deepfields['pz_boot_mean'] - pz_deepfields[
                                                     'error_boot'] / 2,
                                                 'pz_max': pz_deepfields['pz_boot_mean'] + pz_deepfields[
                                                     'error_boot'] / 2}

        # Power laws
        power_laws = [
            ('power law AGN, deep fields', [z_sfg_d, a_d, r_d]),
            ('power law AGN, DR2', [z_sfg_f, a_f, r_f]),
            ('tomographer fit', [z_sfg_t, a_t, r_t]),
        ]
        for key, params in power_laws:
            if params[0][flux_cut]:
                z_sfg, a, r = params[0][flux_cut], params[1][flux_cut], params[2][flux_cut]
                n_arr_min, n_arr_max = None, None
                if isinstance(z_sfg, tuple):
                    z_sfg, z_sfg_err = z_sfg[0], z_sfg[1]
                    a, a_err = a[0], a[1]
                    r, r_err = r[0], r[1]

                    z_sfg_min = z_sfg - z_sfg_err / 2
                    a_min = a - a_err / 2
                    r_min = r - r_err / 2
                    z_sfg_max = z_sfg + z_sfg_err / 2
                    a_max = a + a_err / 2
                    r_max = r + r_err / 2

                    _, n_arr_min = get_lotss_redshift_distribution(z_sfg=z_sfg_min, a=a_min, r=r_min, model='power_law',
                                                                   z_max=6, normalize=False)
                    _, n_arr_max = get_lotss_redshift_distribution(z_sfg=z_sfg_max, a=a_max, r=r_max, model='power_law',
                                                                   z_max=6, normalize=False)

                z_arr, n_arr = get_lotss_redshift_distribution(z_sfg=z_sfg, a=a, r=r, model='power_law', z_max=6,
                                                               normalize=False)

                redshift_distributions[key][flux_cut] = {'z': z_arr, 'pz': n_arr, 'pz_min': n_arr_min,
                                                         'pz_max': n_arr_max}

        # DR1 z tail
        key = 'z tail, DR1'
        if flux_cut in z_tail:
            z_tail_min = z_tail[flux_cut] + z_tail_limit[flux_cut][0]
            z_tail_max = z_tail[flux_cut] + z_tail_limit[flux_cut][1]

            z_arr, n_arr = get_lotss_redshift_distribution(z_tail=z_tail[flux_cut], model='z_tail', z_max=6,
                                                           normalize=True)
            _, n_arr_min = get_lotss_redshift_distribution(z_tail=z_tail_min, model='z_tail', z_max=6, normalize=True)
            _, n_arr_max = get_lotss_redshift_distribution(z_tail=z_tail_max, model='z_tail', z_max=6, normalize=True)

            # redshift_distributions[key][flux_cut] = {'z': z_arr, 'pz': n_arr, 'pz_min': n_arr_min, 'pz_max': n_arr_max}
            redshift_distributions[key][flux_cut] = {'z': z_arr, 'pz': n_arr}

        # Photo-z
        key = 'photo-z, DR2'
        data_cut = data_optical.loc[data_optical['Total_flux'] > flux_cut]

        n_arr, z_arr = np.histogram(data_cut['z_best'][~np.isnan(data_cut['z_best'])], bins=100, density=True)
        z_arr = [(z_arr[i] + z_arr[i + 1]) / 2 for i in range(len(z_arr) - 1)]
        redshift_distributions[key][flux_cut] = {'z': z_arr, 'pz': n_arr}

        # SKADS
        key = 'SKADS'
        data_cut = data_skads.loc[data_skads['S_144'] > flux_cut]

        n_arr, z_arr = np.histogram(data_cut['redshift'][~np.isnan(data_cut['redshift'])], bins=100, density=True)
        z_arr = [(z_arr[i] + z_arr[i + 1]) / 2 for i in range(len(z_arr) - 1)]
        redshift_distributions[key][flux_cut] = {'z': z_arr, 'pz': n_arr}

    return redshift_distributions


def get_lotss_redshift_distribution(config=None, model='power_law', z_sfg=None, a=None, r=None, a_2=None, r_2=None,
                                    offset=None, n=1, z_tail=None, flux_cut=None, A_z_tail=None, z_arr=None, z_max=6,
                                    normalize=False):
    if config:
        model = config.dn_dz_model
        z_sfg = getattr(config, 'z_sfg', None)
        a = getattr(config, 'a', None)
        r = getattr(config, 'r', None)
        offset = getattr(config, 'offset', None)
        a_2 = getattr(config, 'a_2', None)
        r_2 = getattr(config, 'r_2', None)
        n = getattr(config, 'n', None)
        z_tail = getattr(config, 'z_tail', None)
        flux_cut = getattr(config, 'flux_min_cut', None)
        A_z_tail = getattr(config, 'A_z_tail', None)

    if model == 'deep_fields':
        deepfields_file = 'LoTSS/DR2/pz_deepfields/AllFields_Pz_dat_Fllim1_{}_Fllim2_0.0_Final_Trapz_CH_Pz.fits'.format(
            flux_cut)
        pz_deepfields = read_fits_to_pandas(os.path.join(DATA_PATH, deepfields_file))

        # z_arr = pz_deepfields['zbins']
        # n_arr = pz_deepfields['pz']  # pz_boot_mean
        z_arr = pz_deepfields['z']
        n_arr = pz_deepfields['Nz_weighted_fields']

    elif model == 'tomographer':
        filename = 'LoTSS/DR2/tomographer/{}mJy_{}SNR_srl_catalog_inner.csv'.format(
            config.flux_min_cut, config.signal_to_noise)
        tomographer = pd.read_csv(os.path.join(DATA_PATH, filename))
        z_arr = tomographer['z'][:-1]
        n_arr = tomographer['dNdz_b'][:-1]

    else:
        if z_arr is None:
            z_step = 0.01
            z_min = 0
            z_max = z_max + z_step
            z_arr = np.arange(z_min, z_max, z_step)

        if model == 'power_law':
            # n_arr = (z_arr ** 2) / (1 + z_arr) * (np.exp((-z_arr / z_sfg)) + r * np.exp(-z_arr / z_agn))
            n_arr = n * (z_arr ** 2) / (1 + z_arr) * (np.exp((-z_arr / z_sfg)) + r ** 2 / (1 + z_arr) ** a)
        elif model == 'double_power_law':
            n_arr = n * (z_arr ** 2) / (1 + z_arr) * (np.exp((-z_arr / z_sfg)) + r ** 2 / (1 + z_arr) ** a)
            z_arr_offset = z_arr - offset
            n_arr_2 = n * (z_arr_offset ** 2) / (1 + z_arr_offset) * (r_2 ** 2 / (1 + z_arr_offset) ** a_2)
            n_arr_2[z_arr <= offset] = 0
            n_arr += n_arr_2
        elif model == 'z_tail':
            z_0 = 0.1
            gamma = 3.5
            n_arr = ((z_arr / z_0) ** 2) / (1 + (z_arr / z_0) ** 2) / (1 + (z_arr / z_tail) ** gamma)
        else:
            raise Exception('Not known redshift distribution model: {}'.format(model))

    if z_max:
        z_arr, n_arr = z_arr[z_arr < z_max], n_arr[z_arr < z_max]

    if A_z_tail:
        n_arr[z_arr > 3.5] *= A_z_tail

    if normalize:
        area = simps(n_arr, z_arr)
        n_arr /= area

    return z_arr, n_arr


def get_biggest_optical_region(data):
    return data.loc[((data['RA'] < 33) | (data['RA'] > 360 - 29)) & (18 < data['DEC']) & (data['DEC'] < 35)]


def read_lotss_noise_weight_map(nside, data_release, flux_min_cut, signal_to_noise):
    # file_path = os.path.join(DATA_PATH,
    #                          'LoTSS/DR{}/weight_maps_pointings/weight_map__pointing-mean_minflux-{}_snr-{}.fits'.format(
    #                              data_release, flux_min_cut, signal_to_noise))
    # weight_map = hp.read_map(file_path)
    # weight_map = hp.ud_grade(weight_map, nside)

    folder = 'LoTSS/DR{}/weight_maps_randoms'
    file_path_in = os.path.join(DATA_PATH, folder, 'Randoms_input_Nside_{}_Fl_{:.1f}mJy.fits')
    file_path_out = os.path.join(DATA_PATH, folder,
                                 'Randoms_output_Nside_{}_SNR_{:.1f}_Fl_{:.1f}mJy_withFluxShift.fits')

    file_path_in = file_path_in.format(data_release, nside, flux_min_cut)
    file_path_out = file_path_out.format(data_release, nside, signal_to_noise, flux_min_cut)

    map_in = hp.read_map(file_path_in)
    map_out = hp.read_map(file_path_out)

    weight_map = np.array([map_out[i] / map_in[i] if map_in[i] != 0 else 0 for i in range(len(map_in))])
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


def get_lotss_map(lotss_data, data_release, flux_min_cut, signal_to_noise, mask_filename=None, nside=2048, masked=True):
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

    weight_map = read_lotss_noise_weight_map(nside, data_release, flux_min_cut, signal_to_noise)

    # Get noise in larger bins
    # noise_map = get_aggregated_map(lotss_data['RA'].values, lotss_data['DEC'].values,
    #                                lotss_data['Isl_rms'].values, nside=256, aggregation='mean')

    # Cut artificially high count pixels
    # if cut_pixels:
    #     indices_max = np.argsort(counts_map)[::-1]
    #     pix_size = hp.pixelfunc.nside2resol(512)
    #     for ipix in indices_max:
    #         if counts_map[ipix] > 11:
    #             vec = hp.pixelfunc.pix2vec(nside, ipix)
    #             radius = pix_size * 2
    #             indices = hp.query_disc(nside, vec, radius)
    #             mask[indices] = 0
    #         else:
    #             break

    return counts_map, mask, weight_map


def get_lotss_dr2_mask(nside, filename=None):
    # Get inner regions as base mask
    # mask = get_dr2_inner_regions(nside)
    mask = hp.read_map(os.path.join(DATA_PATH, 'LoTSS/DR2/masks/mask_inner/mask_inner_nside={}.fits'.format(nside)))
    mask = mask.astype('float64')

    masks_in_files = [
        'mask_coverage', 'mask_default', 'mask_noise_75percent', 'mask_noise_99_percent', 'mask_noise_median']
    mask_b = 1
    if filename in masks_in_files:
        mask_b = hp.read_map(os.path.join(DATA_PATH, 'LoTSS/DR2/masks/{}.fits'.format(filename)))
        mask_b = hp.ud_grade(mask_b, nside)

    elif filename == 'mask_optical':
        mask_b = get_dr2_optical_region(nside)

    mask *= mask_b

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


def get_dr2_inner_regions(nside):
    mask = np.zeros(hp.nside2npix(nside))

    for i in tqdm(range(len(mask))):
        lon, lat = hp.pixelfunc.pix2ang(nside=nside, ipix=i, nest=False, lonlat=True)
        if (
                (0 < lon < 37 and 25 < lat < 40) or
                (0 < lon < 32 and 19 < lat < 25) or
                (113 < lon < 125 and 30 < lat < 39) or
                (125 < lon < 250 and 30 < lat < 68) or
                (193 < lon < 208 and 25 < lat < 30) or
                (250 < lon < 260 and 30 < lat < 45) or
                (335 < lon < 360 and 19 < lat < 35)
        ):
            mask[i] = 1

    # regions = [
    #     ((0, 37), (25, 40)),
    #     ((0, 32), (19, 25)),
    #     ((113, 125), (30, 39)),
    #     ((125, 250), (30, 68)),
    #     ((193, 208), (25, 30)),
    #     ((250, 260), (30, 45)),
    #     ((335, 360), (19, 35)),
    # ]
    #
    # for region in regions:
    #     lon_min = region[0][0]
    #     lon_max = region[0][1]
    #     lat_min = region[1][0]
    #     lat_max = region[1][1]
    #
    #     a = hp.pixelfunc.ang2vec(lon_min, lat_min, lonlat=True)
    #     b = hp.pixelfunc.ang2vec(lon_min, lat_max, lonlat=True)
    #     c = hp.pixelfunc.ang2vec(lon_max, lat_max, lonlat=True)
    #     d = hp.pixelfunc.ang2vec(lon_max, lat_min, lonlat=True)
    #
    #     indices = hp.query_polygon(nside=nside, vertices=[a, b, c, d], inclusive=False)
    #     mask[indices] = 1

    return mask


# TODO: save to files at higher nside also
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


def get_lotss_data(data_release, flux_min_cut=2, signal_to_noise=None, optical=True, is_mock=False, columns=None):
    if is_mock:
        data = pd.read_csv(os.path.join(DATA_PATH, 'LoTSS/DR2/mocks/mock_flask_deepfields/cat_NonLinear.dat'), sep=' ',
                           header=None, names=['RA', 'DEC'])

    else:
        if data_release == 1:
            filename = 'LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2b_restframe.fits' if optical else \
                'LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits'
        elif data_release == 2:
            filename = 'combined-release-v0.1.fits' if optical else 'LoTSS_DR2_v110_masked.srl.fits'
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

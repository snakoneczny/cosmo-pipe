import math
import os

import healpy as hp
import numpy as np
from scipy.integrate import simps
from scipy.special._ufuncs import erfc
from tqdm.notebook import tqdm

from env_config import DATA_PATH
from utils import get_map, get_masked_map, get_mean_map, read_fits_to_pandas


def get_lotss_redshift_distribution(z_tail):
    z_0 = 0.1
    gamma = 3.5

    z_step = 0.01
    z_min = 0
    z_max = 100 + z_step
    z_arr = np.arange(z_min, z_max, z_step)
    n_arr = ((z_arr / z_0) ** 2) / (1 + (z_arr / z_0) ** 2) / (1 + (z_arr / z_tail) ** gamma)
    return z_arr, n_arr


def get_lotss_noise_weight_map(lotss_noise_map, lotss_mask, flux_min_cut, nside_out):
    n_bins = 1000
    flux_max = 2000
    skads = get_skads_sim_data()
    fluxes = skads['S_144'].loc[skads['S_144'] < flux_max]
    #     skads['S_144'].loc[(skads['S_144'] < flux_max)].hist(bins=100)
    #     plt.yscale('log')

    hist, bin_edges = np.histogram(fluxes, bins=n_bins)
    flux_arr = [bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)]

    # Normalize to unit integral
    d_flux = (bin_edges[1] - bin_edges[0])
    area = simps(hist, dx=d_flux)
    flux_proba_arr = hist / area

    integral_error = abs(sum(flux_proba_arr) * d_flux - 1)
    print('Flux probability integral error: {:.4f}'.format(integral_error))
    print('d flux: {:.4f} (mJy)'.format(d_flux))
    assert integral_error < 0.1, 'Integral error equals {:.4f}'.format(integral_error)

    npix = hp.nside2npix(nside=256)  # 12 * nside ^ 2
    noise_weight_map = np.zeros(npix, dtype=np.float)

    # For each pixel
    a = d_flux * 0.5
    for i in tqdm(range(len(lotss_noise_map))):
        if not lotss_noise_map.mask[i]:
            # Integrate d_flux * flux_proba * 0.5 * erfc((flux_true - flux_thr) / (sqrt(2) * sigma_n))
            b = (math.sqrt(2) * lotss_noise_map[i])
            for j in range(len(flux_proba_arr)):
                flux_thr = max(5 * lotss_noise_map[i], flux_min_cut)
                noise_weight_map[i] += a * flux_proba_arr[j] * erfc((flux_arr[j] - flux_thr) / b)

    noise_weight_map = hp.ud_grade(noise_weight_map, nside_out=nside_out)
    noise_weight_map = get_masked_map(noise_weight_map, lotss_mask)

    return noise_weight_map


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
    spectral_index = -0.7
    s_144 = s_151 * math.pow(144, spectral_index) / math.pow(151, spectral_index)
    return s_144


def get_lotss_hetdex_map(lotss_hetdex_data, nside=2048):
    counts_map, _, _ = get_map(lotss_hetdex_data['RA'].values, lotss_hetdex_data['DEC'].values, nside=nside)
    mask = get_lotss_hetdex_mask(nside)
    counts_map = get_masked_map(counts_map, mask)

    # Get noise in larger bins
    noise_map, _, _ = get_mean_map(lotss_hetdex_data['RA'].values, lotss_hetdex_data['DEC'].values,
                                   lotss_hetdex_data['Isl_rms'].values, nside=256)
    # Fill missing pixels with mean noise value  # TODO: print number of missing pixels and sky area of those
    mean_noise = noise_map[~np.isnan(noise_map)].mean()
    noise_map = np.nan_to_num(noise_map, nan=mean_noise)
    noise_map = get_masked_map(noise_map, hp.ud_grade(mask, nside_out=256))

    return counts_map, mask, noise_map


# TODO: process low resolution images?
def get_lotss_hetdex_mask(nside):
    npix = hp.nside2npix(nside)  # 12 * nside ^ 2
    mask = np.zeros(npix, dtype=np.float)
    pointings = np.loadtxt(os.path.join(DATA_PATH, 'LoTSS/DR1/pointings.txt'))
    pointings_to_skip = [[164.633, 54.685], [211.012, 49.912], [221.510, 47.461], [225.340, 47.483], [227.685, 52.515]]
    radius = math.radians(1.7)  # 0.0296706
    for ra, dec in pointings:
        if [ra, dec] in pointings_to_skip:
            continue
        theta, phi = np.radians(-dec + 90.), np.radians(ra)
        vec = hp.pixelfunc.ang2vec(theta, phi, lonlat=False)
        indices = hp.query_disc(nside, vec, radius)
        mask[indices] = 1
    return mask


def get_lotss_hetdex_data():
    data_path = os.path.join(DATA_PATH, 'LoTSS/DR1', 'LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2.fits')
    data = read_fits_to_pandas(data_path)
    data = get_lotss_hetdex_subsample(data)
    return data


def get_lotss_hetdex_subsample(data):
    print('Original LoTSS hetdex datashape: {}'.format(data.shape))
    # With redshift
    data = data.dropna(subset=['z_best'])
    print('Redshift available: {}'.format(data.shape))
    # S > 2 mJy
    data = data.loc[data['Total_flux'] > 2]
    print('Total flux of S > 2mJy: {}'.format(data.shape))
    # discard S/N < 5
    data = data.loc[data['Total_flux'] / data['E_Total_flux'] > 5]
    print('S/N > 5: {}'.format(data.shape))
    return data

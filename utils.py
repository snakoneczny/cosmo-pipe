import logging
import math
import os
import copy

import numpy as np
from astropy.table import Table
import healpy as hp
import pymaster as nmt
import pyccl as ccl
import yaml
from scipy.integrate import simps
import pandas as pd

from env_config import PROJECT_PATH

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ISWTracer(ccl.Tracer):
    """Specific :class:`Tracer` associated with the integrated Sachs-Wolfe
    effect (ISW). Useful when cross-correlating any low-redshift probe with
    the primary CMB anisotropies. The ISW contribution to the temperature
    fluctuations is:

    .. math::
       \\frac{\\Delta T}{T} = 2\\int_0^{\\chi_{LSS}}d\\chi a\\,\\dot{\\phi}

    Any angular power spectra computed with this tracer, should use
    a three-dimensional power spectrum involving the matter power spectrum.
    The current implementation of this tracers assumes a standard Poisson
    equation relating :math:`\\phi` and :math:`\\delta`, and linear structure
    growth. Although this should be valid  in :math:`\\Lambda`CDM and on
    the large scales the ISW is sensitive to, these approximations must be
    borne in mind.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        zmax (float): maximum redshift up to which we define the
            kernel.
        n_chi (float): number of intervals in the radial comoving
            distance on which we sample the kernel.
    """

    def __init__(self, cosmo, z_max=6., n_chi=1024):
        self.chi_max = ccl.comoving_radial_distance(cosmo, 1. / (1 + z_max))
        chi = np.linspace(0, self.chi_max, n_chi)
        a_arr = ccl.scale_factor_of_chi(cosmo, chi)
        H0 = cosmo['h'] / ccl.physical_constants.CLIGHT_HMPC
        OM = cosmo['Omega_c'] + cosmo['Omega_b']
        Ez = ccl.h_over_h0(cosmo, a_arr)
        fz = ccl.growth_rate(cosmo, a_arr)
        w_arr = 3 * cosmo['T_CMB'] * H0 ** 3 * OM * Ez * chi ** 2 * (1 - fz)

        self._trc = []
        self.add_tracer(cosmo, kernel=(chi, w_arr), der_bessel=-1)


def get_chi_squared(data_vector, model_vector, covariance_matrix):
    inverted_covariance = np.linalg.inv(covariance_matrix)
    diff = data_vector - model_vector
    return diff.dot(inverted_covariance).dot(diff)


def get_corr_mean_diff(corr_a, corr_b, bin_range):
    # Substract the known part of shot noise
    a_cl = corr_a['Cl_gg'] - corr_a['nl_gg']
    b_cl = corr_b['Cl_gg'] - corr_b['nl_gg']

    # Get mean difference on each bin of interest
    diff = a_cl - b_cl
    diff = diff[bin_range[0]:bin_range[1]]
    diff_mean = diff.mean()

    # Get mean error on each bin of interest
    diff_mean_err = {}
    error_methods = ['gauss', 'jackknife'] if 'error_gg_jackknife' in corr_a else ['gauss']
    for error_method in error_methods:
        err_a = corr_a['error_gg_{}'.format(error_method)]
        err_b = corr_b['error_gg_{}'.format(error_method)]
        diff_err = np.sqrt(err_a ** 2 + err_b ** 2)
        diff_err = diff_err[bin_range[0]:bin_range[1]]
        diff_mean_err[error_method] = np.sqrt(np.sum(diff_err ** 2)) / len(diff)

    return diff_mean, diff_mean_err


def get_correlations(map_symbols, masks, maps, correlation_symbols, binnings, noise_curves=None, noise_decoupled=None):
    fields = {}
    workspaces = {}
    correlations = {}
    noise_curves = copy.deepcopy(noise_curves)
    noise_decoupled = copy.deepcopy(noise_decoupled)

    # Get fields
    for map_symbol in map_symbols:
        fields[map_symbol] = nmt.NmtField(masks[map_symbol], [maps[map_symbol]])

    # Get all correlations
    for correlation_symbol in correlation_symbols:
        map_symbol_a = correlation_symbol[0]
        map_symbol_b = correlation_symbol[1]
        correlations[correlation_symbol], workspaces[correlation_symbol] = compute_master(
            fields[map_symbol_a], fields[map_symbol_b], binnings[correlation_symbol])

    # Decouple noise curves
    if noise_curves is not None:
        keys = noise_curves.keys()
        for correlation_symbol in keys:
            if isinstance(noise_curves[correlation_symbol], np.ndarray) and correlation_symbol in workspaces:
                if correlation_symbol == 'gg':
                    noise = workspaces[correlation_symbol].decouple_cell(
                        [noise_curves[correlation_symbol]])[0]
                    noise_decoupled[correlation_symbol] = noise
                    noise_curves[correlation_symbol] = np.mean(noise)
                else:
                    noise_decoupled[correlation_symbol] = decouple_correlation(
                        workspaces[correlation_symbol], noise_curves[correlation_symbol])

    return fields, workspaces, correlations, noise_curves, noise_decoupled


def decouple_correlation(workspace, spectrum):
    return workspace.decouple_cell(workspace.couple_cell([spectrum]))[0]


def compute_master(field_a, field_b, binning):
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field_a, field_b, binning)
    cl_coupled = nmt.compute_coupled_cell(field_a, field_b)
    cl_decoupled = workspace.decouple_cell(cl_coupled)
    return cl_decoupled[0], workspace


def get_correlation_matrix(covariance_matrix):
    correlation_matrix = covariance_matrix.copy()
    for i in range(covariance_matrix.shape[0]):
        for j in range(covariance_matrix.shape[1]):
            correlation_matrix[i, j] = covariance_matrix[i, j] / math.sqrt(
                covariance_matrix[i, i] * covariance_matrix[j, j])
    return correlation_matrix


def get_redshift_distribution(data, n_bins=50, z_col='Z_PHOTO_QSO'):
    n_arr, z_arr = np.histogram(data[z_col], bins=n_bins)
    z_arr = np.array([(z_arr[i + 1] + z_arr[i]) / 2 for i in range(len(z_arr) - 1)])
    return z_arr, n_arr


def get_normalized_dist(data, n_bins=1000, with_print=False):
    hist, bin_edges = np.histogram(data, bins=n_bins)
    x_arr = [bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)]

    # Normalize to unit integral
    dx = (bin_edges[1] - bin_edges[0])
    area = simps(hist, dx=dx)
    proba_arr = hist / area

    integral_error = abs(sum(proba_arr) * dx - 1)
    if with_print:
        print('Integral error: {:.4f}'.format(integral_error))
        print('dx: {:.4f} (mJy)'.format(dx))
    assert integral_error < 0.1, 'Integral error equals {:.4f}'.format(integral_error)

    return x_arr, proba_arr, dx


def get_jackknife_masks(mask, regions, nside):
    jacked_masks = []

    for region in regions:
        lon = region['lon']
        lat = region['lat']
        # TODO: make array of tuples (lon_min, lon_max)
        lon_arr = np.append(np.arange(lon[0], lon[1], (lon[1] - lon[0]) / lon[2]), lon[1])
        lat_arr = np.append(np.arange(lat[0], lat[1], (lat[1] - lat[0]) / lat[2]), lat[1])

        # Iterate through all regions to remove
        for i_lon in range(len(lon_arr) - 1):
            for i_lat in range(len(lat_arr) - 1):
                new_mask = copy.copy(mask)

                # Zero out pixels in the regions to remove
                lon_min = lon_arr[i_lon]
                lon_max = lon_arr[i_lon + 1]
                lat_min = lat_arr[i_lat]
                lat_max = lat_arr[i_lat + 1]

                a = hp.pixelfunc.ang2vec(lon_min, lat_min, lonlat=True)
                b = hp.pixelfunc.ang2vec(lon_min, lat_max, lonlat=True)
                c = hp.pixelfunc.ang2vec(lon_max, lat_max, lonlat=True)
                d = hp.pixelfunc.ang2vec(lon_max, lat_min, lonlat=True)

                indices = hp.query_polygon(nside=nside, vertices=[a, b, c, d], inclusive=False)
                new_mask[indices] = 0

                jacked_masks.append(new_mask)
    return jacked_masks


def process_to_overdensity_map(counts_map, mask, weight_map=None, nside=2048):
    # Add weights to mask, and mask pictures below the min weight
    if weight_map is not None:
        mask *= weight_map
        mask /= mask.max()
        # min_weight = 0.5
        min_weight = np.percentile(mask[mask > 0], 1)
        mask[mask < min_weight] = 0

    # Make all the maps masked ones
    counts_map = get_masked_map(counts_map, mask)
    mask = get_masked_map(mask, mask)
    weight_map = get_masked_map(weight_map, mask)

    # Get overdensity map
    sky_mean = counts_map.sum() / mask.sum()
    overdensity_map = counts_map / mask / sky_mean - 1
    overdensity_map = get_masked_map(overdensity_map, mask)

    # Get shot noise
    noise_curve = np.full(3 * nside, get_shot_noise(counts_map, mask))

    return counts_map, mask, weight_map, overdensity_map, noise_curve


def get_shot_noise(counts_map, mask):
    # sky_frac = np.sum(mask > 0) / np.shape(mask)[0]
    # n_obj = counts_map.sum()
    # shot_noise = 4.0 * math.pi * sky_frac / n_obj
    counts_mean = counts_map.sum() / mask.sum()
    n_pix = len(counts_map)
    density = counts_mean * n_pix / (4 * np.pi)
    shot_noise = mask.sum() / len(mask) / density
    return shot_noise


def add_mask(map, additional_mask):
    map = map.copy()
    map.mask = np.logical_or(map.mask, np.logical_not(additional_mask))
    return map


def get_masked_map(map, mask):
    map = hp.ma(map)
    map.mask = np.logical_not(mask)
    return map


def tansform_map_and_mask_to_nside(map, mask, nside):
    if nside:
        map = hp.pixelfunc.ud_grade(map, nside)
        mask = hp.pixelfunc.ud_grade(mask, nside)
    return map, mask


def get_aggregated_map(ra_arr, dec_arr, v_arr, nside, aggregation='mean'):
    npix = hp.nside2npix(nside)
    mean_map = np.zeros(npix)
    pixel_indices = hp.ang2pix(nside, ra_arr, dec_arr, lonlat=True)
    pix_unique = np.unique(pixel_indices)

    aggregation_func = np.mean if aggregation == 'mean' else np.median
    for i, i_pix in enumerate(pix_unique):
        vals = v_arr[pixel_indices == i_pix]
        mean_map[i_pix] = aggregation_func(vals)

    return mean_map


def get_map(ra_arr, dec_arr, nside=512):
    n_pix = hp.nside2npix(nside)
    pixel_indices = hp.ang2pix(nside, ra_arr, dec_arr, lonlat=True)
    map_n = np.bincount(pixel_indices, minlength=n_pix).astype(float)
    return map_n


def get_pairs(values_arr, join_with=''):
    return [join_with.join(sorted([a, b])) for i, a in enumerate(values_arr) for b in values_arr[i:]]


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


def get_config(config_name):
    configs_file = os.path.join(PROJECT_PATH, 'configs.yml')
    with open(configs_file, 'r') as config_file:
        config = yaml.full_load(config_file)
    config = config[config_name]

    # Dictionary fields, flatten to proper values
    for key in ['z_tail', 'z_sfg', 'a', 'r', 'n', 'b_g', 'b_g_scaled', 'b_0', 'b_1', 'b_2', 'b_eff', 'A_sn',
                'A_z_tail']:
        if key in config:
            config[key] = config[key][config['flux_min_cut']]

    return struct(**config)


# TODO: Save covariance matrices
# TODO: move to experiment as static function
def save_correlations(experiment):
    correlations_filename = get_correlations_filename(experiment)
    file_path = os.path.join(
        PROJECT_PATH, 'outputs/correlations/{}/{}.csv'.format(experiment.config.lss_survey_name, correlations_filename))
    error_methods = experiment.covariance_matrices.keys()

    df = pd.DataFrame()
    for correlation_symbol in experiment.correlation_symbols:
        df['l'] = experiment.binnings[correlation_symbol].get_effective_ells()
        df['Cl_{}'.format(correlation_symbol)] = experiment.data_correlations[correlation_symbol]
        df['nl_{}'.format(correlation_symbol)] = experiment.noise_decoupled[correlation_symbol]
        df['nl_{}_mean'.format(correlation_symbol)] = experiment.noise_curves[correlation_symbol]

        for error_method in error_methods:
            df['error_{}_{}'.format(correlation_symbol, error_method)] = \
                experiment.errors[error_method][correlation_symbol]

        if correlation_symbol == 'gg' and experiment.with_multicomp_noise:
            df['nl_gg_multicomp'] = experiment.multicomp_noise
            for error_method in error_methods:
                df['error_nl_gg_multicomp_{}'.format(error_method)] = experiment.multicomp_noise_err[error_method]

    df.to_csv(file_path, index=False)
    print('Correlations saved to: {}'.format(file_path))

    folder_path = os.path.join(PROJECT_PATH, 'outputs/correlations/{}/{}_cov'.format(
        experiment.config.lss_survey_name, correlations_filename, correlations_filename))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    for error_method in error_methods:
        for correlation_symbol_a in experiment.correlation_symbols:
            for correlation_symbol_b in experiment.correlation_symbols:
                covariance_symbol = '{}-{}'.format(correlation_symbol_a, correlation_symbol_b)
                file_path = os.path.join(folder_path, '{}_cov-{}-{}.csv'.format(
                    correlations_filename, covariance_symbol, error_method))
                np.savetxt(file_path,
                           experiment.covariance_matrices[error_method][covariance_symbol], delimiter=',')


def read_correlations(filename=None, experiment=None):
    if experiment:
        experiment_name = get_correlations_filename(experiment)
        filename = '{}/{}'.format(experiment.config.lss_survey_name, experiment_name)
    file_path = os.path.join(PROJECT_PATH, 'outputs/correlations/{}.csv'.format(filename))
    correlations = pd.read_csv(file_path)
    return correlations


def read_covariances(experiment):
    experiment_name = get_correlations_filename(experiment)
    filename = '{}/{}'.format(experiment.config.lss_survey_name, experiment_name)
    error_methods = ['gauss', 'jackknife'] if experiment.config.error_method == 'jackknife' else ['gauss']
    covariance_matrices = dict([(error_method, {}) for error_method in error_methods])
    for error_method in error_methods:
        for correlation_symbol_a in experiment.correlation_symbols:
            for correlation_symbol_b in experiment.correlation_symbols:
                covariance_symbol = '{}-{}'.format(correlation_symbol_a, correlation_symbol_b)
                file_path = os.path.join(PROJECT_PATH, 'outputs/correlations/{}_cov/{}_cov-{}-{}.csv'.format(
                    filename, experiment_name, covariance_symbol, error_method))
                if os.path.exists(file_path):
                    covariance_matrices[error_method][covariance_symbol] = np.loadtxt(file_path, delimiter=',')
    return covariance_matrices


def get_correlations_filename(experiment):
    config = experiment.config
    optical_name = 'opt' if config.is_optical else 'srl'
    experiment_name = '{}_{}__{}__{}mJy_snr={}_nside={}_{}_bin={}'.format(
        config.lss_survey_name, optical_name, config.lss_mask_name, config.flux_min_cut, config.signal_to_noise,
        config.nside, '-'.join(experiment.correlation_symbols), config.ells_per_bin['gg']
    )
    return experiment_name

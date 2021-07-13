import os
from collections import defaultdict
from functools import partial
import math
import copy
from collections import OrderedDict

import numpy as np
import pymaster as nmt
import pyccl as ccl
import emcee
import seaborn as sns
import matplotlib.pyplot as plt
import json
import yaml
from tqdm.notebook import tqdm

from env_config import PROJECT_PATH
from utils import logger, get_shot_noise, get_overdensity_map, get_pairs, compute_master, get_correlation_matrix, \
    get_redshift_distribution, ISWTracer, get_chi_squared, decouple_correlation, merge_mask_with_weights, \
    read_correlations
from data_lotss import get_lotss_data, get_lotss_map, get_lotss_redshift_distribution, read_lotss_noise_weight_map
from data_nvss import get_nvss_map, get_nvss_redshift_distribution
from data_kids_qso import get_kids_qsos, get_kids_qso_map
from data_cmb import get_cmb_lensing_map, get_cmb_lensing_noise, get_cmb_temperature_map, \
    get_cmb_temperature_power_spectra


class Experiment:
    def __init__(self, config, set_data=False, set_maps=False, set_correlations=False):
        # Data parameters
        self.lss_survey_name = None
        self.is_optical = True
        self.lss_mask_name = None
        self.signal_to_noise = None
        self.flux_min_cut = 0
        self.nside = 0
        self.z_tail = 0
        self.bias = 0
        self.scale_bias = None

        # Correlation parameters
        self.correlation_symbols = []
        self.l_range = {}
        self.ells_per_bin = {}
        self.cosmology_name = None
        self.cosmology_matter_power_spectrum = None
        self.cosmology_params = None

        # MCMC parameters
        self.continue_sampling = False
        self.n_walkers = 0
        self.max_iterations = 0
        self.starting_params = {}

        # Data containters
        self.map_symbols = []
        self.data = {}
        self.base_maps = {}
        self.noise_maps = {}
        self.weight_maps = {}
        self.processed_maps = {}
        self.masks = {}
        self.noise_curves = defaultdict(int)
        self.noise_decoupled = defaultdict(int)

        # Correlation containers
        self.z_arr = []
        self.n_arr = []
        self.theory_correlations = {}
        self.data_correlations = {}
        self.raw_data_correlations = {}
        self.chi_squared = {}
        self.sigmas = {}
        self.fields = {}
        self.workspaces = {}
        self.binnings = {}
        self.covariance_matrices = {}
        self.correlation_matrices = {}
        self.l_arr = None
        self.bin_range = {}
        self.n_ells = {}

        # MCMC containers
        self.inference_covariance = None
        self.inference_correlation = None
        self.data_vector = None
        self.inverted_covariance = None
        self.p0_walkers = None
        self.emcee_sampler = None
        self.arg_names = None
        self.backend_filename = None
        self.tau_filename = None
        self.mcmc_folder = os.path.join(PROJECT_PATH, 'outputs/MCMC')

        # Pipeline flags
        self.are_maps_ready = False
        self.are_correlations_ready = False
        self.is_sampler_ready = False

        self.config = config
        for key, value in config.items():
            setattr(self, key, value)

        # Generate necessary correlation symbols
        self.correlation_symbols = list(self.ells_per_bin.keys())
        self.map_symbols = list(set(''.join(self.correlation_symbols)))
        self.all_correlation_symbols = get_pairs(self.map_symbols)

        # Create experiment name
        self.arg_names = list(self.starting_params.keys())
        experiment_name_parts = ['-'.join(self.correlation_symbols), '_'.join(self.arg_names)]
        if 'experiment_tag' in config and config['experiment_tag'] is not None and len(config['experiment_tag']) > 0:
            experiment_name_parts.append(config['experiment_tag'])
        self.experiment_name = '__'.join(experiment_name_parts)

        # Set maps and correlations
        if set_data:
            self.set_data()
        if set_maps:
            self.set_maps()
        if set_correlations:
            self.set_correlations()

    def run_emcee(self):
        if not self.are_maps_ready:
            self.set_maps()
        if not self.are_correlations_ready:
            self.set_correlations()
        self.set_emcee_sampler()

        config_file_path = os.path.join(self.mcmc_folder, '{}.config.json'.format(self.experiment_name))
        with open(config_file_path, 'w') as outfile:
            json.dump(self.config, outfile)
            logger.info('Experiment config saved to: {}'.format(config_file_path))

        tau_arr = np.load(self.tau_filename) if (
                os.path.isfile(self.tau_filename) and self.continue_sampling) else np.array([])
        if not self.continue_sampling:
            self.emcee_sampler.reset()

        for _ in self.emcee_sampler.sample(self.p0_walkers, iterations=self.max_iterations, progress=True):
            tau = self.emcee_sampler.get_autocorr_time(tol=0)
            tau_arr = np.append(tau_arr, [np.mean(tau)])
            np.save(self.tau_filename, tau_arr)

            if len(tau_arr) > 1:
                tau_change = np.abs(tau_arr[-2] - tau) / tau
                converged = np.all(tau * 100 < self.emcee_sampler.iteration)
                converged &= np.all(tau_change < 0.01)
                if not self.emcee_sampler.iteration % 100:
                    logger.info(
                        'Iteration: {}, tau: {}, tau change: {}'.format(self.emcee_sampler.iteration, tau, tau_change))
                if converged:
                    break

    def set_emcee_sampler(self):
        assert self.are_correlations_ready

        self.set_walkers_starting_params()

        self.backend_filename = os.path.join(self.mcmc_folder, '{}.h5'.format(self.experiment_name))
        self.tau_filename = os.path.join(self.mcmc_folder, '{}.tau.npy'.format(self.experiment_name))
        logger.info('emcee backend file path: {}'.format(self.backend_filename))
        logger.info('emcee tau file path: {}'.format(self.tau_filename))

        n_walkers = self.p0_walkers.shape[0]
        n_dim = self.p0_walkers.shape[1]
        backend = emcee.backends.HDFBackend(self.backend_filename)
        if not self.continue_sampling:
            backend.reset(n_walkers, n_dim)
        self.emcee_sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.get_log_prob, backend=backend)

        self.is_sampler_ready = True

    def get_log_prob(self, theta):
        # Check the priors
        log_prior = self.get_log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf

        # Update default cosmological parameters with new sampled parameters
        params = self.cosmology_params.copy()
        for param_name in self.arg_names:
            if param_name in params:
                params[param_name] = theta[self.arg_names.index(param_name)]
        cosmo = ccl.Cosmology(**params)

        # Update data parameters
        z_tail = theta[self.arg_names.index('z_tail')] if 'z_tail' in self.arg_names else self.z_tail
        bias = theta[self.arg_names.index('bias')] if 'bias' in self.arg_names else self.bias

        # Get redshift distribution
        z_arr, n_arr = get_lotss_redshift_distribution(z_tail=z_tail)
        bias_arr = bias * np.ones(len(z_arr))
        bias_arr = bias_arr / ccl.growth_factor(cosmo, 1. / (1 + z_arr))

        # Get correlations
        number_counts_tracer = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr))
        cmb_lensing_tracer = ccl.CMBLensingTracer(cosmo, 1091)
        correlations = {}
        if 'gg' in self.correlation_symbols:
            correlations['gg'] = ccl.angular_cl(cosmo, number_counts_tracer, number_counts_tracer, self.l_arr)
        if 'gk' in self.correlation_symbols:
            correlations['gk'] = ccl.angular_cl(cosmo, number_counts_tracer, cmb_lensing_tracer, self.l_arr)

        # Bin spectra using coupling matrices in workspaces
        for correlation_symbol in self.correlation_symbols:
            correlations[correlation_symbol] = decouple_correlation(self.workspaces[correlation_symbol],
                                                                    correlations[correlation_symbol])
            correlations[correlation_symbol] = correlations[correlation_symbol][:self.n_ells[correlation_symbol]]

        # Calculate log prob
        model = np.concatenate(
            [correlations[correlation_symbol][:self.n_ells[correlation_symbol]] for correlation_symbol in
             self.correlation_symbols])
        diff = self.data_vector - model
        log_prob = log_prior - np.dot(diff, np.dot(self.inverted_covariance, diff)) / 2.0

        return log_prob

    def get_log_prior(self, theta):
        prior = 0
        if 'bias' in self.arg_names and (
                theta[self.arg_names.index('bias')] < 0.6 or theta[self.arg_names.index('bias')] > 6):
            prior = -np.inf
        if 'sigma8' in self.arg_names and (
                theta[self.arg_names.index('sigma8')] < 0.2 or theta[self.arg_names.index('sigma8')] > 2):
            prior = -np.inf
        return prior

    def set_walkers_starting_params(self):
        p0 = np.array([self.starting_params[key][0] for key in self.starting_params])
        p0_scales = np.array([self.starting_params[key][1] for key in self.starting_params])
        n_dim = len(p0)
        self.p0_walkers = np.array(
            [p0 + p0_scales * np.random.uniform(low=-1, high=1, size=n_dim) for i in range(self.n_walkers)])

    def set_correlations(self, with_covariance=True):
        assert self.are_maps_ready

        self.set_binning()
        self.set_data_correlations()
        self.set_theory_correlations()

        if with_covariance:
            self.set_covariance_matrices()
            self.set_sigmas()

            self.set_inference_covariance()
            self.inverted_covariance = np.linalg.inv(self.inference_covariance)
            self.set_data_vector()

            self.are_correlations_ready = True

    def set_sigmas(self):
        for corr_symbol in self.data_correlations:
            bin_range = self.bin_range[corr_symbol]
            data = self.data_correlations[corr_symbol][bin_range[0]:bin_range[1]]
            cov_matrix = self.covariance_matrices[corr_symbol + '-' + corr_symbol]
            cov_matrix = cov_matrix[bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]]

            model = self.theory_correlations[corr_symbol]
            model = decouple_correlation(self.workspaces[corr_symbol], model)[bin_range[0]:bin_range[1]]
            self.chi_squared[corr_symbol] = get_chi_squared(data, model, cov_matrix)

            zero_chi_squared = get_chi_squared(data, 0, cov_matrix)
            diff = zero_chi_squared - self.chi_squared[corr_symbol]
            self.sigmas[corr_symbol] = math.sqrt(diff) if diff > 0 else None

    def set_inference_covariance(self):
        total_length = sum(self.n_ells.values())
        self.inference_covariance = np.empty((total_length, total_length))

        a_start = 0
        for i, corr_symbol_a in enumerate(self.correlation_symbols):
            bin_range_a = self.bin_range[corr_symbol_a]
            n_ells_a = self.n_ells[corr_symbol_a]

            b_start = 0
            for j, corr_symbol_b in enumerate(self.correlation_symbols):
                bin_range_b = self.bin_range[corr_symbol_b]
                n_ells_b = self.n_ells[corr_symbol_b]

                a_end = a_start + n_ells_a
                b_end = b_start + n_ells_b

                # TODO: make sure the order is right, fix last indexing
                self.inference_covariance[a_start: a_end, b_start: b_end] = self.covariance_matrices[
                                                                                corr_symbol_b + '-' + corr_symbol_a][
                                                                            bin_range_a[0]:bin_range_a[1],
                                                                            bin_range_b[0]:bin_range_b[1]]

                b_start += n_ells_b
            a_start += n_ells_a

        self.inference_correlation = get_correlation_matrix(self.inference_covariance)

    def set_covariance_matrices(self):
        correlation_pairs = get_pairs(self.correlation_symbols, join_with='-')
        for correlation_pair in tqdm(correlation_pairs, desc='covariance matrices'):
            a1 = correlation_pair[0]
            a2 = correlation_pair[1]
            b1 = correlation_pair[3]
            b2 = correlation_pair[4]

            covariance_workspace = nmt.NmtCovarianceWorkspace()
            covariance_workspace.compute_coupling_coefficients(
                self.fields[a1], self.fields[a2], self.fields[b1], self.fields[b2]
            )

            self.covariance_matrices[correlation_pair] = nmt.gaussian_covariance(
                covariance_workspace,
                0, 0, 0, 0,
                [self.theory_correlations[''.join(sorted([a1, b1]))]],
                [self.theory_correlations[''.join(sorted([a1, b2]))]],
                [self.theory_correlations[''.join(sorted([a2, b1]))]],
                [self.theory_correlations[''.join(sorted([a2, b2]))]],
                wa=self.workspaces[a1 + a2],
                wb=self.workspaces[b1 + b2],
            )

            transpose_corr_symbol = b1 + b2 + '-' + a1 + a2
            self.covariance_matrices[transpose_corr_symbol] = np.transpose(self.covariance_matrices[correlation_pair])

            if a1 + a2 == b1 + b2:
                self.correlation_matrices[correlation_pair] = get_correlation_matrix(
                    self.covariance_matrices[correlation_pair])

    def set_data_vector(self):
        data_vectors = []
        for correlation_symbol in self.correlation_symbols:
            # Subtract noise from data because theory spectra are created without noise during the sampling
            noise = self.noise_decoupled[correlation_symbol]
            bin_range = self.bin_range[correlation_symbol]
            noise = noise[bin_range[0]:bin_range[1]] if isinstance(noise, np.ndarray) else noise
            data_vectors.append(self.data_correlations[correlation_symbol][bin_range[0]:bin_range[1]] - noise)
        self.data_vector = np.concatenate(data_vectors)

    def set_data_correlations(self):
        # Get fields
        for map_symbol in self.map_symbols:
            self.fields[map_symbol] = nmt.NmtField(self.masks[map_symbol], [self.processed_maps[map_symbol]])

        # Get all correlations
        for correlation_symbol in tqdm(self.correlation_symbols, desc='data correlations'):
            map_symbol_a = correlation_symbol[0]
            map_symbol_b = correlation_symbol[1]
            self.data_correlations[correlation_symbol], self.workspaces[correlation_symbol] = compute_master(
                self.fields[map_symbol_a], self.fields[map_symbol_b], self.binnings[correlation_symbol])

        # Decouple noise curves
        keys = self.noise_curves.keys()
        for correlation_symbol in keys:
            if isinstance(self.noise_curves[correlation_symbol], np.ndarray) and correlation_symbol in self.workspaces:
                if correlation_symbol == 'gg':
                    noise_decoupled = self.workspaces[correlation_symbol].decouple_cell(
                        [self.noise_curves[correlation_symbol]])[0]
                    self.noise_decoupled[correlation_symbol] = noise_decoupled
                    self.noise_curves[correlation_symbol] = np.mean(noise_decoupled)
                else:
                    self.noise_decoupled[correlation_symbol] = decouple_correlation(
                        self.workspaces[correlation_symbol], self.noise_curves[correlation_symbol])

        # Scale auto-correlations for LoTSS DR2 non-optical data
        # TODO: what about scaling gg in case of gt? is it needed? probably not
        if self.lss_survey_name == 'LoTSS_DR2' and not self.is_optical and 'gt' not in self.correlation_symbols:
            # TODO: use get correlations filename function
            fname_template = 'LoTSS_DR1/LoTSS_DR1_{}_{}mJy_snr={}_nside={}_gg-gk_bin={}'
            fname_optical = fname_template.format('optical', self.flux_min_cut, self.signal_to_noise, self.nside,
                                                  self.ells_per_bin['gg'])
            fname_srl = fname_template.format('srl', self.flux_min_cut, self.signal_to_noise, self.nside,
                                              self.ells_per_bin['gg'])
            corr_optical = read_correlations(fname_optical)
            corr_srl = read_correlations(fname_srl)
            if corr_optical is not None and corr_srl is not None:
                ratio = (corr_optical['Cl_gg'] - corr_optical['nl_gg']) / (corr_srl['Cl_gg'] - corr_srl['nl_gg'])
                self.raw_data_correlations['gg'] = self.data_correlations['gg'].copy()
                self.data_correlations['gg'] -= self.noise_curves['gg']
                self.data_correlations['gg'] *= ratio
                self.data_correlations['gg'] += self.noise_curves['gg']

    def set_theory_correlations(self):
        get_redshift_distribution_functions = {
            'LoTSS_DR2': partial(get_lotss_redshift_distribution, z_tail=self.z_tail),
            'LoTSS_DR1': partial(get_lotss_redshift_distribution, z_tail=self.z_tail),
            'NVSS': get_nvss_redshift_distribution,
            # TODO: should include mask (?)
            'KiDS_QSO': partial(get_redshift_distribution, self.data.get('g'), n_bins=50, z_col='Z_PHOTO_QSO')
        }
        self.z_arr, self.n_arr = get_redshift_distribution_functions[self.lss_survey_name]()

        # Get cosmology parameters
        with open(os.path.join(PROJECT_PATH, 'cosmologies.yml'), 'r') as cosmology_file:
            self.cosmology_params = yaml.full_load(cosmology_file)[self.cosmology_name]
        self.cosmology_params['matter_power_spectrum'] = self.cosmology_matter_power_spectrum

        cosmology = ccl.Cosmology(**self.cosmology_params)

        bias_arr = self.bias * np.ones(len(self.z_arr))
        if self.scale_bias:
            bias_arr = bias_arr / ccl.growth_factor(cosmology, 1. / (1. + self.z_arr))

        tracers_dict = {
            'g': ccl.NumberCountsTracer(cosmology, has_rsd=False, dndz=(self.z_arr, self.n_arr),
                                        bias=(self.z_arr, bias_arr)),
            'k': ccl.CMBLensingTracer(cosmology, 1091),
            't': ISWTracer(cosmology, z_max=6., n_chi=1024),
        }

        # TODO: make possible to rerun this function, save symbols for manually set correlations
        for correlation_symbol in self.all_correlation_symbols:
            # Pass if theory correlation was set earlier with maps
            if correlation_symbol not in self.theory_correlations:
                tracer_symbol_a = correlation_symbol[0]
                tracer_symbol_b = correlation_symbol[1]
                self.theory_correlations[correlation_symbol] = ccl.angular_cl(cosmology, tracers_dict[tracer_symbol_a],
                                                                              tracers_dict[tracer_symbol_b], self.l_arr)

        for correlation_symbol in self.theory_correlations.keys():
            self.theory_correlations[correlation_symbol] += self.noise_curves[correlation_symbol]

    def set_binning(self):
        for correlation_symbol in self.correlation_symbols:
            ells_per_bin = self.ells_per_bin[correlation_symbol]

            l_min = self.l_range[correlation_symbol][0]
            l_max = self.l_range[correlation_symbol][1]

            l_min = 2 if not l_min else l_min
            l_max = 3 * self.nside if not l_max else l_max

            # Create binning based on varying bin size
            if isinstance(ells_per_bin, list):
                ell_start_arr = np.array([sum(ells_per_bin[:i]) for i in range(len(ells_per_bin))])
                ell_end_arr = np.array([sum(ells_per_bin[:i+1]) for i in range(len(ells_per_bin))])
                ell_start_arr += 2
                ell_end_arr += 2
                self.binnings[correlation_symbol] = nmt.NmtBin.from_edges(ell_start_arr, ell_end_arr)

                assert (l_max <= ell_end_arr[-1])
                bin_min = sum(ell_start_arr < l_min)
                bin_max = sum(ell_end_arr <= l_max) - 1

            # Create linear binning with constant bin size
            else:
                self.binnings[correlation_symbol] = nmt.NmtBin.from_nside_linear(self.nside, ells_per_bin)

                bin_min = int((l_min - 2) / ells_per_bin)
                bin_max = int((l_max - 2) / ells_per_bin)

            self.bin_range[correlation_symbol] = (bin_min, bin_max)
            self.n_ells[correlation_symbol] = bin_max - bin_min

        # Create dense array of ells, for theoretical power spectra
        self.l_arr = np.arange(3 * self.nside)

    def set_maps(self):
        set_map_functions = {
            'LoTSS_DR2': partial(self.set_lotss_maps, data_release=2),
            'LoTSS_DR1': partial(self.set_lotss_maps, data_release=1),
            'NVSS': self.set_nvss_maps,
            'KiDS_QSO': self.set_kids_qso_maps,
        }

        if 'g' in self.map_symbols:
            set_map_functions[self.lss_survey_name]()
            self.processed_maps['g'] = get_overdensity_map(self.base_maps['g'], self.masks['g'])
            self.noise_curves['gg'] = np.full(3 * self.nside, get_shot_noise(self.base_maps['g'], self.masks['g']))

        if 'k' in self.map_symbols:
            self.base_maps['k'], self.masks['k'] = get_cmb_lensing_map(self.nside)
            self.processed_maps['k'] = self.base_maps['k']
            self.noise_curves['kk'] = get_cmb_lensing_noise(self.nside)

        # TODO: kT correlation
        if 't' in self.map_symbols:
            self.base_maps['t'], self.masks['t'] = get_cmb_temperature_map(nside=self.nside)
            self.processed_maps['t'] = self.base_maps['t']
            self.theory_correlations['tt'] = get_cmb_temperature_power_spectra(self.nside)

        self.are_maps_ready = True

    def set_lotss_maps(self, data_release=2):
        self.base_maps['g'], self.masks['g'], self.noise_maps['g'] = get_lotss_map(
            self.data['g'], data_release=data_release, mask_filename=self.lss_mask_name, nside=self.nside)
        # Probability mask
        self.weight_maps['g'] = read_lotss_noise_weight_map(self.nside, data_release, self.flux_min_cut, 5)
        self.masks['g'] = merge_mask_with_weights(self.masks['g'], self.weight_maps['g'], min_weight=0.5)

        # TODO: delete
        # self.masks['g'][self.base_maps['g'] == 0] = 0

    def set_nvss_maps(self):
        self.base_maps['g'], self.masks['g'] = get_nvss_map(nside=self.nside)

    def set_kids_qso_maps(self):
        self.base_maps['g'], self.masks['g'] = get_kids_qso_map(self.data['g'], self.nside)

    def set_data(self):
        if self.lss_survey_name == 'LoTSS_DR2':
            self.data['g'] = get_lotss_data(data_release=2, flux_min_cut=self.flux_min_cut,
                                            signal_to_noise=self.signal_to_noise, optical=self.is_optical)
        elif self.lss_survey_name == 'LoTSS_DR1':
            self.data['g'] = get_lotss_data(data_release=1, flux_min_cut=self.flux_min_cut,
                                            signal_to_noise=self.signal_to_noise, optical=self.is_optical)
        elif self.lss_survey_name == 'KiDS_QSO':
            self.data['g'] = get_kids_qsos()

    def print_correlation_statistics(self):
        for correlation_symbol in self.correlation_symbols:
            print('C_{} sigma: {:.2f}'.format(correlation_symbol, self.sigmas[correlation_symbol]))
            print('C_{} chi squared: {:.2f}'.format(correlation_symbol, self.chi_squared[correlation_symbol]))


def show_mcmc_report(experiment_name, thin=10):
    mcmc_folder_path = os.path.join(PROJECT_PATH, 'outputs/MCMC')
    with open(os.path.join(mcmc_folder_path, '{}.config.json'.format(experiment_name))) as file:
        config = json.load(file)
    labels = list(config['starting_params'].keys())
    n_walkers = config['n_walkers']

    backend_reader = emcee.backends.HDFBackend(os.path.join(mcmc_folder_path, '{}.h5'.format(experiment_name)))
    tau_arr = np.load(os.path.join(mcmc_folder_path, '{}.tau.npy'.format(experiment_name)))
    emcee_sampler = emcee.EnsembleSampler(n_walkers, len(labels), None, backend=backend_reader)

    print('Mean acceptance fraction: {}'.format(np.mean(emcee_sampler.acceptance_fraction)))
    plot_mean_tau(tau_arr)

    tau = emcee_sampler.get_autocorr_time(tol=0)
    burnin = int(2 * np.max(tau))
    # thin = 1  # int(0.5 * np.min(tau))
    samples = emcee_sampler.get_chain(discard=burnin, flat=False, thin=thin)
    log_prob_samples = emcee_sampler.get_log_prob(discard=burnin, flat=False, thin=thin)
    #     log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

    print('burn-in: {0}'.format(burnin))
    print('thin: {0}'.format(thin))
    print('flat chain shape: {0}'.format(samples.shape))
    print('flat log prob shape: {0}'.format(log_prob_samples.shape))
    #     print('flat log prior shape: {0}'.format(log_prior_samples.shape))

    # all_samples = np.concatenate(
    #     (samples, log_prob_samples[:, None], log_prior_samples[:, None]), axis=1
    # )

    for i in range(len(labels)):
        mcmc = np.percentile(samples[:, :, i].flatten(), [16, 50, 84])
        q = np.diff(mcmc)
        txt = '\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}'
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))

    # TODO: r'$\sigma_8$' r'$b_g$' pretty print function with map
    sns.jointplot(samples[:, :, 1].flatten(), samples[:, :, 0].flatten(), kind='kde',
                  stat_func=None).set_axis_labels(
        labels[1], labels[0])
    sns.jointplot(log_prob_samples.flatten(), samples[:, :, 0].flatten(), kind='kde',
                  stat_func=None).set_axis_labels(
        r'log prob', labels[0])
    sns.jointplot(log_prob_samples.flatten(), samples[:, :, 1].flatten(), kind='kde',
                  stat_func=None).set_axis_labels(
        r'log prob', labels[1])
    plt.show()

    # Samples history
    fig, axes = plt.subplots(len(labels) + 1, sharex='all')
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], 'k', alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.set_xlim(0, samples.shape[0])
    ax = axes[-1]
    ax.plot(log_prob_samples, 'k', alpha=0.3)
    ax.set_ylabel('log prob')
    ax.set_xlabel('step number')
    ax.set_xlim(0, log_prob_samples.shape[0])

    plt.show()


def plot_mean_tau(autocorr_time_arr):
    n = np.arange(1, len(autocorr_time_arr) + 1)
    plt.plot(n, n / 50.0, '--k')
    plt.plot(n, autocorr_time_arr)
    plt.xlabel('number of steps')
    plt.ylabel(r'mean $\hat{\tau}$')
    plt.show()


def run_experiments(config, params_to_update, recalculate_data=False, recalculate_maps=False, with_covariance=True):
    # Create base experiment to prepare data which is common for all experiments
    experiment_base = Experiment(config)
    if not recalculate_data:
        experiment_base.set_data()
    if not recalculate_maps:
        experiment_base.set_maps()

    # TODO: many params simultaneously
    # Iterate through the parameters
    experiments = OrderedDict()
    for param_name, param_arr in params_to_update.items():
        for param_val in tqdm(param_arr):
            # TODO: hack
            label = param_val if not isinstance(param_val, dict) else list(param_val.values())[0]

            # Update experiment parameters
            experiments[label] = copy.deepcopy(experiment_base)
            setattr(experiments[label], param_name, param_val)

            # Set data if not done previously
            if recalculate_data:
                experiments[label].set_data()
            if recalculate_maps:
                experiments[label].set_maps()

            # Set correlations, necessary for every experiment
            experiments[label].set_correlations(with_covariance=with_covariance)

    return experiments

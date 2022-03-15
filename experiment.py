import os
from collections import defaultdict
from functools import partial
import math
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd
import pymaster as nmt
import pyccl as ccl
import emcee
import json
import yaml
from tqdm import tqdm

from env_config import PROJECT_PATH, DATA_PATH
from utils import logger, process_to_overdensity_map, get_pairs, get_correlation_matrix, \
    get_redshift_distribution, get_chi_squared, decouple_correlation, read_correlations, \
    get_corr_mean_diff, get_correlations, get_jackknife_masks, read_covariances
from data_lotss import get_lotss_data, get_lotss_map, get_lotss_redshift_distribution, LOTSS_JACKKNIFE_REGIONS
from data_nvss import get_nvss_map, get_nvss_redshift_distribution
from data_kids_qso import get_kids_qsos, get_kids_qso_map
from data_cmb import get_cmb_lensing_map, get_cmb_lensing_noise, get_cmb_temperature_map, \
    get_cmb_temperature_power_spectra


class Experiment:
    def __init__(self, config, set_data=False, set_maps=False, set_correlations=False):
        # Correlation parameters
        self.correlation_symbols = []
        self.cosmology_params = None

        # Data containters
        self.map_symbols = []
        self.data = {}
        self.base_maps = {}
        self.weight_maps = {}
        self.processed_maps = {}
        self.masks = {}
        self.noise_curves = defaultdict(int)
        self.noise_decoupled = defaultdict(int)
        self.multicomp_noise = None
        self.multicomp_noise_err = {}
        self.get_galaxy_map_function = None

        # Correlation containers
        self.z_arr = []
        self.n_arr = []
        self.theory_correlations = {}
        self.data_correlations = {}
        self.with_multicomp_noise = False
        self.chi_squared = {}
        self.sigmas = {}
        self.fields = {}
        self.workspaces = {}
        self.binnings = {}
        self.covariance_matrices = defaultdict(dict)
        self.errors = defaultdict(dict)
        self.correlation_matrices = defaultdict(dict)
        self.l_arr = None
        self.bin_range = {}
        self.n_ells = {}
        self.cosmology = None

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
        self.tomographer_z_arr = None
        self.tomographer_n_arr = None

        # Pipeline flags
        self.are_data_ready = False
        self.are_maps_ready = False
        self.are_correlations_ready = False

        # Set config
        self.config = config

        # Generate necessary correlation symbols
        self.correlation_symbols = list(config.ells_per_bin.keys())
        self.map_symbols = list(set(''.join(self.correlation_symbols)))
        self.all_correlation_symbols = get_pairs(self.map_symbols)

        # Create experiment name
        self.arg_names = config.to_infere
        experiment_name_parts = ['-'.join(self.correlation_symbols), '_'.join(self.arg_names)]
        if hasattr(config, 'experiment_tag') and config.experiment_tag is not None and len(config.experiment_tag) > 0:
            experiment_name_parts.append(config.experiment_tag)
        self.experiment_name = '__'.join(experiment_name_parts)

        # Set paths
        mcmc_folder = 'outputs/MCMC/{}/{}'.format(config.lss_survey_name, self.experiment_name)
        self.mcmc_folder = os.path.join(PROJECT_PATH, mcmc_folder)

        # Set maps and correlations
        if set_data:
            self.set_data()
        if set_maps:
            self.set_maps()
        if set_correlations:
            self.set_correlations()

    def run_emcee(self):
        assert self.are_correlations_ready

        if not os.path.exists(self.mcmc_folder):
            os.makedirs(self.mcmc_folder)
        self.set_emcee_sampler()

        config_file_path = os.path.join(self.mcmc_folder, '{}.config.json'.format(self.experiment_name))
        with open(config_file_path, 'w') as outfile:
            json.dump(self.config.__dict__, outfile)
            logger.info('Experiment config saved to: {}'.format(config_file_path))

        tau_arr = np.load(self.tau_filename) if (
                os.path.isfile(self.tau_filename) and self.config.continue_sampling) else np.array([])
        if not self.config.continue_sampling:
            self.emcee_sampler.reset()

        for _ in self.emcee_sampler.sample(self.p0_walkers, iterations=self.config.max_iterations, progress=True):
            tau = self.emcee_sampler.get_autocorr_time(tol=0)
            tau_arr = np.append(tau_arr, [np.mean(tau)])
            np.save(self.tau_filename, tau_arr)

            if len(tau_arr) > 1:
                tau_change = np.abs(tau_arr[-2] - tau) / tau
                converged = np.all(tau * 100 < self.emcee_sampler.iteration)
                converged &= np.all(tau_change < 0.01)
                # if not self.emcee_sampler.iteration % 10:
                # logger.info(
                #     'Iteration: {}, tau: {}, tau change: {}'.format(self.emcee_sampler.iteration, np.around(tau, 1),
                #                                                     np.around(tau_change, 3)))
                if converged:
                    break

    def set_emcee_sampler(self):
        self.set_walkers_starting_params()

        self.backend_filename = os.path.join(self.mcmc_folder, '{}.h5'.format(self.experiment_name))
        self.tau_filename = os.path.join(self.mcmc_folder, '{}.tau.npy'.format(self.experiment_name))
        logger.info('emcee backend file path: {}'.format(self.backend_filename))
        logger.info('emcee tau file path: {}'.format(self.tau_filename))

        n_walkers = self.p0_walkers.shape[0]
        n_dim = self.p0_walkers.shape[1]
        backend = emcee.backends.HDFBackend(self.backend_filename)
        if not self.config.continue_sampling:
            backend.reset(n_walkers, n_dim)
        self.emcee_sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.get_log_prob, backend=backend)

    def set_walkers_starting_params(self):
        # TODO: cosmo params, A_sn
        p0 = np.array([self.config.__dict__[key] for key in self.config.to_infere])
        p0_scales = p0 * 0.1
        n_dim = len(p0)
        self.p0_walkers = np.array(
            [p0 + p0_scales * np.random.uniform(low=-1, high=1, size=n_dim) for _ in range(self.config.n_walkers)])

    def get_log_prob(self, theta):
        # TODO: uncomment
        # Check the priors
        # log_prior = self.get_log_prior(theta)
        # if not np.isfinite(log_prior):
        #     return -np.inf
        log_prior = 0

        # Update default cosmological parameters with new sampled parameters
        # TODO: flag indicating whether cosmology changed
        # cosmo_params = self.cosmology_params.copy()
        # for param_name in self.arg_names:
        #     if param_name in cosmo_params:
        #         cosmo_params[param_name] = theta[self.arg_names.index(param_name)]

        # Update data parameters
        config = deepcopy(self.config)
        to_update = dict(zip(self.arg_names, theta))
        config.__dict__.update(to_update)

        # TODO: pass cosmo params
        # Get theory correlations and bin spectra using coupling matrices in workspaces
        try:
            _, _, correlations_dict = self.get_theory_correlations(config, cosmology_params=None,
                                                                   correlation_symbols=self.correlation_symbols)
        except:
            return -np.inf

        model_correlations = []
        for correlation_symbol in self.correlation_symbols:
            correlation = decouple_correlation(self.workspaces[correlation_symbol],
                                               correlations_dict[correlation_symbol])

            if (correlation_symbol == 'gg') and ('A_sn' in self.arg_names):
                correlation += ((config.A_sn - 1) * self.noise_decoupled['gg'])

            bin_range = self.bin_range[correlation_symbol]
            model_correlations.append(correlation[bin_range[0]:bin_range[1]])
        model_correlations = np.concatenate(model_correlations)

        if self.config.fit_tomographer:
            # TODO: refactor
            _, n_arr = get_lotss_redshift_distribution(
                z_sfg=getattr(config, 'z_sfg', None), a=getattr(config, 'a', None), r=getattr(config, 'r', None),
                n=getattr(config, 'n', None), z_tail=getattr(config, 'z_tail', None), z_arr=self.tomographer_z_arr,
                flux_cut=getattr(config, 'flux_min_cut', None), model=config.dn_dz_model, normalize=False)
            model_correlations = np.append(model_correlations, n_arr)

        # Calculate log prob
        # diff = self.data_vector - model_correlations
        diff = self.data_vector - model_correlations
        log_prob = log_prior - np.dot(diff, np.dot(self.inverted_covariance, diff)) / 2.0

        return log_prob

    def get_log_prior(self, theta):
        prior_dict = {
            'b_0_scaled': (0.6, 6.0),
            # 'sigma8': (0.2, 2.0),
            # 'z_tail': (0.5, 2.5),
        }

        prior = 0
        for param in prior_dict:
            if param in self.arg_names:
                param_val = theta[self.arg_names.index(param)]
                if param_val < prior_dict[param][0] or param_val > prior_dict[param][1]:
                    prior = -np.inf

        return prior

    def set_correlations(self, with_covariance=True):
        # assert self.are_maps_ready or self.config.read_correlations_flag
        self.set_binning()

        logger.info('Setting data correlations..')
        if self.config.read_correlations_flag:
            self.read_data_correlations()
        else:
            self.set_data_correlations()

        logger.info('Setting theory correlations..')
        self.set_theory_correlations()

        logger.info('Setting covariance..')
        # TODO: remove with_covariance flag
        if self.config.read_covariance_flag and with_covariance:
            self.covariance_matrices = read_covariances(self)
        else:
            self.set_gauss_covariance()
            if self.config.error_method == 'jackknife':
                self.set_jackknife_covariance()

        self.set_errors()

        if not self.config.read_correlations_flag:
            self.set_sigmas()

        self.set_inference_covariance()
        self.set_data_vector()

        self.are_correlations_ready = True

    def set_sigmas(self):
        for corr_symbol in self.data_correlations:
            bin_range = self.bin_range[corr_symbol]
            data = self.data_correlations[corr_symbol][bin_range[0]:bin_range[1]]
            cov_matrix = self.covariance_matrices[self.config.error_method][corr_symbol + '-' + corr_symbol]
            cov_matrix = cov_matrix[bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]]

            model = self.theory_correlations[corr_symbol]
            model = decouple_correlation(self.workspaces[corr_symbol], model)[bin_range[0]:bin_range[1]]
            self.chi_squared[corr_symbol] = get_chi_squared(data, model, cov_matrix)

            zero_chi_squared = get_chi_squared(data, 0, cov_matrix)
            diff = zero_chi_squared - self.chi_squared[corr_symbol]
            self.sigmas[corr_symbol] = math.sqrt(diff) if diff > 0 else None

    def set_errors(self):
        for error_method in self.covariance_matrices.keys():
            for correlation_symbol in self.correlation_symbols:
                covariance_symbol = '{c}-{c}'.format(c=correlation_symbol)
                self.errors[error_method][correlation_symbol] = np.sqrt(
                    np.diag(self.covariance_matrices[error_method][covariance_symbol]))

    def set_inference_covariance(self):
        total_length = sum(self.n_ells.values())

        if self.config.fit_tomographer:
            # TODO: move to data setting, here more general, add additional part if with tomographer
            tomographer_file = os.path.join(DATA_PATH, 'LoTSS/DR2/tomographer/{}mJy_{}SNR_srl_catalog_{}.csv'.format(
                self.config.flux_min_cut, self.config.signal_to_noise, self.config.lss_mask_name.split('_')[1]))
            tomographer = pd.read_csv(tomographer_file)
            self.tomographer_z_arr = tomographer['z'][:-1]
            self.tomographer_n_arr = tomographer['dNdz_b'][:-1]
            tomographer_n_err_arr = tomographer['dNdz_b_err'][:-1]

            # TODO: what if cosmology changes too?
            # TODO: add other redshift distributions here
            if self.config.bias_model == 'scaled':
                growth_factor = ccl.growth_factor(self.cosmology, 1. / (1. + self.tomographer_z_arr))
                self.tomographer_n_arr *= growth_factor
                tomographer_n_err_arr *= growth_factor

            total_length += len(self.tomographer_z_arr)
        self.inference_covariance = np.zeros((total_length, total_length))

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

                self.inference_covariance[a_start: a_end, b_start: b_end] = \
                    self.covariance_matrices[self.config.error_method][corr_symbol_a + '-' + corr_symbol_b][
                    bin_range_a[0]:bin_range_a[1], bin_range_b[0]:bin_range_b[1]]

                b_start += n_ells_b
            a_start += n_ells_a

        if self.config.fit_tomographer:
            n_tomo = len(self.tomographer_z_arr)
            np.fill_diagonal(self.inference_covariance[-n_tomo:, -n_tomo:], tomographer_n_err_arr ** 2)

        self.inference_correlation = get_correlation_matrix(self.inference_covariance)
        self.inverted_covariance = np.linalg.inv(self.inference_covariance)

    def set_jackknife_covariance(self):
        # TODO: initialize rather than copy, or not use dict at all, its just g that we care here
        # First stage of setting maps
        base_maps = deepcopy(self.base_maps)
        masks = deepcopy(self.masks)
        weight_maps = deepcopy(self.weight_maps)
        processed_maps = deepcopy(self.processed_maps)
        base_maps['g'], masks['g'], weight_maps['g'] = self.get_galaxy_map_function()
        # TODO: weight map changes that
        original_mask_size = masks['g'].nonzero()[0].shape[0]

        # Get jackknife regions
        jacked_masks = get_jackknife_masks(masks['g'], LOTSS_JACKKNIFE_REGIONS, nside=self.config.nside)
        jacked_correlations = dict([(corr_symbol, []) for corr_symbol in self.correlation_symbols])
        jacked_noise = dict([(corr_symbol, []) for corr_symbol in self.correlation_symbols])
        n_regions = len(jacked_masks)

        norm_factors = []
        for jacked_mask in tqdm(jacked_masks):
            # TODO: affected by weight map
            jacked_mask_size = jacked_mask.nonzero()[0].shape[0]

            # Second stage of setting maps, with a jacked mask
            jk_base_maps = deepcopy(base_maps)
            jk_masks = deepcopy(masks)
            jk_weight_maps = deepcopy(weight_maps)
            jk_processed_maps = deepcopy(processed_maps)
            noise_curves = defaultdict(int)
            noise_decoupled = defaultdict(int)
            jk_base_maps['g'], jk_masks['g'], jk_weight_maps['g'], jk_processed_maps['g'], noise_curves[
                'gg'] = process_to_overdensity_map(base_maps['g'], jacked_mask, weight_maps['g'])

            # Calculate all relevant correlations
            _, _, correlations, _, noise_decoupled = get_correlations(
                self.map_symbols, jk_masks, jk_processed_maps, self.correlation_symbols, self.binnings, noise_curves,
                noise_decoupled)

            # Add new correlation
            for corr_symbol in correlations:
                jacked_correlations[corr_symbol].append(correlations[corr_symbol] - noise_decoupled[corr_symbol])
                jacked_noise[corr_symbol].append(noise_decoupled[corr_symbol])

            # Dependence normalizing factor
            norm_factors.append(jacked_mask_size / original_mask_size)

        mean_correlations = dict(
            [(corr_symbol, np.mean(jacked_correlations[corr_symbol], axis=0)) for corr_symbol in jacked_correlations])

        # Calculate covariances
        correlation_pairs = get_pairs(self.correlation_symbols, join_with='-')
        for correlation_pair in correlation_pairs:
            corr_symbol_a = correlation_pair.split('-')[0]
            corr_symbol_b = correlation_pair.split('-')[1]

            m_a = mean_correlations[corr_symbol_a]
            m_b = mean_correlations[corr_symbol_b]

            covariance = np.zeros((len(mean_correlations[corr_symbol_a]), len(mean_correlations[corr_symbol_b])))
            for i in range(covariance.shape[0]):
                for j in range(covariance.shape[1]):
                    for k in range(n_regions):
                        jk_a = jacked_correlations[corr_symbol_a][k]
                        jk_b = jacked_correlations[corr_symbol_b][k]
                        to_add = (norm_factors[k] * (jk_a[i] - m_a[i]) * (jk_b[j] - m_b[j]))
                        covariance[i][j] += to_add

            self.covariance_matrices['jackknife'][correlation_pair] = covariance
            transpose_corr_symbol = corr_symbol_b + '-' + corr_symbol_a
            self.covariance_matrices['jackknife'][transpose_corr_symbol] = np.transpose(covariance)

            if corr_symbol_a == corr_symbol_b:
                self.correlation_matrices['jackknife'][correlation_pair] = get_correlation_matrix(covariance)

    def set_gauss_covariance(self):
        correlation_pairs = get_pairs(self.correlation_symbols, join_with='-')
        for correlation_pair in correlation_pairs:
            a1 = correlation_pair[0]
            a2 = correlation_pair[1]
            b1 = correlation_pair[3]
            b2 = correlation_pair[4]

            covariance_workspace = nmt.NmtCovarianceWorkspace()
            covariance_workspace.compute_coupling_coefficients(
                self.fields[a1], self.fields[a2], self.fields[b1], self.fields[b2]
            )

            self.covariance_matrices['gauss'][correlation_pair] = nmt.gaussian_covariance(
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
            self.covariance_matrices['gauss'][transpose_corr_symbol] = np.transpose(
                self.covariance_matrices['gauss'][correlation_pair])

            if a1 + a2 == b1 + b2:
                self.correlation_matrices['gauss'][correlation_pair] = get_correlation_matrix(
                    self.covariance_matrices['gauss'][correlation_pair])

    def set_data_vector(self):
        data_vectors = []
        for correlation_symbol in self.correlation_symbols:
            # Subtract noise from data because theory spectra are created without noise during the sampling
            noise = self.noise_decoupled[correlation_symbol]
            bin_range = self.bin_range[correlation_symbol]
            noise = noise[bin_range[0]:bin_range[1]] if isinstance(noise, np.ndarray) else noise
            data_vector = self.data_correlations[correlation_symbol][bin_range[0]:bin_range[1]]
            data_vectors.append(data_vector - noise)
        self.data_vector = np.concatenate(data_vectors)

        if self.config.fit_tomographer:
            self.data_vector = np.append(self.data_vector, self.tomographer_n_arr)

    def read_data_correlations(self):
        correlations_df = read_correlations(experiment=self)
        error_methods = ['gauss', 'jackknife'] if self.config.error_method == 'jackknife' else ['gauss']
        for correlation_symbol in self.correlation_symbols:
            self.data_correlations[correlation_symbol] = correlations_df['Cl_{}'.format(correlation_symbol)]
            self.noise_decoupled[correlation_symbol] = correlations_df['nl_{}'.format(correlation_symbol)]
            self.noise_curves[correlation_symbol] = correlations_df['nl_{}_mean'.format(correlation_symbol)][0]
            for error_method in error_methods:
                self.errors[error_method][correlation_symbol] = \
                    correlations_df['error_{}_{}'.format(correlation_symbol, error_method)]
            if 'nl_{}_multicomp'.format(correlation_symbol) in correlations_df:
                self.multicomp_noise = correlations_df['nl_{}_multicomp'.format(correlation_symbol)]
                for error_method in error_methods:
                    self.multicomp_noise_err = \
                        correlations_df['error_nl_{}_multicomp_{}'.format(correlation_symbol, error_method)]

    def set_data_correlations(self):
        self.fields, self.workspaces, self.data_correlations, self.noise_curves, self.noise_decoupled = \
            get_correlations(self.map_symbols, self.masks, self.processed_maps, self.correlation_symbols, self.binnings,
                             self.noise_curves, self.noise_decoupled)

        # Scale auto-correlations for LoTSS DR2 non-optical data
        # TODO: what about scaling gg in case of gt? is it needed? probably not
        # TODO: refactor?
        self.with_multicomp_noise = (
                self.config.lss_survey_name == 'LoTSS_DR2'
                and not self.config.is_optical
                and not self.config.lss_mask_name == 'mask_optical'
                and 'gt' not in self.correlation_symbols
                and 'gg' in self.correlation_symbols
        )
        if self.with_multicomp_noise:
            # TODO: use get correlations filename function
            fname_template = 'LoTSS_DR2/LoTSS_DR2_{}__{}__{}mJy_snr={}_nside={}_gg-gk_bin={}'
            fname_optical = fname_template.format(
                'opt', 'mask_optical', self.config.flux_min_cut, self.config.signal_to_noise, self.config.nside,
                self.config.ells_per_bin['gg']
            )
            fname_srl = fname_template.format(
                'srl', 'mask_optical', self.config.flux_min_cut, self.config.signal_to_noise, self.config.nside,
                self.config.ells_per_bin['gg']
            )
            corr_optical = read_correlations(filename=fname_optical)
            corr_srl = read_correlations(filename=fname_srl)
            if corr_optical is not None and corr_srl is not None:
                self.multicomp_noise, self.multicomp_noise_err = get_corr_mean_diff(corr_srl, corr_optical,
                                                                                    self.bin_range['gg'])
                self.noise_decoupled['gg'] += self.multicomp_noise
                self.noise_curves['gg'] += self.multicomp_noise

        # Once multicomponent shot noise is added, use A_sn to change amplitude to shot noise
        self.noise_decoupled['gg'] *= self.config.A_sn
        self.noise_curves['gg'] *= self.config.A_sn

    def set_theory_correlations(self):
        # Get cosmology params
        with open(os.path.join(PROJECT_PATH, 'cosmologies.yml'), 'r') as cosmology_file:
            self.cosmology_params = yaml.full_load(cosmology_file)[self.config.cosmology_name]
        self.cosmology_params['matter_power_spectrum'] = self.config.cosmology_matter_power_spectrum

        correlations_to_set = [x for x in self.all_correlation_symbols if x not in self.theory_correlations]
        self.z_arr, self.n_arr, correlations_dict = self.get_theory_correlations(self.config, self.cosmology_params,
                                                                                 correlations_to_set)
        self.theory_correlations.update(correlations_dict)

        for correlation_symbol in self.theory_correlations.keys():
            self.theory_correlations[correlation_symbol] += self.noise_curves[correlation_symbol]

    def get_theory_correlations(self, config, cosmology_params, correlation_symbols):
        # Get redshift distribution
        lotss_partial = partial(get_lotss_redshift_distribution, z_sfg=getattr(config, 'z_sfg', None),
                                a=getattr(config, 'a', None), r=getattr(config, 'r', None),
                                n=getattr(config, 'n', None), z_tail=getattr(config, 'z_tail', None),
                                flux_cut=getattr(config, 'flux_min_cut', None), model=config.dn_dz_model,
                                normalize=False)
        get_redshift_distribution_functions = {
            'LoTSS_DR2': lotss_partial,
            'LoTSS_DR1': lotss_partial,
            'NVSS': get_nvss_redshift_distribution,
            # TODO: should include mask (?)
            'KiDS_QSO': partial(get_redshift_distribution, self.data.get('g'), n_bins=50, z_col='Z_PHOTO_QSO')
        }
        z_arr, n_arr = get_redshift_distribution_functions[config.lss_survey_name]()

        # Get cosmology
        if self.cosmology is None:
            self.cosmology = ccl.Cosmology(**cosmology_params)

        # Get bias
        if config.bias_model == 'scaled':
            bias_arr = config.b_0_scaled * np.ones(len(z_arr))
            bias_arr = bias_arr / ccl.growth_factor(self.cosmology, 1. / (1. + z_arr))
        elif config.bias_model == 'polynomial':
            bias_params = [config.b_0, config.b_1, config.b_2]
            bias_arr = sum(bias_params[i] * np.power(z_arr, i) for i in range(len(bias_params)))
        elif config.bias_model == 'tomographer':
            bias_arr = config.b_eff * np.ones(len(z_arr))

        tracers_dict = {
            'g': ccl.NumberCountsTracer(self.cosmology, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr)),
            'k': ccl.CMBLensingTracer(self.cosmology, 1091),
            # 't': ISWTracer(cosmology, z_max=6., n_chi=1024),
        }

        correlations_dict = {}
        for correlation_symbol in correlation_symbols:
            tracer_symbol_a = correlation_symbol[0]
            tracer_symbol_b = correlation_symbol[1]
            correlations_dict[correlation_symbol] = ccl.angular_cl(self.cosmology, tracers_dict[tracer_symbol_a],
                                                                   tracers_dict[tracer_symbol_b], self.l_arr)
        return z_arr, n_arr, correlations_dict

    def set_binning(self):
        for correlation_symbol in self.correlation_symbols:
            ells_per_bin = self.config.ells_per_bin[correlation_symbol]

            l_min = self.config.l_range[correlation_symbol][0]
            l_max = self.config.l_range[correlation_symbol][1]

            l_min = 2 if not l_min else l_min
            l_max = 3 * self.config.nside if not l_max else l_max

            # Create binning based on varying bin size
            if isinstance(ells_per_bin, list):
                ell_start_arr = np.array([sum(ells_per_bin[:i]) for i in range(len(ells_per_bin))])
                ell_end_arr = np.array([sum(ells_per_bin[:i + 1]) for i in range(len(ells_per_bin))])
                ell_start_arr += 2
                ell_end_arr += 2
                self.binnings[correlation_symbol] = nmt.NmtBin.from_edges(ell_start_arr, ell_end_arr)

                assert (l_max <= ell_end_arr[-1])
                bin_min = sum(ell_start_arr < l_min)
                bin_max = sum(ell_end_arr <= l_max) - 1

            # Create linear binning with constant bin size
            else:
                self.binnings[correlation_symbol] = nmt.NmtBin.from_nside_linear(self.config.nside, ells_per_bin)

                bin_min = int((l_min - 2) / ells_per_bin)
                bin_max = int((l_max - 2) / ells_per_bin)

            self.bin_range[correlation_symbol] = (bin_min, bin_max)
            self.n_ells[correlation_symbol] = bin_max - bin_min

        # Create dense array of ells, for theoretical power spectra
        self.l_arr = np.arange(3 * self.config.nside)

    def set_maps(self):
        logger.info('Setting maps..')
        get_map_functions = {
            'LoTSS_DR2': partial(self.get_lotss_maps, data_release=2),
            'LoTSS_DR1': partial(self.get_lotss_maps, data_release=1),
            'NVSS': self.get_nvss_maps,
            'KiDS_QSO': self.get_kids_qso_maps,
        }
        self.get_galaxy_map_function = get_map_functions[self.config.lss_survey_name]

        if 'g' in self.map_symbols:
            # First stage of setting galaxy maps
            self.base_maps['g'], self.masks['g'], self.weight_maps['g'] = self.get_galaxy_map_function()

            # Second stage of setting galaxy maps
            self.base_maps['g'], self.masks['g'], self.weight_maps['g'], self.processed_maps['g'], self.noise_curves[
                'gg'] = process_to_overdensity_map(self.base_maps['g'], self.masks['g'], self.weight_maps['g'])

        if 'k' in self.map_symbols:
            self.base_maps['k'], self.masks['k'] = get_cmb_lensing_map(self.config.nside)
            self.processed_maps['k'] = self.base_maps['k']
            self.noise_curves['kk'] = get_cmb_lensing_noise(self.config.nside)

        # TODO: kT correlation
        if 't' in self.map_symbols:
            self.base_maps['t'], self.masks['t'] = get_cmb_temperature_map(nside=self.config.nside)
            self.processed_maps['t'] = self.base_maps['t']
            self.theory_correlations['tt'] = get_cmb_temperature_power_spectra(self.config.nside)

        self.are_maps_ready = True

    def get_lotss_maps(self, data_release=2):
        mask_filename = None if data_release == 1 else self.config.lss_mask_name
        coutns_map, mask, weight_map = get_lotss_map(self.data['g'], data_release, self.config.flux_min_cut,
                                                     self.config.signal_to_noise, mask_filename=mask_filename,
                                                     nside=self.config.nside)
        return coutns_map, mask, weight_map

    def get_nvss_maps(self):
        coutns_map, mask = get_nvss_map(nside=self.config.nside)
        return coutns_map, mask, None

    def get_kids_qso_maps(self):
        coutns_map, mask = get_kids_qso_map(self.data['g'], self.config.nside)
        return coutns_map, mask, None

    def set_data(self):
        logger.info('Setting data..')
        if self.config.lss_survey_name == 'LoTSS_DR2':
            self.data['g'] = get_lotss_data(data_release=2, flux_min_cut=self.config.flux_min_cut,
                                            signal_to_noise=self.config.signal_to_noise, optical=self.config.is_optical)
        elif self.config.lss_survey_name == 'LoTSS_DR1':
            self.data['g'] = get_lotss_data(data_release=1, flux_min_cut=self.config.flux_min_cut,
                                            signal_to_noise=self.config.signal_to_noise,
                                            optical=self.config.is_optical)
        elif self.config.lss_survey_name == 'KiDS_QSO':
            self.data['g'] = get_kids_qsos()

        self.are_data_ready = True

    def print_correlation_statistics(self):
        for correlation_symbol in self.correlation_symbols:
            print('C_{} sigma: {:.2f}'.format(correlation_symbol, self.sigmas[correlation_symbol]))
            print('C_{} chi squared: {:.2f}'.format(correlation_symbol, self.chi_squared[correlation_symbol]))


# TODO: should me berged with cosmologies.py and Experiment class
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

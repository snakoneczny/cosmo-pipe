import os
from collections import defaultdict
from functools import partial
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import pymaster as nmt
import pyccl as ccl
import emcee
import zeus
import json
import yaml
from tqdm import tqdm
from scipy.interpolate import interp1d

from env_config import PROJECT_PATH, DATA_PATH
from utils import logger, process_to_overdensity_map, get_pairs, get_correlation_matrix, get_redshift_distribution,\
    get_chi_squared, decouple_correlation, read_correlations, get_corr_mean_diff, get_correlations,\
    get_jackknife_masks, read_covariances, read_fits_to_pandas
from data_lotss import get_lotss_data, get_lotss_map, get_lotss_redshift_distribution, LOTSS_JACKKNIFE_REGIONS
from data_nvss import get_nvss_map, get_nvss_redshift_distribution
from data_kids_qso import get_kids_qsos, get_kids_qso_map
from data_cmb import get_cmb_lensing_map, get_cmb_lensing_noise, get_cmb_temperature_map,\
    get_cmb_temperature_power_spectra


class SaveStatisticsCallback:
    def __init__(self, autocorr_callback, split_r_callback, filename='./chains.h5', ncheck=10):
        self.directory = filename
        self.ncheck = ncheck
        self.autocorr_callback = autocorr_callback
        self.split_r_callback = split_r_callback

    def __call__(self, i, x, y):
        if i % self.ncheck == 0:
            np.save(self.directory, self.autocorr_callback.estimates)


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
        self.get_redshift_dist_function = None

        # MCMC containers
        self.inference_covariance = None
        self.inference_correlation = None
        self.data_vector = None
        self.inverted_covariance = None
        self.p0_walkers = None
        self.arg_names = None
        self.backend_filename = None
        self.tau_filename = None
        self.dz_to_fit = []
        self.dn_dz_to_fit = []
        self.dn_dz_err_to_fit = []

        # Pipeline flags
        self.are_data_ready = False
        self.are_maps_ready = False
        self.are_correlations_ready = False

        # Set constants
        self.config = config
        self.arg_names = config.to_infere
        self.correlation_symbols = list(config.l_range.keys())
        self.map_symbols = list(set(''.join(self.correlation_symbols)))
        self.all_correlation_symbols = get_pairs(self.map_symbols)

        self.set_name()
        self.set_cosmology()
        self.set_dn_dz_function()

        mcmc_folder = 'outputs/MCMC/{}/{}'.format(config.lss_survey_name, self.experiment_name)
        self.mcmc_folder = os.path.join(PROJECT_PATH, mcmc_folder)
        self.mcmc_functions = {
            'zeus': self.run_zeus_sampler,
            'emcee': self.run_emcee_sampler,
        }

        # Set maps and correlations
        if set_data:
            self.set_data()
        if set_maps:
            self.set_maps()
        if set_correlations:
            self.set_correlations()

    def run_mcmc(self):
        assert self.are_correlations_ready

        if not os.path.exists(self.mcmc_folder):
            os.makedirs(self.mcmc_folder)

        self.backend_filename = os.path.join(self.mcmc_folder, '{}.h5'.format(self.experiment_name))
        self.tau_filename = os.path.join(self.mcmc_folder, '{}.tau.npy'.format(self.experiment_name))
        logger.info('Samples/backend file path: {}'.format(self.backend_filename))
        logger.info('Autocorrelation time file path: {}'.format(self.tau_filename))

        config_file_path = os.path.join(self.mcmc_folder, '{}.config.json'.format(self.experiment_name))
        with open(config_file_path, 'w') as outfile:
            json.dump(self.config.__dict__, outfile)
            logger.info('Experiment config saved to: {}'.format(config_file_path))

        self.set_walkers_starting_params()
        self.mcmc_functions[self.config.mcmc_engine]()

    def run_zeus_sampler(self):
        n_walkers = self.p0_walkers.shape[0]
        n_dim = self.p0_walkers.shape[1]

        autocorr_cb = zeus.callbacks.AutocorrelationCallback(ncheck=10, dact=0.01, nact=50, discard=0.5)
        split_r_cb = zeus.callbacks.SplitRCallback(ncheck=10, epsilon=0.01, nsplits=2, discard=0.5)
        min_iter_cb = zeus.callbacks.MinIterCallback(nmin=500)
        save_progress_cb = zeus.callbacks.SaveProgressCallback(self.backend_filename, ncheck=10)
        save_stats_cb = SaveStatisticsCallback(autocorr_cb, split_r_cb, filename=self.tau_filename, ncheck=10)
        callbacks = [autocorr_cb, split_r_cb, min_iter_cb, save_progress_cb, save_stats_cb]

        sampler = zeus.EnsembleSampler(n_walkers, n_dim, self.get_log_prob)
        sampler.run_mcmc(self.p0_walkers, self.config.max_iterations, callbacks=callbacks)
        print(sampler.summary)

    def run_emcee_sampler(self):
        n_walkers = self.p0_walkers.shape[0]
        n_dim = self.p0_walkers.shape[1]
        backend = emcee.backends.HDFBackend(self.backend_filename)
        if not self.config.continue_sampling:
            backend.reset(n_walkers, n_dim)
        emcee_sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.get_log_prob, backend=backend)

        tau_arr = np.load(self.tau_filename) if (
                os.path.isfile(self.tau_filename) and self.config.continue_sampling) else np.array([])

        if not self.config.continue_sampling:
            emcee_sampler.reset()

        for _ in emcee_sampler.sample(self.p0_walkers, iterations=self.config.max_iterations, progress=True):
            tau = emcee_sampler.get_autocorr_time(tol=0)
            tau_arr = np.append(tau_arr, [np.mean(tau)])
            np.save(self.tau_filename, tau_arr)

            if len(tau_arr) > 1:
                tau_change = np.abs(tau_arr[-2] - tau) / tau
                converged = np.all(tau * 50 < emcee_sampler.iteration)
                converged &= np.all(tau_change < 0.1)
                if converged:
                    break

    def set_walkers_starting_params(self):
        p0 = []
        for key in self.config.to_infere:
            if key in self.config.__dict__:
                p0.append(self.config.__dict__[key])
            elif key in self.cosmology_params:
                p0.append(self.cosmology_params[key])
            elif key == 'Omega_m':
                p0.append(self.cosmology_params['Omega_c'] + self.cosmology_params['Omega_b'])
            else:
                raise ValueError('No such parameter: {}'.format(key))

        p0 = np.array(p0)
        p0_scales = p0 * 0.1
        n_dim = len(p0)
        self.p0_walkers = np.array(
            [p0 + p0_scales * np.random.uniform(low=-1, high=1, size=n_dim) for _ in range(self.config.n_walkers)])

    def get_log_prob(self, theta):
        # Check the priors
        log_prior = self.get_log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf

        # Update parameters
        config = deepcopy(self.config)
        to_update = dict(zip(self.arg_names, theta))
        config.__dict__.update(to_update)
        cosmology_params = self.get_updated_cosmology_parameters(to_update)

        # Check the bias prior if present
        if self.config.fit_bias_to_tomo:
            z_arr, n_arr = self.get_redshift_dist_function(config=config, normalize=False)
            bias_arr = self.get_bias(z_arr, self.cosmology, config)
            if (bias_arr <= 0).any():
                return -np.inf

        # Get theory correlations and bin spectra using coupling matrices in workspaces
        try:
            _, _, correlations_dict = self.get_theory_correlations(config, self.config.correlations_to_use,
                                                                   interpolate=False, cosmology_params=cosmology_params)
        except:
            return -np.inf

        model_correlations = []
        for correlation_symbol in correlations_dict:
            correlation = decouple_correlation(self.workspaces[correlation_symbol],
                                               correlations_dict[correlation_symbol])

            if (correlation_symbol == 'gg') and ('A_sn' in self.arg_names):
                correlation += ((config.A_sn - 1) * self.noise_decoupled['gg'])

            bin_range = self.bin_range[correlation_symbol]
            model_correlations.append(correlation[bin_range[0]:bin_range[1]])
        model_correlations = np.concatenate(model_correlations)

        for i, redshift_to_fit in enumerate(self.config.redshifts_to_fit):
            normalize = False if redshift_to_fit == 'tomographer' else True
            z_arr, n_arr = self.get_redshift_dist_function(config=config, z_arr=self.dz_to_fit[i], normalize=normalize)
            if redshift_to_fit == 'tomographer' and self.config.fit_bias_to_tomo:
                bias_arr = self.get_bias(z_arr, self.cosmology, config)
                n_arr *= bias_arr
            model_correlations = np.append(model_correlations, n_arr)

        # Calculate log prob
        diff = self.data_vector - model_correlations
        log_prob = log_prior - np.dot(diff, np.dot(self.inverted_covariance, diff)) / 2.0

        return log_prob

    def get_log_prior(self, theta):
        prior_dict = {
            'A_sn': (-np.inf, np.inf),
            'A_z_tail': (0.5, 2.0),
            'Omega_m': (0, np.inf),
            'sigma8': (0, np.inf),
            'b_g': (0, np.inf),
            'b_g_scaled': (0, np.inf),
            'b_0': (-np.inf, np.inf),
            'b_1': (-np.inf, np.inf),
            'b_2': (-np.inf, np.inf),
            'z_sfg': (0, np.inf),
            'r': (0, np.inf),
            'n': (0, np.inf),
        }

        prior = 0
        for param in prior_dict:
            if param in self.arg_names:
                param_val = theta[self.arg_names.index(param)]
                if param_val < prior_dict[param][0] or param_val > prior_dict[param][1]:
                    prior = -np.inf
                    break

        return prior

    def get_updated_cosmology_parameters(self, to_update):
        cosmology_params = deepcopy(self.cosmology_params)
        params_to_update = list(cosmology_params) + ['Omega_m']
        params_to_update = [name for name in self.arg_names if name in params_to_update]
        for param_name in params_to_update:
            if param_name == 'Omega_m':
                baryon_fraction = 0.05 / 0.3
                Omega_m = to_update['Omega_m']
                cosmology_params['Omega_c'] = Omega_m * (1 - baryon_fraction)
                cosmology_params['Omega_b'] = Omega_m * baryon_fraction
            else:
                cosmology_params[param_name] = to_update[param_name]
        return cosmology_params

    def set_correlations(self):
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
        if self.config.read_covariance_flag:
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
        total_length = sum([self.n_ells[key] for key in self.config.correlations_to_use])

        # TODO: move to data setting, here more general
        for redshift_to_fit in self.config.redshifts_to_fit:
            if redshift_to_fit == 'tomographer':
                tomographer_file = os.path.join(DATA_PATH,
                                                'LoTSS/DR2/tomographer/{}mJy_{}SNR_srl_catalog_{}.csv'.format(
                                                    self.config.flux_min_cut, self.config.signal_to_noise,
                                                    self.config.lss_mask_name.split('_')[1]))
                tomographer = pd.read_csv(tomographer_file)
                self.dz_to_fit.append(tomographer['z'][:-1])
                self.dn_dz_to_fit.append(tomographer['dNdz_b'][:-1])
                self.dn_dz_err_to_fit.append(tomographer['dNdz_b_err'][:-1])

                # Scale tomographer with D(z), it will allow to skip the scaling process in the get_log_prob function
                if self.config.bias_model == 'scaled':
                    growth_factor = ccl.growth_factor(self.cosmology, 1. / (1. + self.dz_to_fit[-1]))
                    self.dn_dz_to_fit[-1] *= growth_factor
                    self.dn_dz_err_to_fit[-1] *= growth_factor

            elif redshift_to_fit == 'deep_fields':
                deepfields_file = 'LoTSS/DR2/pz_deepfields/Pz_booterrors_wsum_deepfields_{:.1f}mJy.fits'.format(
                    self.config.flux_min_cut)
                pz_deepfields = read_fits_to_pandas(os.path.join(DATA_PATH, deepfields_file))
                self.dz_to_fit.append(pz_deepfields['zbins'])
                self.dn_dz_to_fit.append(pz_deepfields['pz_boot_mean'])
                self.dn_dz_err_to_fit.append(pz_deepfields['error_boot'])

            total_length += len(self.dz_to_fit[-1])
        self.inference_covariance = np.zeros((total_length, total_length))

        a_start = 0
        for i, corr_symbol_a in enumerate(self.config.correlations_to_use):
            bin_range_a = self.bin_range[corr_symbol_a]
            n_ells_a = self.n_ells[corr_symbol_a]

            b_start = 0
            for j, corr_symbol_b in enumerate(self.config.correlations_to_use):
                bin_range_b = self.bin_range[corr_symbol_b]
                n_ells_b = self.n_ells[corr_symbol_b]

                a_end = a_start + n_ells_a
                b_end = b_start + n_ells_b

                self.inference_covariance[a_start: a_end, b_start: b_end] = \
                    self.covariance_matrices[self.config.error_method][corr_symbol_a + '-' + corr_symbol_b][
                    bin_range_a[0]:bin_range_a[1], bin_range_b[0]:bin_range_b[1]]

                b_start += n_ells_b
            a_start += n_ells_a

        for i, redshift_to_fit in enumerate(self.config.redshifts_to_fit):
            n_dz_to_fit = len(self.dz_to_fit[i])
            a_end = a_start + n_dz_to_fit
            np.fill_diagonal(self.inference_covariance[a_start: a_end, a_start: a_end],
                             self.dn_dz_err_to_fit[i] ** 2)
            a_start += n_dz_to_fit

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
        for correlation_symbol in self.config.correlations_to_use:
            # Subtract noise from data because theory spectra are created without noise during the sampling
            noise = self.noise_decoupled[correlation_symbol]
            bin_range = self.bin_range[correlation_symbol]
            noise = noise[bin_range[0]:bin_range[1]] if isinstance(noise, np.ndarray) else noise
            data_vector = self.data_correlations[correlation_symbol][bin_range[0]:bin_range[1]]
            data_vectors.append(data_vector - noise)
        self.data_vector = np.concatenate(data_vectors)

        for i, redshift_to_fit in enumerate(self.config.redshifts_to_fit):
            self.data_vector = np.append(self.data_vector, self.dn_dz_to_fit[i])

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
        self.with_multicomp_noise = (
                self.config.lss_survey_name == 'LoTSS_DR2'
                and not self.config.is_optical
                and not self.config.lss_mask_name == 'mask_optical'
                and 'gg' in self.correlation_symbols
        )
        if self.with_multicomp_noise:
            config_tmp = deepcopy(self.config)
            config_tmp.lss_mask_name = 'mask_optical'
            config_tmp.is_optical = True
            corr_optical = read_correlations(config=config_tmp)
            config_tmp.is_optical = False
            corr_srl = read_correlations(config=config_tmp)

            self.multicomp_noise, self.multicomp_noise_err = get_corr_mean_diff(corr_srl, corr_optical,
                                                                                self.bin_range['gg'])
            self.noise_decoupled['gg'] += self.multicomp_noise
            self.noise_curves['gg'] += self.multicomp_noise

        # Once multicomponent shot noise is added, use A_sn to change amplitude to shot noise
        self.noise_decoupled['gg'] *= self.config.A_sn
        self.noise_curves['gg'] *= self.config.A_sn

    def set_theory_correlations(self):
        correlations_to_set = [x for x in self.all_correlation_symbols if x not in self.theory_correlations]
        self.z_arr, self.n_arr, correlations_dict = self.get_theory_correlations(self.config, correlations_to_set)
        self.theory_correlations.update(correlations_dict)

        # TODO: should include all theory correlations if gaussian covariance is to work
        for correlation_symbol in self.correlation_symbols:
            self.theory_correlations[correlation_symbol] += self.noise_curves[correlation_symbol]

    def get_theory_correlations(self, config, correlation_symbols, cosmology_params=None, interpolate=False):
        # Get redshift distribution
        z_arr, n_arr = self.get_redshift_dist_function(config=config)

        # Get cosmology
        cosmology = self.cosmology if ((not cosmology_params) or cosmology_params == self.cosmology_params) else \
            ccl.Cosmology(**cosmology_params)

        # Get bias
        bias_arr = self.get_bias(z_arr, self.cosmology, config)

        # Get tracers
        tracers_dict = {
            'g': ccl.NumberCountsTracer(cosmology, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr)),
            'k': ccl.CMBLensingTracer(cosmology, 1091),
            # 't': ISWTracer(cosmology, z_max=6., n_chi=1024),
        }

        correlations_dict = {}
        for correlation_symbol in correlation_symbols:
            tracer_symbol_a = correlation_symbol[0]
            tracer_symbol_b = correlation_symbol[1]

            ell_arr = self.l_arr
            if interpolate:
                nl_per_decade = 30
                l_min_sample = self.config.l_range[correlation_symbol][0]
                l_max_sample = self.config.l_range[correlation_symbol][1]

                nl_sample = int(np.log10(l_max_sample / l_min_sample) * nl_per_decade)
                ell_arr = np.unique(np.geomspace(l_min_sample, l_max_sample + 1, nl_sample).astype(int)).astype(float)
                if l_min_sample > 0:
                    ell_arr = np.concatenate((np.array([0.]), ell_arr))
                if l_max_sample < 3 * self.config.nside:
                    ell_arr = np.concatenate((ell_arr, np.array([3 * self.config.nside])))

            correlation = ccl.angular_cl(cosmology, tracers_dict[tracer_symbol_a], tracers_dict[tracer_symbol_b],
                                         ell_arr)

            if interpolate:
                f = interp1d(ell_arr, correlation)
                correlation = f(self.l_arr)

            correlations_dict[correlation_symbol] = correlation
        return z_arr, n_arr, correlations_dict

    def get_bias(self, z_arr, cosmology=None, config=None):
        config = config if config else self.config
        cosmology = cosmology if cosmology else self.cosmology

        bias_arr = None
        if config.bias_model == 'constant':
            bias_arr = config.b_g * np.ones(len(z_arr))
        elif config.bias_model == 'scaled':
            bias_arr = config.b_g_scaled * np.ones(len(z_arr))
            bias_arr = bias_arr / ccl.growth_factor(cosmology, 1. / (1. + z_arr))
        elif config.bias_model == 'quadratic':
            bias_params = [config.b_0, config.b_1, config.b_2]
            bias_arr = sum(bias_params[i] * np.power(z_arr, i) for i in range(len(bias_params)))
        elif config.bias_model == 'tomographer':
            bias_arr = config.b_eff * np.ones(len(z_arr))

        return bias_arr

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

    def set_cosmology(self):
        # Get cosmology params
        with open(os.path.join(PROJECT_PATH, 'cosmologies.yml'), 'r') as cosmology_file:
            self.cosmology_params = yaml.full_load(cosmology_file)[self.config.cosmology_name]
        self.cosmology_params['matter_power_spectrum'] = self.config.cosmology_matter_power_spectrum

        # TODO: iterate through self.cosmology_params, if present in self.config then change, remove cosmo params update
        if getattr(self.config, 'sigma8', None):
            self.cosmology_params['sigma8'] = self.config.sigma8

        self.cosmology = ccl.Cosmology(**self.cosmology_params)

    def set_dn_dz_function(self):
        # Get redshift distribution
        lotss_partial = partial(get_lotss_redshift_distribution, config=self.config, normalize=False)
        get_redshift_distribution_functions = {
            'LoTSS_DR2': lotss_partial,
            'LoTSS_DR1': lotss_partial,
            'NVSS': get_nvss_redshift_distribution,
            'KiDS_QSO': partial(get_redshift_distribution, self.data.get('g'), n_bins=50, z_col='Z_PHOTO_QSO'),
        }
        self.get_redshift_dist_function = get_redshift_distribution_functions[self.config.lss_survey_name]

    def set_name(self):
        l_range = self.config.l_range['gg']
        correlations_part = '_'.join([
            '-'.join(self.correlation_symbols),
            'ell-{}-{}'.format(l_range[0], l_range[1]),
        ])

        redshift_part = 'redshift_' + '-'.join(self.config.dn_dz_model.split('_'))
        for redshift_to_fit in self.config.redshifts_to_fit:
            redshift_part += '_' + '-'.join(redshift_to_fit.split('_'))

        bias_part = 'bias_' + self.config.bias_model
        mcmc_part = self.config.mcmc_engine + '_' + '_'.join(self.arg_names)
        experiment_name_parts = [correlations_part, redshift_part, bias_part, mcmc_part]

        if hasattr(self.config, 'experiment_tag') and self.config.experiment_tag is not None and len(
                self.config.experiment_tag) > 0:
            experiment_name_parts.append(self.config.experiment_tag)

        self.experiment_name = '__'.join(experiment_name_parts)

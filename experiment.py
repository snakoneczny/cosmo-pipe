import os
from collections import defaultdict
from functools import partial

import numpy as np
import pymaster as nmt
import pyccl as ccl
import emcee
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Math
import json

from env_config import PROJECT_PATH
from utils import logger, get_shot_noise, get_overdensity_map, get_pairs, compute_master, get_correlation_matrix, \
    get_redshift_distribution, ISWTracer
from data_lotss import get_lotss_hetdex_data, get_lotss_hetdex_map, get_lotss_noise_weight_map, \
    get_lotss_redshift_distribution
from data_nvss import get_nvss_map, get_nvss_redshift_distribution
from data_kids_qso import get_kids_qsos, get_kids_qso_map
from data_cmb import get_cmb_lensing_map, get_cmb_lensing_noise, get_cmb_temperature_map


class Experiment:
    def __init__(self, config, set_maps=False, set_correlations=False):
        # Input spectra and all required map symbols
        self.correlation_symbols = []
        self.map_symbols = []

        # Parameters
        self.l_min = 0
        self.l_max = 0
        self.ells_per_bin = 0
        self.ell_lengths = None
        self.lotss_flux_min_cut = 0.0  # mJy
        self.nside = 0
        self.z_tail = 0.0
        self.bias = 0.0
        self.scale_bias = None
        self.continue_sampling = False
        self.n_walkers = 0
        self.max_iterations = 0
        self.starting_params = {}
        self.default_params = {}
        self.lss_survey = None
        self.cosmology_matter_power_spectrum = 'linear'

        # Data containters
        self.data = {}
        self.masks = {}
        self.original_maps = {}
        self.processed_maps = {}
        self.noise_maps = {}
        self.noise_weight_maps = {}
        self.noise_curves = defaultdict(int)
        self.noise_decoupled = defaultdict(int)

        # Correlation containers
        self.z_arr = []
        self.n_arr = []
        self.theory_correlations = {}
        self.data_correlations = {}
        self.fields = {}
        self.workspaces = {}
        self.binning = None
        self.covariance_matrices = {}
        self.correlation_matrices = {}
        self.l_arr = None
        self.n_ells = None

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

        self.map_symbols = list(set(''.join(self.correlation_symbols)))
        self.all_correlation_symbols = get_pairs(self.map_symbols)

        self.arg_names = list(self.starting_params.keys())
        experiment_name_parts = ['-'.join(self.correlation_symbols), '_'.join(self.arg_names)]
        if 'experiment_tag' in config and config['experiment_tag'] is not None and len(config['experiment_tag']) > 0:
            experiment_name_parts.append(config['experiment_tag'])
        self.experiment_name = '__'.join(experiment_name_parts)

        self.set_binning()
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

        self.set_data_vector()
        self.inverted_covariance = np.linalg.inv(self.inference_covariance)
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

        # Update default parameters with given parameters
        params = self.default_params.copy()
        for param_name in self.arg_names:
            params[param_name] = theta[self.arg_names.index(param_name)]

        cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=params['sigma8'], n_s=0.96,
                              matter_power_spectrum=self.cosmology_matter_power_spectrum)

        z_arr, n_arr = get_lotss_redshift_distribution(z_tail=params['z_tail'])
        bias_arr = params['bias'] * np.ones(len(z_arr))
        bias_arr = bias_arr / ccl.growth_factor(cosmo, 1. / (1 + z_arr))

        number_counts_tracer = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr))
        cmb_lensing_tracer = ccl.CMBLensingTracer(cosmo, 1091)

        correlations = {}
        if 'gg' in self.correlation_symbols:
            correlations['gg'] = ccl.angular_cl(cosmo, number_counts_tracer, number_counts_tracer, self.l_arr)
        if 'gk' in self.correlation_symbols:
            correlations['gk'] = ccl.angular_cl(cosmo, number_counts_tracer, cmb_lensing_tracer, self.l_arr)

        for correlation_symbol in self.correlation_symbols:
            correlations[correlation_symbol] = self.workspaces[correlation_symbol].decouple_cell(
                self.workspaces[correlation_symbol].couple_cell([correlations[correlation_symbol]]))[0]

        model = np.concatenate(
            [correlations[correlation_symbol][:self.n_ells] for correlation_symbol in self.correlation_symbols])

        diff = self.data_vector - model
        return log_prior - np.dot(diff, np.dot(self.inverted_covariance, diff)) / 2.0

    def get_log_prior(self, theta):
        prior = 0
        if 'bias' in self.arg_names and (
                theta[self.arg_names.index('bias')] < 0 or theta[self.arg_names.index('bias')] > 6):
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

    def set_data_vector(self):
        data_vectors = []
        for correlation_symbol in self.correlation_symbols:
            # Subtract noise from data because theory spectra are created without noise during the sampling
            noise = self.noise_curves[correlation_symbol]
            noise = noise[:self.n_ells] if isinstance(noise, np.ndarray) else noise
            data_vectors.append(self.data_correlations[correlation_symbol][:self.n_ells] - noise)
        self.data_vector = np.concatenate(data_vectors)

    def set_correlations(self):
        assert self.are_maps_ready

        get_redshift_distribution_functions = {
            'LoTSS_DR1': partial(get_lotss_redshift_distribution, z_tail=self.z_tail),
            'NVSS': get_nvss_redshift_distribution,
            # TODO: should include mask
            'KiDS_QSO': partial(get_redshift_distribution, self.data.get('g'), n_bins=50, z_col='Z_PHOTO_QSO')
        }

        self.z_arr, self.n_arr = get_redshift_distribution_functions[self.lss_survey]()
        self.set_theory_correlations()
        for correlation_symbol in self.theory_correlations.keys():
            self.theory_correlations[correlation_symbol] += self.noise_curves[correlation_symbol]
        self.set_data_correlations()
        self.set_covariance_matrices()
        self.set_inference_covariance()
        self.are_correlations_ready = True

    def set_inference_covariance(self):
        total_length = self.n_ells * len(self.correlation_symbols)
        self.inference_covariance = np.empty((total_length, total_length))

        for i, corr_symbol_a in enumerate(self.correlation_symbols):
            for j, corr_symbol_b in enumerate(self.correlation_symbols):
                a_start = i * self.n_ells
                a_end = (i + 1) * self.n_ells
                b_start = j * self.n_ells
                b_end = (j + 1) * self.n_ells
                # TODO: make sure the order is right
                self.inference_covariance[a_start: a_end, b_start: b_end] = \
                    self.covariance_matrices[corr_symbol_b + '-' + corr_symbol_a][:self.n_ells, :self.n_ells]

        self.inference_correlation = get_correlation_matrix(self.inference_covariance)

    def set_covariance_matrices(self):
        correlation_pairs = get_pairs(self.all_correlation_symbols, join_with='-')
        for correlation_pair in correlation_pairs:
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

            self.covariance_matrices[b1 + b2 + '-' + a1 + a2] = np.transpose(self.covariance_matrices[correlation_pair])

    def set_data_correlations(self):
        # Get fields
        for map_symbol in self.map_symbols:
            self.fields[map_symbol] = nmt.NmtField(self.masks[map_symbol], [self.processed_maps[map_symbol]])

        # Get all correlations
        for correlation_symbol in self.all_correlation_symbols:
            map_symbol_a = correlation_symbol[0]
            map_symbol_b = correlation_symbol[1]
            self.data_correlations[correlation_symbol], self.workspaces[correlation_symbol] = compute_master(
                self.fields[map_symbol_a], self.fields[map_symbol_b], self.binning)

        # Decouple noise curves
        for correlation_symbol in self.noise_curves.keys():
            if isinstance(self.noise_curves[correlation_symbol], np.ndarray):
                self.noise_decoupled[correlation_symbol] = self.workspaces[correlation_symbol].decouple_cell(
                    self.workspaces[correlation_symbol].couple_cell([self.noise_curves[correlation_symbol]]))[0]

    def set_theory_correlations(self):
        cosmology = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96,
                                  matter_power_spectrum=self.cosmology_matter_power_spectrum)

        bias_arr = self.bias * np.ones(len(self.z_arr))
        if self.scale_bias:
            bias_arr = bias_arr / ccl.growth_factor(cosmology, 1. / (1. + self.z_arr))

        tracers_dict = {
            'g': ccl.NumberCountsTracer(cosmology, has_rsd=False, dndz=(self.z_arr, self.n_arr),
                                        bias=(self.z_arr, bias_arr)),
            'k': ccl.CMBLensingTracer(cosmology, 1091),
            't': ISWTracer(cosmology, z_max=6., n_chi=1024),
        }

        for correlation_symbol in self.all_correlation_symbols:
            tracer_symbol_a = correlation_symbol[0]
            tracer_symbol_b = correlation_symbol[1]
            correlation_symbol = tracer_symbol_a + tracer_symbol_b
            self.theory_correlations[correlation_symbol] = ccl.angular_cl(cosmology, tracers_dict[tracer_symbol_a],
                                                                          tracers_dict[tracer_symbol_b], self.l_arr)

    def set_binning(self):
        # n_ells: number of bins, for inference covariance indexing
        if self.ell_lengths:
            ell_ini = [int(self.l_min + np.sum(self.ell_lengths[:i])) for i in range(len(self.ell_lengths))]
            ell_end = [int(self.l_min + np.sum(self.ell_lengths[:i + 1])) for i in range(len(self.ell_lengths))]
            self.binning = nmt.NmtBin.from_edges(ell_ini, ell_end)
            self.n_ells = len(self.ell_lengths)
        else:
            self.binning = nmt.NmtBin.from_lmax_linear(self.l_max, self.ells_per_bin)
            self.n_ells = int((self.l_max - 2) / self.ells_per_bin)

        # Create dense array of ells, for theoretical power spectra
        self.l_arr = np.arange(3 * self.nside)

    def set_maps(self):
        set_map_functions = {
            'LoTSS_DR1': self.set_lotss_dr1_maps,
            'NVSS': self.set_nvss_maps,
            'KiDS_QSO': self.set_kids_qso_maps,
        }

        if 'g' in self.map_symbols:
            set_map_functions[self.lss_survey]()
            self.processed_maps['g'] = get_overdensity_map(self.original_maps['g'], self.masks['g'],
                                                           self.noise_weight_maps.get('g'))
            self.noise_curves['gg'] = np.full(3 * self.nside, get_shot_noise(self.original_maps['g'], self.masks['g']))

        if 'k' in self.map_symbols:
            self.original_maps['k'], self.masks['k'] = get_cmb_lensing_map(self.nside)
            self.processed_maps['k'] = self.original_maps['k']
            self.noise_curves['kk'] = get_cmb_lensing_noise(self.nside)

        if 't' in self.map_symbols:
            self.original_maps['t'], self.masks['t'] = get_cmb_temperature_map(nside=self.nside)
            self.processed_maps['t'] = self.original_maps['t']
            pass

        self.are_maps_ready = True

    def set_lotss_dr1_maps(self):
        self.data['g'] = get_lotss_hetdex_data()
        self.original_maps['g'], self.masks['g'], self.noise_maps['g'] = get_lotss_hetdex_map(self.data['g'],
                                                                                              nside=self.nside)
        self.noise_weight_maps['g'] = get_lotss_noise_weight_map(self.noise_maps['g'], self.masks['g'],
                                                                 self.lotss_flux_min_cut, self.nside)

    def set_nvss_maps(self):
        self.original_maps['g'], self.masks['g'] = get_nvss_map(nside=self.nside)

    def set_kids_qso_maps(self):
        self.data['g'] = get_kids_qsos()
        self.original_maps['g'], self.masks['g'] = get_kids_qso_map(self.data['g'], self.nside)


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

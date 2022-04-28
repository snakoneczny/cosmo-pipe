import json
import os
from collections import defaultdict
from random import random, sample

import emcee
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
from copy import deepcopy
import zeus
import h5py
from scipy.interpolate import interp1d

from env_config import PROJECT_PATH
from data_lotss import get_lotss_redshift_distribution
from experiment import Experiment
from utils import struct, decouple_correlation


def compare_biases(experiments, data_name, x_scale='log', x_max=None, y_max=None):
    for experiment_name, experiment_label in experiments:
        mcmc_folder_path = os.path.join(PROJECT_PATH, 'outputs/MCMC/{}/{}'.format(data_name, experiment_name))
        mcmc_filepath = os.path.join(mcmc_folder_path, '{}.config.json'.format(experiment_name))
        with open(mcmc_filepath) as file:
            config = json.load(file)

        if 'mcmc_engine' in config and config['mcmc_engine'] == 'zeus':
            samples, _, _, _, _ = get_zeus_samples(experiment_name, mcmc_folder_path)
        else:
            _, samples, _, _, _, _ = get_emcee_samples(experiment_name, mcmc_folder_path, config)

        # Final estimate
        best_fit_params = {}
        best_fit_params_min = {}
        best_fit_params_max = {}
        labels = config['to_infere']
        for i in range(len(labels)):
            mcmc = np.percentile(samples[:, i], [16, 50, 84])
            best_fit_params[labels[i]] = mcmc[1]
            best_fit_params_min[labels[i]] = mcmc[0]
            best_fit_params_max[labels[i]] = mcmc[2]

        config_mean = struct(**config)
        config_min = struct(**config)
        config_max = struct(**config)
        config_mean.__dict__.update(best_fit_params)
        config_min.__dict__.update(best_fit_params_min)
        config_max.__dict__.update(best_fit_params_max)

        experiment = Experiment(config_mean, set_data=False, set_maps=False)
        z_arr, _ = experiment.get_redshift_dist_function(config=config_mean, normalize=False)
        if x_max:
            z_arr = z_arr[z_arr < x_max]
        bias_arr = experiment.get_bias(z_arr, config=config_mean)
        bias_arr_min = experiment.get_bias(z_arr, config=config_min)
        bias_arr_max = experiment.get_bias(z_arr, config=config_max)

        if x_scale == 'log':
            z_arr = np.log(z_arr + 1)
        plt.plot(z_arr, bias_arr, label=experiment_label)
        plt.fill_between(z_arr, bias_arr_min, bias_arr_max, alpha=0.2)

    plt.legend(loc='upper left')
    plt.xlabel('log(1 + z)' if x_scale == 'log' else 'z')
    plt.ylabel('$b_g(z)$')
    plt.ylim((None, y_max))
    plt.show()


# TODO: refactor!
def compare_redshifts(experiments, data_name):
    for experiment_name, experiment_label in experiments:
        mcmc_folder_path = os.path.join(PROJECT_PATH, 'outputs/MCMC/{}/{}'.format(data_name, experiment_name))
        mcmc_filepath = os.path.join(mcmc_folder_path, '{}.config.json'.format(experiment_name))
        with open(mcmc_filepath) as file:
            config = json.load(file)

        labels = config['to_infere']
        n_walkers = config['n_walkers']

        backend_reader = emcee.backends.HDFBackend(os.path.join(mcmc_folder_path, '{}.h5'.format(experiment_name)))
        emcee_sampler = emcee.EnsembleSampler(n_walkers, len(labels), None, backend=backend_reader)

        tau = emcee_sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        samples = emcee_sampler.get_chain(discard=burnin, flat=True, thin=thin)

        # Final estimate
        best_fit_params = {}
        for i in range(len(labels)):
            mcmc = np.percentile(samples[:, i], [16, 50, 84])
            best_fit_params[labels[i]] = mcmc[1]

        best_fit_config = deepcopy(config)
        best_fit_config.update(best_fit_params)
        best_fit_config = struct(**best_fit_config)
        experiment = Experiment(best_fit_config, set_data=False, set_maps=False)

        z_arr, n_arr = experiment.get_redshift_dist_function(z_max=6, normalize=True)
        bias_arr = experiment.get_bias(z_arr)
        n_arr *= bias_arr
        # z_arr = np.log(z_arr + 1)

        # TODO: subplots for log
        plt.plot(z_arr, n_arr, label=experiment_label)

    plt.axhline(y=0, color='gray', linestyle='-')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('b * dN/dz')
    plt.show()


def show_mcmc_report(experiment_name, data_name, quick=False):
    mcmc_folder_path = os.path.join(PROJECT_PATH, 'outputs/MCMC/{}/{}'.format(data_name, experiment_name))
    mcmc_filepath = os.path.join(mcmc_folder_path, '{}.config.json'.format(experiment_name))
    with open(mcmc_filepath) as file:
        config = json.load(file)

    if config['mcmc_engine'] == 'emcee':
        emcee_sampler, samples, log_prob_samples, tau_arr, burnin, thin = get_emcee_samples(
            experiment_name, mcmc_folder_path, config)
    elif config['mcmc_engine'] == 'zeus':
        samples, log_prob_samples, tau_arr, burnin, thin = get_zeus_samples(experiment_name, mcmc_folder_path)

    # Final estimate
    best_fit_params = {}
    labels = config['to_infere']
    for i in range(len(labels)):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        best_fit_params[labels[i]] = mcmc[1]
        q = np.diff(mcmc)
        print('{} = {:.3f} (+{:.3f}, -{:.3f})'.format(labels[i], mcmc[1], q[0], q[1]))

    # Sigmas and chi-squared
    make_sigmas_report(config, best_fit_params)

    # Corner plot
    truths = [None] * len(labels)
    if 'sigma8' in labels:
        truths[labels.index('sigma8')] = 0.81
    if 'Omega_m' in labels:
        truths[labels.index('Omega_m')] = 0.31
    _, _ = zeus.cornerplot(samples, labels=labels)  # , truth=truths)
    # _ = corner(samples, labels=labels, truths=truths)
    plt.show()

    # Tau statistics
    plot_mean_tau(tau_arr)
    print('Burn-in: {}; thin: {}'.format(burnin, thin))
    if config['mcmc_engine'] == 'emcee':
        mean_acceptance_fraction = np.mean(emcee_sampler.acceptance_fraction) * 100
        print('Mean acceptance fraction: {:.1f}%'.format(mean_acceptance_fraction))

    # Samples history
    plot_samples_history(labels, samples, log_prob_samples)

    # Correlation, redshift and bias plots
    if not quick:
        make_param_plots(config, labels, samples)


def get_zeus_samples(experiment_name, mcmc_folder_path):
    burnin = 0.5
    thin = 2

    with h5py.File(os.path.join(mcmc_folder_path, '{}.h5'.format(experiment_name)), 'r') as hf:
        samples = np.copy(hf['samples'])
        log_prob_samples = np.copy(hf['logprob'])
    tau_arr = np.load(os.path.join(mcmc_folder_path, '{}.tau.npy'.format(experiment_name)))

    # Resize tau array which is not calculated every interation
    n_steps = samples.shape[0] / tau_arr.shape[0]
    x_arr = np.arange(tau_arr.shape[0]) * n_steps
    f = interp1d(x_arr, tau_arr)
    tau_arr = f(np.arange(x_arr[-1]))

    # burnin, thin, flatten
    random.seed(235742)
    samples = samples[samples.shape[0] // (1 / burnin):]
    samples = sample(samples, int(samples.shape[0] / thin))
    samples = samples.reshape(-1, samples.shape[-1])

    return samples, log_prob_samples, tau_arr, burnin, thin


def get_emcee_samples(experiment_name, mcmc_folder_path, config, burnin=None, thin=None):
    labels = config['to_infere']
    n_walkers = config['n_walkers']

    backend_reader = emcee.backends.HDFBackend(os.path.join(mcmc_folder_path, '{}.h5'.format(experiment_name)))
    tau_arr = np.load(os.path.join(mcmc_folder_path, '{}.tau.npy'.format(experiment_name)))
    emcee_sampler = emcee.EnsembleSampler(n_walkers, len(labels), None, backend=backend_reader)

    tau = emcee_sampler.get_autocorr_time(tol=0)
    burnin = int(2 * np.max(tau)) if burnin is None else burnin
    thin = int(0.5 * np.min(tau)) if thin is None else thin
    samples = emcee_sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = emcee_sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    # log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

    return emcee_sampler, samples, log_prob_samples, tau_arr, burnin, thin


def make_sigmas_report(config, best_fit_params):
    best_fit_config = deepcopy(config)
    best_fit_config.update(best_fit_params)
    best_fit_config = struct(**best_fit_config)
    best_fit_config.read_correlations_flag = False
    best_fit_config.read_covariance_flag = True
    experiment = Experiment(best_fit_config, set_data=True, set_maps=True)
    experiment.set_correlations(with_covariance=True)
    experiment.print_correlation_statistics()


def make_param_plots(config, arg_names, samples):
    # Create experiment based on config, but only read correlations
    config = struct(**config)
    config.read_correlations_flag = False
    config.read_covariance_flag = True
    experiment = Experiment(config, set_data=True, set_maps=True)
    experiment.set_correlations(with_covariance=True)

    # Iterate samples
    redshift_functions_store = defaultdict(list)
    correlations_store = dict([(correlation_symbol, []) for correlation_symbol in experiment.correlation_symbols])
    bias_arr_store = []
    inds = np.random.randint(len(samples), size=100)
    for ind in tqdm_notebook(inds):
        # Update data params
        sample = samples[ind]
        to_update = dict(zip(arg_names, sample))
        config.__dict__.update(to_update)

        # Update cosmo parameters
        cosmology_params = deepcopy(experiment.cosmology_params)
        cosmo_params = list(experiment.cosmology_params.keys()) + ['Omega_m']
        param_names = [param_name for param_name in arg_names if param_name in cosmo_params]
        for param_name in param_names:
            if param_name == 'Omega_m':
                baryon_fraction = 0.05 / 0.3
                Omega_m = to_update['Omega_m']
                cosmology_params['Omega_c'] = Omega_m * (1 - baryon_fraction)
                cosmology_params['Omega_b'] = Omega_m * baryon_fraction
            else:
                cosmology_params[param_name] = to_update[param_name]

        # TODO: make sure later that cosmology is taken into account correctly
        # Add correlation function to samples stores
        _, _, correlations_dict = experiment.get_theory_correlations(config, experiment.correlation_symbols,
                                                                     cosmology_params=cosmology_params)

        # Decoupling
        for correlation_symbol in experiment.correlation_symbols:
            correlations_dict[correlation_symbol] = decouple_correlation(experiment.workspaces[correlation_symbol],
                                                                         correlations_dict[correlation_symbol])
            # TODO: A_sn should modify data instead?
            if correlation_symbol == 'gg' and 'A_sn' in arg_names:
                correlations_dict[correlation_symbol] += (config.A_sn - 1) * experiment.noise_decoupled['gg']

        # Store it
        for correlation_symbol in correlations_dict:
            correlations_store[correlation_symbol].append(correlations_dict[correlation_symbol])

        # Store redshift distribution
        for redshift_to_fit in experiment.config.redshifts_to_fit:
            normalize = False if redshift_to_fit == 'tomographer' else True
            z_arr, n_arr = get_lotss_redshift_distribution(config=config, normalize=normalize)
            if redshift_to_fit == 'tomographer' and config.fit_bias_to_tomo:
                bias_arr = experiment.get_bias(z_arr, experiment.cosmology, config)
                n_arr *= bias_arr
            redshift_functions_store[redshift_to_fit].append(n_arr)

        # Store bias function
        z_arr, _ = get_lotss_redshift_distribution(config=config, normalize=False)
        bias_arr_store.append(experiment.get_bias(z_arr, experiment.cosmology, config))

    # Plot correlations
    for correlation_symbol in experiment.correlation_symbols:

        # Theory
        ell_arr = experiment.binnings[correlation_symbol].get_effective_ells()
        for correlation in correlations_store[correlation_symbol]:
            plt.plot(ell_arr, correlation, 'C1', alpha=0.02)

        # Data
        noise = experiment.noise_decoupled[correlation_symbol]
        correlation_dict = experiment.data_correlations
        data_to_plot = correlation_dict[correlation_symbol] - noise
        y_err = experiment.errors[experiment.config.error_method][correlation_symbol]
        plt.errorbar(ell_arr, data_to_plot, yerr=y_err, fmt='ob', label='data', markersize=2)
        if correlation_symbol == 'gg':
            plt.plot(ell_arr, noise, color='grey', marker='o', label='noise', markersize=2)
        # ell range lines
        l_range = experiment.config.l_range[correlation_symbol]
        plt.axvline(l_range[0], label='ell range', color='green')
        plt.axvline(l_range[1], color='green')

        plt.xlim(xmin=2)
        if correlation_symbol == 'gg':
            plt.ylim(ymin=1e-8, ymax=1e-5)
        elif correlation_symbol == 'gk':
            plt.ylim(ymin=1e-9)
        plt.yscale('log')
        plt.xlabel('$\\ell$', fontsize=16)
        plt.ylabel('$C_\\ell^{{{}}}$'.format(correlation_symbol), fontsize=16)
        plt.legend(loc='upper right', ncol=2, labelspacing=0.005)
        plt.grid()
        plt.show()

    # Plot redshift
    for i, (redshift_to_fit, redshift_function_arr) in enumerate(redshift_functions_store.items()):
        # TODO: subplots for log
        for n_arr in redshift_function_arr:
            plt.plot(z_arr, n_arr, 'C1', alpha=0.02)

        plt.errorbar(experiment.dz_to_fit[i], experiment.dn_dz_to_fit[i], experiment.dn_dz_err_to_fit[i], fmt='b.',
                     label=redshift_to_fit)
        plt.axhline(y=0, color='gray', linestyle='-')

        plt.legend()
        plt.xlabel('z')
        plt.ylabel('dN/dz')
        plt.show()

    # Plot bias
    for bias_arr in bias_arr_store:
        plt.plot(z_arr, bias_arr, 'C1', alpha=0.02)
        plt.xlabel('z')
        plt.ylabel('b')
    plt.show()


def plot_mean_tau(autocorr_time_arr):
    n = np.arange(1, len(autocorr_time_arr) + 1)
    plt.plot(n, n / 50.0, '--k')
    plt.plot(n, autocorr_time_arr)
    plt.xlabel('number of steps')
    plt.ylabel(r'mean $\hat{\tau}$')
    plt.show()


def plot_samples_history(labels, samples, log_prob_samples):
    fig, axes = plt.subplots(len(labels) + 1, sharex='all')
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, i], 'k', alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.set_xlim(0, samples.shape[0])
    ax = axes[-1]
    ax.plot(log_prob_samples, 'k', alpha=0.3)
    ax.set_ylabel('log prob')
    ax.set_xlabel('step number')
    ax.set_xlim(0, log_prob_samples.shape[0])
    plt.show()

import os
from collections import defaultdict
from random import random, sample
import logging
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook
from copy import deepcopy
import h5py
from scipy.interpolate import interp1d
import emcee
import zeus

from env_config import PROJECT_PATH
from experiment import Experiment
from utils import struct, get_config, decouple_correlation
from bias import get_sherwin_qso_bias


def print_lotss_constraints_table(rows, data_part, bias_models):
    df = pd.DataFrame()

    for row in rows:
        correlations = row[0]
        redshifts = row[1]
        cosmo_params = row[2]

        for bias_model in bias_models:
            for with_A_sn in [False, True]:
                redshift_part = '_'.join(redshifts)

                param_part = 'z_sfg_a_r'
                if 'tomographer' in redshifts:
                    param_part += '_n'
                if with_A_sn:
                    param_part = 'A_sn_' + param_part

                if bias_model == 'constant':
                    param_part = 'b_g_' + param_part
                elif bias_model == 'scaled':
                    param_part = 'b_g_scaled_' + param_part
                elif bias_model == 'quadratic_limited':
                    param_part = 'b_a_b_b_' + param_part
                elif bias_model == 'quadratic':
                    param_part = 'b_0_b_1_b_2_' + param_part

                if cosmo_params is not None and len(cosmo_params) > 0:
                    param_part = '_'.join(cosmo_params) + '_' + param_part


                experiment_name = '{}__{}_ell-52-502__redshift_power-law_{}__bias_{}__emcee_{}'.format(
                    data_part, '-'.join(correlations), redshift_part, bias_model, param_part)

                config, samples, _, _ = get_samples(experiment_name, data_name='LoTSS_DR2', print_stats=False)
                if config:
                    labels = config['to_infere']

                    for i, label in enumerate(labels):
                        mcmc = np.percentile(samples[:, i], [16, 50, 84])
                        q = np.diff(mcmc)

                        if label in ['b_g', 'b_g_scaled', 'b_a', 'b_b', 'b_0', 'b_1', 'b_2', 'A_sn', 'sigma8']:
                            row_key = ' + '.join(correlations)
                            if 'deep-fields' in redshifts:
                                row_key += ' + DF'
                            if 'tomographer' in redshifts:
                                row_key += ' + tomo'
                            if with_A_sn:
                                row_key += ' + A_sn'
                            if cosmo_params is not None and len(cosmo_params) > 0:
                                row_key += (' + ' + ' + '.join(cosmo_params))
                            col_key = '{} {}'.format(bias_model, label)

                            df.loc[row_key, col_key] = '${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$'.format(mcmc[1], q[1], q[0])

    display(df)
    print(df.to_latex(escape=False, na_rep=''))


def compare_biases(experiments, data_name, x_scale='log', x_max=None, y_max=None, add_qsos=False):
    n_exp = len(experiments)
    alpha = 1.0 / n_exp

    plt.figure()
    for experiment_label, experiment_name in experiments:
        config, samples, _, _ = get_samples(experiment_name, data_name, print_stats=False)
        arg_names = config['to_infere']

        # Create experiment based on config, only to use the get_bias function
        config = struct(**config)
        experiment = Experiment(config, set_data=False, set_maps=False)

        # Get z array
        z_arr, _ = experiment.get_redshift_dist_function(config=config, normalize=False)
        if x_max:
            z_arr = z_arr[z_arr < x_max]

        # Iterate samples
        bias_arr_store = []
        inds = np.random.randint(len(samples), size=1000)
        for ind in inds:
            # Update data params
            sample = samples[ind]
            to_update = dict(zip(arg_names, sample))
            config.__dict__.update(to_update)

            # Store bias function
            bias_arr_store.append(experiment.get_bias(z_arr, experiment.cosmology, config))
        bias_arr_store = np.array(bias_arr_store)

        bias_arr_mean, bias_arr_min, bias_arr_max = [], [], []
        for i in range(len(z_arr)):
            min, mean, max = np.percentile(bias_arr_store[:, i], [16, 50, 84])
            bias_arr_mean.append(mean)
            bias_arr_min.append(min)
            bias_arr_max.append(max)

        if x_scale == 'log':
            z_arr = np.log(z_arr + 1)

        plt.plot(z_arr, bias_arr_mean, label=experiment_label)
        plt.fill_between(z_arr, bias_arr_min, bias_arr_max, alpha=alpha)

    if add_qsos:
        z_arr, b_arr, b_err = get_sherwin_qso_bias()
        plt.errorbar(z_arr, b_arr, yerr=b_err, marker='s', color='k', linestyle='', markersize=3,
                     label='Sherwin et al. 2012')

    plt.legend(loc='upper left')
    plt.xlabel('log(1 + z)' if x_scale == 'log' else 'z')
    plt.ylabel('$b_g(z)$')
    plt.ylim((None, y_max))
    plt.show()


def compare_redshifts(experiments, data_name):
    for experiment_label, experiment_name in experiments:
        config, samples, _, _ = get_samples(experiment_name, data_name, print_stats=False)
        labels = config['to_infere']

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

        plt.plot(z_arr, n_arr, label=experiment_label)

    plt.axhline(y=0, color='gray', linestyle='-')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('b * dN/dz')
    plt.show()


def show_mcmc_report(experiment_name, data_name, quick=False):
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'ERROR'))
    config, samples, log_prob_samples, tau_arr = get_samples(experiment_name, data_name, print_stats=True)

    # Tau statistics
    plot_mean_tau(tau_arr)

    # Final estimate
    best_fit_params = {}
    labels = config['to_infere']
    print('------------------------------')
    for i in range(len(labels)):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        best_fit_params[labels[i]] = mcmc[1]
        q = np.diff(mcmc)
        print('{} = {:.2f} (+{:.2f}, -{:.2f})'.format(labels[i], mcmc[1], q[1], q[0]))
    print('------------------------------')

    # Sigmas and chi-squared
    make_sigmas_report(config, best_fit_params)

    # Zeus traingle plot
    if len(labels) > 1:
        # truths = [None] * len(labels)
        # if 'sigma8' in labels:
        #     truths[labels.index('sigma8')] = 0.811
        # if 'Omega_m' in labels:
        #     truths[labels.index('Omega_m')] = 0.315

        samples_size = samples.shape[0]
        n_smpl = 1000
        samples_to_traingle = samples[np.random.randint(samples_size, size=n_smpl), :] if samples_size > n_smpl else samples

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, _ = zeus.cornerplot(samples_to_traingle, labels=labels)  # , truth=truths)
            plt.show()

    if 'sigma8' in labels:
        # pp_dict = {
        #     'Omega_m': '\Omega_m',
        #     'sigma8': '\sigma_8',
        #     'b_g_scaled': 'b_g',
        # }
        # pp_labels = [lbl if lbl not in pp_dict else pp_dict[lbl] for lbl in labels]
        sns.kdeplot(samples[:, labels.index('sigma8')], bw=0.5, label='LoTSS DR2 x CMB')
        plt.xlabel('$\sigma_8$')
        plt.ylabel('probability')
        plt.axvline(x=0.811, ymin=0, color='r', label='Planck')
        plt.legend()
        plt.show()

    # Samples history
    plot_samples_history(labels, samples, log_prob_samples)

    # Correlation, redshift and bias plots
    if not quick:
        make_param_plots(config, labels, samples)


def get_samples(experiment_name, data_name, print_stats=False):
    mcmc_folder_path = os.path.join(PROJECT_PATH, 'outputs/MCMC/{}/{}'.format(data_name, experiment_name))
    if not os.path.exists(mcmc_folder_path):
        return None, None, None, None
    config = get_config(data_name, experiment_name)

    if 'mcmc_engine' in config and config['mcmc_engine'] == 'zeus':
        samples, log_prob_samples, tau_arr, burnin, thin = get_zeus_samples(experiment_name, mcmc_folder_path)

    else:  # emcee
        emcee_sampler, samples, log_prob_samples, tau_arr, burnin, thin = get_emcee_samples(
            experiment_name, mcmc_folder_path, config)
        if print_stats:
            mean_acceptance_fraction = np.mean(emcee_sampler.acceptance_fraction) * 100
            if print_stats:
                print('Mean acceptance fraction: {:.1f}%'.format(mean_acceptance_fraction))

    if print_stats:
        print('Final chain length: {}; burn-in: {}; thin: {}'.format(samples.shape[0], burnin, thin))

    return config, samples, log_prob_samples, tau_arr


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
    experiment.set_correlations()
    experiment.print_correlation_statistics()


def make_param_plots(config, arg_names, samples):
    # Create experiment based on config, but only read correlations
    config = struct(**config)
    config.read_correlations_flag = False
    config.read_covariance_flag = True
    experiment = Experiment(config, set_data=True, set_maps=True)
    experiment.set_correlations()

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
        cosmology_params = experiment.get_updated_cosmology_parameters(to_update)

        # Add correlation function to samples stores
        _, _, correlations_dict = experiment.get_theory_correlations(config, experiment.correlation_symbols,
                                                                     cosmology_params=cosmology_params)

        # Decoupling
        for correlation_symbol in experiment.correlation_symbols:
            correlations_dict[correlation_symbol] = decouple_correlation(experiment.workspaces[correlation_symbol],
                                                                         correlations_dict[correlation_symbol])
            if correlation_symbol == 'gg' and 'A_sn' in arg_names:
                correlations_dict[correlation_symbol] += (config.A_sn - 1) * experiment.noise_decoupled['gg']

        # Store it
        for correlation_symbol in correlations_dict:
            correlations_store[correlation_symbol].append(correlations_dict[correlation_symbol])

        # Store redshift distribution
        for redshift_to_fit in experiment.config.redshifts_to_fit:
            normalize = False if redshift_to_fit == 'tomographer' else True
            z_arr, n_arr = experiment.get_redshift_dist_function(config=config, normalize=normalize)
            if redshift_to_fit == 'tomographer':
                bias_arr = experiment.get_bias(z_arr, experiment.cosmology, config)
                n_arr *= bias_arr
            redshift_functions_store[redshift_to_fit].append(n_arr)

        # Store bias function
        z_arr, _ = experiment.get_redshift_dist_function(config=config, normalize=False)
        bias_arr_store.append(experiment.get_bias(z_arr, experiment.cosmology, config))

    # Plot correlations
    for correlation_symbol in experiment.correlation_symbols:
        plt.figure()

        # Theory
        ell_arr = experiment.binnings[correlation_symbol].get_effective_ells()
        ell_dense = np.arange(ell_arr[0], ell_arr[-1], 1)
        for correlation in correlations_store[correlation_symbol]:
            f = interp1d(ell_arr, correlation, kind='cubic')
            correlation_interpolated = f(ell_dense)
            plt.plot(ell_dense, correlation_interpolated, 'C1', alpha=0.02)

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

        # TODO: different limits for C_qq and C_gg
        if correlation_symbol == 'gg':
            plt.ylim(ymin=1e-8, ymax=1e-3)
        elif correlation_symbol == 'gk':
            plt.ylim(ymin=1e-9, ymax=1e-6)

        rename_dict = {'g': 'q'} if experiment.config.lss_survey_name == 'KiDS_QSO' else None
        if rename_dict:
            for key in rename_dict.keys():
                correlation_symbol = correlation_symbol.replace(key, rename_dict[key])

        plt.xlim(xmin=2)
        plt.yscale('log')
        plt.xlabel('$\\ell$', fontsize=16)
        plt.ylabel('$C_\\ell^{{{}}}$'.format(correlation_symbol), fontsize=16)
        plt.legend(loc='upper right', ncol=2, labelspacing=0.005)
        plt.grid()
        plt.show()

    # Plot redshift
    for i, (redshift_to_fit, redshift_function_arr) in enumerate(redshift_functions_store.items()):
        plt.figure()

        for n_arr in redshift_function_arr:
            plt.plot(z_arr, n_arr, 'C1', alpha=0.02)

        plt.errorbar(experiment.dz_to_fit[i], experiment.dn_dz_to_fit[i], experiment.dn_dz_err_to_fit[i], fmt='b.',
                     label=redshift_to_fit)
        plt.axhline(y=0, color='gray', linestyle='-')

        plt.legend()
        plt.xlabel('z')
        if redshift_to_fit == 'tomographer':
            plt.ylabel('b * dN/dz')
        else:
            plt.ylabel('dN/dz')
        plt.show()

    # Plot bias
    plt.figure()
    for bias_arr in bias_arr_store:
        plt.plot(z_arr, bias_arr, 'C1', alpha=0.02)
        plt.xlabel('z')
        plt.ylabel('b')
    plt.show()


def plot_mean_tau(autocorr_time_arr):
    n = np.arange(1, len(autocorr_time_arr) + 1)
    plt.figure()
    plt.plot(n, n / 50, '--k')
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

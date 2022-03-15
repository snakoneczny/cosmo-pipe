import json
import os

import emcee
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from corner import corner
import pyccl as ccl
from tqdm import tqdm_notebook

from env_config import PROJECT_PATH, DATA_PATH
from data_lotss import get_lotss_redshift_distribution
from experiment import Experiment
from utils import struct, decouple_correlation


def show_mcmc_report(experiment_name, data_name, burnin=None, thin=None):
    mcmc_folder_path = os.path.join(PROJECT_PATH, 'outputs/MCMC/{}/{}'.format(data_name, experiment_name))
    mcmc_filepath = os.path.join(mcmc_folder_path, '{}.config.json'.format(experiment_name))
    with open(mcmc_filepath) as file:
        config = json.load(file)
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

    # Final estimates
    for i in range(len(labels)):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print('{} = {:.3f} (+{:.3f}, -{:.3f})'.format(labels[i], mcmc[1], q[0], q[1]))

    # Corner plot
    corner(samples, labels=labels)  # , truths=[m_true, b_true, np.log(f_true)])
    plt.show()

    # Correlation and redshift plots
    make_param_plots(config, labels, samples)

    # Tau plot
    print('Mean acceptance fraction: {}'.format(np.mean(emcee_sampler.acceptance_fraction)))
    # print('Number of iterations: {}'.format(len(tau_arr)))
    print('burn-in: {0}'.format(burnin))
    print('thin: {0}'.format(thin))
    plot_mean_tau(tau_arr)

    # Samples history
    plot_samples_history(labels, samples, log_prob_samples)


def make_param_plots(config, arg_names, samples):
    # Create experiment based on config, but only read correlations
    config = struct(**config)
    config.read_correlations_flag = False
    config.read_covariance_flag = True
    experiment = Experiment(config, set_data=True, set_maps=True)
    experiment.set_correlations(with_covariance=True)

    # Iterate samples
    redshift_functions = []
    correlations = dict([(correlation_symbol, []) for correlation_symbol in experiment.correlation_symbols])
    inds = np.random.randint(len(samples), size=200)
    for ind in tqdm_notebook(inds):
        sample = samples[ind]
        to_update = dict(zip(arg_names, sample))
        config.__dict__.update(to_update)

        # TODO: make sure later that cosmology is taken into account correctly
        # Add correlation function to samples stores
        _, _, correlations_dict = experiment.get_theory_correlations(config, experiment.cosmology_params,
                                                                     experiment.correlation_symbols)

        # Decoupling
        for correlation_symbol in experiment.correlation_symbols:
            correlations_dict[correlation_symbol] = decouple_correlation(experiment.workspaces[correlation_symbol],
                                                                         correlations_dict[correlation_symbol])
            # TODO: A_sn should modify data instead?
            if correlation_symbol == 'gg' and 'A_sn' in arg_names:
                correlations_dict[correlation_symbol] += (config.A_sn - 1) * experiment.noise_decoupled['gg']

        # Store it
        for correlation_symbol in correlations_dict:
            correlations[correlation_symbol].append(correlations_dict[correlation_symbol])

        # Store redshift distribution
        if 'z_sfg' in arg_names:
            z_arr, n_arr = get_lotss_redshift_distribution(
                z_sfg=getattr(config, 'z_sfg', None), a=getattr(config, 'a', None), r=getattr(config, 'r', None),
                n=getattr(config, 'n', None), z_tail=getattr(config, 'z_tail', None), flux_cut=config.flux_min_cut,
                model=config.dn_dz_model, normalize=False)
            redshift_functions.append(n_arr)

    # Plot correlations
    for correlation_symbol in experiment.correlation_symbols:

        # Theory
        ell_arr = experiment.binnings[correlation_symbol].get_effective_ells()
        for correlation in correlations[correlation_symbol]:
            plt.plot(ell_arr, correlation, 'C1', alpha=0.01)

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
    if len(redshift_functions) > 0:
        for n_arr in redshift_functions:
            plt.plot(z_arr, n_arr, 'C1', alpha=0.01)

        tomographer_file = os.path.join(DATA_PATH, 'LoTSS/DR2/tomographer/{}mJy_{}SNR_srl_catalog_{}.csv'.format(
            config.flux_min_cut, config.signal_to_noise, config.lss_mask_name.split('_')[1]))
        tomographer = pd.read_csv(tomographer_file)
        z_arr = tomographer['z'][:-1]
        n_arr = tomographer['dNdz_b'][:-1]
        n_err_arr = tomographer['dNdz_b_err'][:-1]

        growth_factor = ccl.growth_factor(experiment.cosmology, 1. / (1. + z_arr))
        n_arr *= growth_factor
        n_err_arr *= growth_factor

        plt.errorbar(z_arr, n_arr, n_err_arr, fmt='b.', label='tomographer')
        plt.axhline(y=0, color='gray', linestyle='-')

        plt.legend()
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

import os
import math

import numpy as np
import pymaster as nmt
import pyccl as ccl
import emcee

from data_lotss import get_lotss_redshift_distribution


def run_emcee(sampler, position, max_iterations, autocorr_time_arr=None, reset=False, progress='notebook'):
    if autocorr_time_arr is None:
        autocorr_time_arr = []

    if reset:
        sampler.reset()

    for sample in sampler.sample(position, iterations=max_iterations, progress=progress):
        tau = sampler.get_autocorr_time(tol=0)
        autocorr_time_arr.append(np.mean(tau))

        if len(autocorr_time_arr) > 1:
            tau_change = np.abs(autocorr_time_arr[-2] - tau) / tau
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(tau_change < 0.01)

            if not sampler.iteration % 100:
                print('Iteration: {}, tau: {}, tau change: {}'.format(sampler.iteration, tau, tau_change))

            if converged:
                break

    return autocorr_time_arr


def init_emcee_sampler(p0_walkers, arg_names, log_prob_function, default_params, ells, data, icov, filename, pool=None):
    args = [arg_names, default_params, ells, data, icov]
    n_walkers = p0_walkers.shape[0]
    n_dim = p0_walkers.shape[1]
    file_path = os.path.join('../outputs/MCMC', filename)
    backend = emcee.backends.HDFBackend(file_path)
    backend.reset(n_walkers, n_dim)
    emcee_sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_function, backend=backend, args=args, pool=pool)
    return emcee_sampler


def get_walkers_starting_params(starting_params, n_walkers):
    p0 = np.array([starting_params[key][0] for key in starting_params])
    p0_scales = np.array([starting_params[key][1] for key in starting_params])
    n_dim = len(p0)
    p0_walkers = np.array([p0 + p0_scales * np.random.uniform(low=-1, high=1, size=n_dim) for i in range(n_walkers)])
    return p0_walkers


def get_log_prior(theta, arg_names):
    prior = 0
    if 'bias' in arg_names and theta[arg_names.index('bias')] < 0:
        prior = -np.inf
    if 'sigma8' in arg_names and theta[arg_names.index('sigma8')] < 1e-5:
        prior = -np.inf
    return prior


def get_log_prob(theta, arg_names, default_params, ells, data, icov):
    # Check the priors
    log_prior = get_log_prior(theta, arg_names)
    if not np.isfinite(log_prior):
        return -np.inf

    # Update default parameters with given parameters
    params = default_params.copy()
    for param_name in arg_names:
        params[param_name] = theta[arg_names.index(param_name)]

    z_arr, n_arr = get_lotss_redshift_distribution(z_tail=params['z_tail'])
    bias_arr = params['bias'] * np.ones(len(z_arr))

    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=params['sigma8'], n_s=0.96,
                          matter_power_spectrum='linear')
    tracer1 = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr))
    # TODO: check if tacer 1 doubled is enough
    tracer2 = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr))
    model = ccl.angular_cl(cosmo, tracer1, tracer2, ells)
    diff = data - model
    return log_prior - np.dot(diff, np.dot(icov, diff)) / 2.0


def get_chi_squared(data, theory, covariance):
    diff = data - theory
    cov_inv = np.linalg.inv(covariance)
    return diff.dot(cov_inv).dot(diff)


def get_correlation_matrix(covariance_matrix):
    correlation_matrix = covariance_matrix.copy()
    for i in range(covariance_matrix.shape[0]):
        for j in range(covariance_matrix.shape[1]):
            correlation_matrix[i, j] = covariance_matrix[i, j] / math.sqrt(
                covariance_matrix[i, i] * covariance_matrix[j, j])
    return correlation_matrix


def get_covariance_matrix(field_a1, field_a2, field_b1, field_b2, cl_theory_a1_b1, cl_theory_a1_b2, cl_theory_a2_b1,
                          cl_theory_a2_b2, workspace_a1_a2, workspace_b1_b2):
    covariance_workspace = nmt.NmtCovarianceWorkspace()
    covariance_workspace.compute_coupling_coefficients(field_a1, field_a2, field_b1, field_b2)

    covariance = nmt.gaussian_covariance(
        covariance_workspace,
        0, 0, 0, 0,
        [cl_theory_a1_b1],
        [cl_theory_a1_b2],
        [cl_theory_a2_b1],
        [cl_theory_a2_b2],
        workspace_a1_a2,
        wb=workspace_b1_b2
    )
    return covariance


def get_theory_correlations(l_arr, z_arr, n_arr, bias, scale_bias=False):
    # Omega_c = None, Omega_b = None, h = None, n_s = None,
    # sigma8 = None, A_s = None,
    # Omega_k = 0., Omega_g = None, Neff = 3.046, m_nu = 0., m_nu_type = None,
    # w0 = -1., wa = 0., T_CMB = None,
    # bcm_log10Mc = np.log10(1.2e14), bcm_etab = 0.5,
    # bcm_ks = 55., mu_0 = 0., sigma_0 = 0., z_mg = None, df_mg = None,
    # transfer_function = 'boltzmann_camb',
    # matter_power_spectrum = 'halofit',
    # baryons_power_spectrum = 'nobaryons',
    # mass_function = 'tinker10',
    # halo_concentration = 'duffy2008',
    # emulator_neutrinos = 'strict'
    cosmology = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96,
                              matter_power_spectrum='linear')
    bias_arr = bias * np.ones(len(z_arr))
    if scale_bias:
        bias_arr = bias_arr / ccl.growth_factor(cosmology, 1. / (1 + z_arr))
    number_counts_tracer = ccl.NumberCountsTracer(cosmology, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr))
    lensing_tracer = ccl.WeakLensingTracer(cosmology, dndz=(z_arr, n_arr))
    cmb_lensing_tracer = ccl.CMBLensingTracer(cosmology, 1091)
    cl_gg = ccl.angular_cl(cosmology, number_counts_tracer, number_counts_tracer, l_arr)
    cl_gk = ccl.angular_cl(cosmology, lensing_tracer, cmb_lensing_tracer, l_arr)
    cl_kk = ccl.angular_cl(cosmology, cmb_lensing_tracer, cmb_lensing_tracer, l_arr)
    return cl_gg, cl_gk, cl_kk


def get_data_correlations(map_counts, map_g, mask_g, map_cmb_k, mask_cmb_k, nside, l_max=None, with_shot_noise=True,
                          mask_aposize=1.0, ells_per_bandpower=4):
    # Get shot noise for discrete objects
    shot_noise = 0
    if with_shot_noise:
        shot_noise = get_shot_noise(map_counts, mask_g)
    # Apodize mask  # TODO: should the apodization influence any other mask usage?
    if mask_aposize:
        mask_g = nmt.mask_apodization(mask_g, mask_aposize, apotype='Smooth')
    # Get field
    field_g = nmt.NmtField(mask_g, [map_g])
    field_cmb_k = nmt.NmtField(mask_cmb_k, [map_cmb_k])
    # field_cmb_t = nmt.NmtField(mask_cmb, [map_cmb_t])
    # Initialize binning scheme
    if l_max:
        ells = np.arange(l_max, dtype='int32')
        weights = 1.0 / ells_per_bandpower * np.ones_like(ells)
        bandpower_indices = -1 + np.zeros_like(ells)
        i = 0
        while ells_per_bandpower * (i + 1) + 2 < l_max:
            bandpower_indices[ells_per_bandpower * i + 2:ells_per_bandpower * (i + 1) + 2] = i
            i += 1
        binning = nmt.NmtBin(nside=nside, bpws=bandpower_indices, ells=ells, weights=weights)
    else:
        binning = nmt.NmtBin.from_nside_linear(nside, ells_per_bandpower)

    # Get all correlations
    cl_coupled_gg, cl_decoupled_gg, workspace_gg = compute_master(field_g, field_g, binning)
    cl_coupled_gk, cl_decoupled_gk, workspace_gk = compute_master(field_g, field_cmb_k, binning)
    cl_coupled_kk, cl_decoupled_kk, workspace_kk = compute_master(field_cmb_k, field_cmb_k, binning)

    # Substract shot noise
    cl_coupled_gg = cl_coupled_gg - shot_noise
    cl_decoupled_gg = cl_decoupled_gg - shot_noise

    # Return correlation with substracted shot noise
    return field_g, field_cmb_k, cl_coupled_gg, cl_decoupled_gg, workspace_gg, cl_coupled_gk, cl_decoupled_gk, \
           workspace_gk, cl_coupled_kk, cl_decoupled_kk, workspace_kk, binning, shot_noise


def get_shot_noise(map, mask):
    sky_frac = np.sum(mask) / np.shape(mask)[0]
    n_obj = np.sum(map[np.nonzero(mask)])
    shot_noise = 4.0 * math.pi * sky_frac / n_obj
    return shot_noise


def compute_master(field_a, field_b, binning):
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field_a, field_b, binning)
    cl_coupled = nmt.compute_coupled_cell(field_a, field_b)
    cl_decoupled = workspace.decouple_cell(cl_coupled)
    return cl_coupled[0], cl_decoupled[0], workspace

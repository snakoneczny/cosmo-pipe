import math

import numpy as np

from data_lotss import get_lotss_hetdex_data, get_lotss_hetdex_map, get_lotss_noise_weight_map, \
    get_lotss_redshift_distribution
from data_cmb import get_cmb_lensing_map
from utils import get_overdensity_map
from cosmology import get_theory_correlations, get_data_correlations, get_covariance_matrix, \
    get_walkers_starting_params, init_emcee_sampler, get_log_prob, run_emcee

# Define parameters
continue_sampling = False
backend_filename = 'outputs/MCMC/gg_b-2-8_z-2-0.h5'
n_walkers = 36
max_iterations = 5000

bias = 2.8
z_tail = 2.0

starting_params = {'bias': (2.8, 0.5), 'sigma8': (0.83, 0.1)}
default_params = {'bias': 2.8, 'sigma8': 0.83, 'z_tail': 2.0}

nside = 512
lotss_flux_min_cut = 2  # mJy

# Get data
lotss_data = get_lotss_hetdex_data()
lotss_counts_map, lotss_mask, lotss_noise_map = get_lotss_hetdex_map(lotss_data, nside=nside)
cmb_lensing_map, cmb_lensing_mask = get_cmb_lensing_map(nside=nside, fwhm=math.radians(0.8))
lotss_noise_weight_map = get_lotss_noise_weight_map(lotss_noise_map, lotss_mask, lotss_flux_min_cut, nside)
lotss_overdensity_map = get_overdensity_map(lotss_counts_map, lotss_mask, lotss_noise_weight_map)

# Get theory correlations
z_arr, n_arr = get_lotss_redshift_distribution(z_tail=z_tail)
l_arr = np.arange(2, 3 * nside + 2)
gg_theory, gk_theory, kk_theory = get_theory_correlations(l_arr, z_arr, n_arr, bias, scale_bias=True)

# Get cross correlations
ells_per_bandpower = 50
l_max = None
mask_aposize = None
lotss_field, cmb_k_field, gg_coupled, gg_decoupled, gg_workspace, gk_coupled, gk_decoupled, gk_workspace, kk_coupled, \
kk_decoupled, kk_workspace, binning, shot_noise = get_data_correlations(
    lotss_counts_map, lotss_overdensity_map, lotss_mask, cmb_lensing_map, cmb_lensing_mask, nside,
    ells_per_bandpower=ells_per_bandpower, with_shot_noise=True, mask_aposize=mask_aposize, l_max=l_max)

# Get covariance matrix
covariance_gg_gg = get_covariance_matrix(lotss_field, lotss_field, lotss_field, lotss_field, gg_theory, gg_theory,
                                         gg_theory, gg_theory, gg_workspace, gg_workspace)
# covariance_gk_gk = None

# Run MCMC
n_ells = 10
icov_gg = np.linalg.inv(covariance_gg_gg)[:n_ells, :n_ells]
data_gg = gg_decoupled[:n_ells]
ells = binning.get_effective_ells()[:n_ells]

p0_walkers = get_walkers_starting_params(starting_params, n_walkers)
arg_names = list(starting_params.keys())

sampler = init_emcee_sampler(p0_walkers, arg_names, get_log_prob, default_params, ells, data_gg, icov_gg,
                             backend_filename, continue_sampling=continue_sampling)

tau_filename = backend_filename.replace('.h5', '_tau.npy')
tau_arr = run_emcee(sampler, p0_walkers, max_iterations, tau_filename=tau_filename, continue_sampling=continue_sampling)

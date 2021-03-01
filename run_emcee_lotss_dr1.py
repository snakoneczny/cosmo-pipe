from experiment import Experiment

config = {
    'correlation_symbols': ['gg', 'gk'],
    'l_max': 500,
    'ells_per_bin': 50,
    'lotss_flux_min_cut': 2,  # mJy
    'nside': 512,
    'z_tail': 2.0,
    'bias': 2.1,
    'scale_bias': True,
    'continue_sampling': False,
    'n_walkers': 32,
    'max_iterations': 5000,
    'starting_params': {'bias': (2.1, 0.5), 'sigma8': (0.83, 0.1)},
    'default_params': {'bias': 2.1, 'sigma8': 0.83, 'z_tail': 2.0},
    'experiment_tag': '',
}

experiment = Experiment(config)
experiment.run_emcee()

# # Define parameters
# continue_sampling = False
# backend_filename = 'outputs/MCMC/gg-gk_z-2-0_b-2-8_bias-inv-scaled.h5'
# n_walkers = 32
# max_iterations = 5000
#
# bias = 2.8
# z_tail = 2.0
#
# starting_params = {'bias': (2.8, 0.5), 'sigma8': (0.83, 0.1)}
# default_params = {'bias': 2.8, 'sigma8': 0.83, 'z_tail': 2.0}
#
# nside = 512
# lotss_flux_min_cut = 2  # mJy
#
# # Get data
# cmb_lensing_map, cmb_lensing_mask = get_cmb_lensing_map(nside=nside)
# cmb_lensing_spectra = get_cmb_lensing_noise_spectra(nside)
#
# lotss_data = get_lotss_hetdex_data()
# lotss_counts_map, lotss_mask, lotss_noise_map = get_lotss_hetdex_map(lotss_data, nside=nside)
# lotss_shot_noise = get_shot_noise(lotss_counts_map, lotss_mask)
#
# lotss_noise_weight_map = get_lotss_noise_weight_map(lotss_noise_map, lotss_mask, lotss_flux_min_cut, nside)
# lotss_overdensity_map = get_overdensity_map(lotss_counts_map, lotss_mask, lotss_noise_weight_map)
#
# l_arr = np.arange(3 * nside)
# ells_per_bandpower = 50
# binning = nmt.NmtBin.from_nside_linear(nside, nlb=ells_per_bandpower)
#
# # Get theory correlations
# z_arr, n_arr = get_lotss_redshift_distribution(z_tail=z_tail)
# gg_theory, gk_theory, kk_theory = get_theory_correlations(l_arr, z_arr, n_arr, bias, scale_bias=True)
#
# # Add noise "curves"
# gg_theory += lotss_shot_noise
#
# l_min = int(cmb_lensing_spectra['l'][0])
# l_max = min(l_arr[-1], cmb_lensing_spectra['l'].values[-1])
# kk_l_arr = np.arange(l_min, l_max + 1)
# kk_theory_2 = np.zeros(len(l_arr))
# kk_theory_2[kk_l_arr] = kk_theory[kk_l_arr]
# kk_theory_2[kk_l_arr] += cmb_lensing_spectra['nl'].values[:l_max-l_min+1]
# kk_theory = kk_theory_2
#
# # Get cross correlations
# ells_per_bandpower = 50
# lotss_field, cmb_k_field, gg_coupled, gg_decoupled, gg_workspace, gk_coupled, gk_decoupled, gk_workspace, kk_coupled, \
# kk_decoupled, kk_workspace = get_data_correlations(lotss_overdensity_map, lotss_mask, cmb_lensing_map, cmb_lensing_mask,
#                                                    binning)
#
# # Get covariance matrix
# covariance_gg_gg = get_covariance_matrix(lotss_field, lotss_field, lotss_field, lotss_field, gg_theory, gg_theory,
#                                          gg_theory, gg_theory, gg_workspace, gg_workspace)
# covariance_gg_gk = get_covariance_matrix(lotss_field, lotss_field, lotss_field, cmb_k_field, gg_theory, gk_theory,
#                                          gg_theory, gk_theory, gg_workspace, gk_workspace)
# covariance_gk_gk = get_covariance_matrix(lotss_field, cmb_k_field, lotss_field, cmb_k_field, gg_theory, gk_theory,
#                                          gk_theory, kk_theory, gk_workspace, gk_workspace)
#
# n_ells = 10
# top_row = np.concatenate((covariance_gg_gg[:n_ells, :n_ells], np.transpose(covariance_gg_gk[:n_ells, :n_ells])), axis=1)
# bottom_row = np.concatenate((covariance_gg_gk[:n_ells, :n_ells], covariance_gk_gk[:n_ells, :n_ells]), axis=1)
# covariance_gg_gk_full = np.concatenate((top_row, bottom_row), axis=0)
# icov = np.linalg.inv(covariance_gg_gk_full)
#
# # Run MCMC, subtract noise from data because theory spectra are created without noise during the sampling
# data = np.concatenate((gg_decoupled[:n_ells] - lotss_shot_noise, gk_decoupled[:n_ells]))
#
# p0_walkers = get_walkers_starting_params(starting_params, n_walkers)
# arg_names = list(starting_params.keys())
#
# get_log_prob_args = [arg_names, default_params, l_arr, data, icov, gg_workspace, gk_workspace]
# sampler = init_emcee_sampler(p0_walkers, get_log_prob, get_log_prob_args, backend_filename,
#                              continue_sampling=continue_sampling)
#
# tau_filename = backend_filename.replace('.h5', '_tau.npy')
# tau_arr = run_emcee(sampler, p0_walkers, max_iterations, tau_filename=tau_filename, continue_sampling=continue_sampling)

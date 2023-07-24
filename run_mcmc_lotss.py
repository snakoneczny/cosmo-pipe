import argparse
import copy

from utils import get_config
from experiment import Experiment

# Parameter order: minimum flux (mJy), minimum signal to noise, correlation symbols, redshifts, with A_sn, ell_max,
# matter power spectrum, cosmology params, is mock

# Cosmology
bias_models = ['scaled']
to_run = [
    (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 252, 502, 'halofit', ['sigma8'], False),
    (2.0, 5.0, ['gg', 'gk'], ['deep_fields'], True, 252, 502, 'halofit', ['sigma8'], False),

    (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 252, 502, 'linear', ['sigma8'], False),
    (2.0, 5.0, ['gg', 'gk'], ['deep_fields'], True, 252, 502, 'linear', ['sigma8'], False),

    (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], False, 152, 502, 'halofit', ['sigma8'], False),
    (2.0, 5.0, ['gg', 'gk'], ['deep_fields'], False, 152, 502, 'halofit', ['sigma8'], False),

    (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], False, 152, 502, 'linear', ['sigma8'], False),
    (2.0, 5.0, ['gg', 'gk'], ['deep_fields'], False, 152, 502, 'linear', ['sigma8'], False),

    (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 502, 802, 'halofit', ['sigma8'], False),
    (2.0, 5.0, ['gg', 'gk'], ['deep_fields'], True, 502, 802, 'halofit', ['sigma8'], False),

    (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 502, 802, 'linear', ['sigma8'], False),
    (2.0, 5.0, ['gg', 'gk'], ['deep_fields'], True, 502, 802, 'linear', ['sigma8'], False),
]

# C_gg & C_gk tests
# bias_models = ['constant', 'scaled']
# to_run = [
#     (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 252, 502, 'halofit', [], False),
#     (1.5, 7.5, ['gg'], ['deep_fields'], True, 252, 502, 'halofit', [], False),
#     (1.5, 7.5, ['gk'], ['deep_fields'], False, 252, 502, 'halofit', [], False),
# ]

# ell range and linear vs halofit tests
# bias_models = ['scaled', 'quadratic']
# to_run = [
#     (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 252, 502, 'halofit', [], False),
#     (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 502, 802, 'halofit', [], False),
#     (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 252, 502, 'linear', [], False),
#     (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 502, 802, 'linear', [], False),
# ]

# Data tests
# bias_models = ['scaled']
# to_run = [
#     (2.0, 5.0, ['gg', 'gk'], ['deep_fields'], True, 252, 502, 'halofit', [], False),
# ]

# C_gg ell range test
# bias_models = ['scaled']
# to_run = [
#     (1.5, 7.5, ['gg'], ['deep_fields'], False, 152, 502, 'halofit', [], False),
#     (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], False, 152, 502, 'halofit', [], False),
# ]

# Mock tests
# bias_models = ['scaled']
# to_run = [
#     (1.5, 7.5, ['gg', 'gk'], ['deep_fields'], True, 252, 502, 'halofit', [], False),
#     (1.5, 7.5, ['gg'], [], True, 252, 'linear', [], True),
#     (1.5, 7.5, ['gg'], [], True, 502, 'linear', [], True),
#     (1.5, 7.5, ['gg'], [], True, 252, 'halofit', [], True),
#     (1.5, 7.5, ['gg'], [], True, 502, 'halofit', [], True),
# ]

# Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tag', dest='tag', help='tag, added as suffix to the experiment name')
args = parser.parse_args()

# Read YAML configuration file
config = get_config('LoTSS_DR2')
config.experiment_tag = args.tag

# Set proper flags, assuming covariance ready and saved, correlations have to be calculated
# to get NaMaster workspaces for decoupling
config.read_correlations_flag = False
config.read_covariance_flag = True
config.max_iterations = 10000

for i, (flux_cut, snr_cut, correlation_symbols, redshifts, with_A_sn, ell_max_gg, ell_max_gk, matter_power_spectrum,
        cosmo_params, is_mock) in enumerate(to_run):
    for j, bias_model in enumerate(bias_models):
        print('Setup {}/{}: bias {}/{}'.format(i + 1, len(to_run), j + 1, len(bias_models)))
        print(
            '{} mJy, {} SNR, correlations: {}, redshifts: {}, with A_sn = {}, ell_max_gg = {}, ell_max_gk = {}, matter power spectrum = {}, bias_model = {}'.format(
                flux_cut, snr_cut, correlation_symbols, redshifts, with_A_sn, ell_max_gg, ell_max_gk,
                matter_power_spectrum, bias_model
            ))

        config.is_mock = is_mock
        config.flux_min_cut = flux_cut
        config.signal_to_noise = snr_cut
        config.correlations_to_use = correlation_symbols
        config.ells_per_bin = dict([(correlation_symbol, 50) for correlation_symbol in ['gg', 'gk']])
        config.ells_per_bin['gt'] = 16
        config.l_range['gg'] = [52, ell_max_gg]
        config.l_range['gk'] = [52, ell_max_gk]
        config.l_range['gt'] = [2, 36]

        config.redshifts_to_fit = redshifts
        config.dn_dz_model = 'deep_fields' if len(redshifts) == 0 else 'power_law'
        config.bias_model = bias_model
        config.matter_power_spectrum = matter_power_spectrum

        to_infere = copy.copy(cosmo_params)
        if bias_model == 'constant':
            to_infere += ['b_g']
        elif bias_model == 'scaled':
            to_infere += ['b_g_scaled']
        elif bias_model == 'quadratic_limited':
            to_infere += ['b_a', 'b_b']
        elif bias_model == 'quadratic':
            to_infere += ['b_0', 'b_1', 'b_2']

        if with_A_sn:
            to_infere.append('A_sn')

        if len(redshifts) > 0:
            to_infere += ['z_sfg', 'a', 'r']
        if 'tomographer' in redshifts:
            to_infere.append('n')

        config.to_infere = to_infere

        # Run emcee
        experiment = Experiment(config, set_data=True, set_maps=True, set_correlations=True)
        experiment.run_mcmc()

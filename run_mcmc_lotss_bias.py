import argparse

from utils import get_config
from experiment import Experiment

to_run = [
    # Correlation symbol, redshifts, with A_sn, ell_max
    # Bias study
    # (['gg'], ['deep_fields'], False),
    # (['gg'], ['deep_fields'], True),
    # (['gk'], ['deep_fields'], False),
    # (['gg', 'gk'], ['deep_fields'], False),
    # (['gg', 'gk'], ['deep_fields'], True),
    # (['gg', 'gk'], ['tomographer'], False),
    # (['gg', 'gk'], ['tomographer'], True),
    # (['gg', 'gk'], ['deep_fields', 'tomographer'], False),
    # (['gg', 'gk'], ['deep_fields', 'tomographer'], True),

    # ell range test
    (['gg', 'gk'], ['deep_fields'], True, 252),
    (['gg', 'gk'], ['deep_fields'], True, 802),
]

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

for correlation_symbols, redshifts, with_A_sn, ell_max in to_run:
    for bias_model in ['scaled']:
        print('Processing correlations {}, redshifts = {}, with A_sn = {}, bias_model = {}'.format(
            correlation_symbols, redshifts, with_A_sn, bias_model
        ))

        config.correlations_to_use = correlation_symbols
        config.l_range = dict([(correlation_symbol, [52, 502]) for correlation_symbol in correlation_symbols])
        config.ells_per_bin = dict([(correlation_symbol, 50) for correlation_symbol in correlation_symbols])
        config.redshifts_to_fit = redshifts
        config.fit_bias_to_tomo = ('tomographer' in redshifts)
        config.bias_model = bias_model

        config.l_range['gg'][1] = ell_max

        to_infere = []
        if bias_model == 'constant':
            to_infere = ['b_g']
        elif bias_model == 'scaled':
            to_infere = ['b_g_scaled']
        elif bias_model == 'quadratic_limited':
            to_infere = ['b_a', 'b_b']
        elif bias_model == 'quadratic':
            to_infere = ['b_0', 'b_1', 'b_2']

        if with_A_sn:
            to_infere.append('A_sn')

        to_infere += ['z_sfg', 'a', 'r']
        if 'tomographer' in redshifts:
            to_infere.append('n')

        config.to_infere = to_infere

        # Run emcee
        experiment = Experiment(config, set_data=True, set_maps=True, set_correlations=True)
        experiment.run_mcmc()

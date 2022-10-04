import argparse

from utils import get_config, save_correlations
from experiment import Experiment

# Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config_name', required=True, help='configuration name')
parser.add_argument('-t', '--tag', dest='tag', help='tag, added as suffix to the experiment name')
args = parser.parse_args()

# Read YAML configuration file
config = get_config(args.config_name)
config.experiment_tag = args.tag
config.read_correlations_flag = False
config.read_covariance_flag = False
config.redshifts_to_fit = []
print(config.__dict__)

# Iterate thorugh parameters
for flux_min_cut in [1.5]:
    for signal_to_noise in [7.5]:
        print('Processing: flux={}, snr={}'.format(flux_min_cut, signal_to_noise))
        config.flux_min_cut = flux_min_cut
        config.signal_to_noise = signal_to_noise

        # Radio in inner mask
        experiment = Experiment(config, set_data=True, set_maps=True)
        experiment.set_correlations()
        save_correlations(experiment)

        # Mock
        if flux_min_cut == 1.5 and signal_to_noise == 7.5:
            config.is_mock = True
            experiment = Experiment(config, set_data=True, set_maps=True)
            experiment.set_correlations()
            save_correlations(experiment)

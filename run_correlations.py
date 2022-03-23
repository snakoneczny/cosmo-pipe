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
config.read_data_correlations_flag = False
config.read_covariance_flag = False
config.fit_tomographer = False  # TODO: this will change
print(config)

# Iterate thorugh parameters
for flux_min_cut in [1.5]:
    for signal_to_noise in [7.5]:
        print('Processing: flux={}, snr={}'.format(flux_min_cut, signal_to_noise))
        config.flux_min_cut = flux_min_cut
        config.signal_to_noise = signal_to_noise

        # VAC in optical field
        config.is_optical = True
        config.lss_mask_name = 'mask_optical'

        experiment = Experiment(config, set_data=True, set_maps=True)
        experiment.set_correlations(with_covariance=True)
        save_correlations(experiment)

        # Radio in optical field
        config.is_optical = False
        config.lss_mask_name = 'mask_optical'

        experiment = Experiment(config, set_data=True, set_maps=True)
        experiment.set_correlations(with_covariance=True)
        save_correlations(experiment)

        # Radio in inner mask
        config.is_optical = False
        config.lss_mask_name = 'mask_inner'

        experiment = Experiment(config, set_data=True, set_maps=True)
        experiment.set_correlations(with_covariance=True)
        save_correlations(experiment)

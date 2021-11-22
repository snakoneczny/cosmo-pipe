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
print(config)

# Set correlations
experiment = Experiment(config, set_data=True, set_maps=True)
experiment.set_correlations(with_covariance=True)

# Save correlations
# save_correlations(experiment)

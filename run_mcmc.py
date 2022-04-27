import argparse

from utils import get_config
from experiment import Experiment

# Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config_name', required=True, help='configuration name')
parser.add_argument('-t', '--tag', dest='tag', help='tag, added as suffix to the experiment name')
args = parser.parse_args()

# Read YAML configuration file
config = get_config(args.config_name)
config.experiment_tag = args.tag

# Set proper flags, assuming covariance ready and saved, correlations have to be calculated
# to get NaMaster workspaces for decoupling
config.read_correlations_flag = False
config.read_covariance_flag = True

# Run emcee
experiment = Experiment(config, set_data=True, set_maps=True, set_correlations=True)
experiment.run_mcmc()

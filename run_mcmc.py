import argparse
import os

import yaml

from env_config import PROJECT_PATH
from experiment import Experiment

# Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config_name', required=True, help='configuration name')
parser.add_argument('-t', '--tag', dest='tag', help='tag, added as suffix to the experiment name')
args = parser.parse_args()

# Read YAML configuration file
with open(os.path.join(PROJECT_PATH, 'configs.yml'), 'r') as config_file:
    config = yaml.full_load(config_file)[args.config_name]
config['experiment_tag'] = args.tag

# Run emcee
experiment = Experiment(config)
experiment.run_emcee()

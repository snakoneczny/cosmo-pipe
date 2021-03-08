import argparse

import yaml

from experiment import Experiment

# Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config_file', required=True, help='config file name')
parser.add_argument('-t', '--tag', dest='tag', help='catalog tag, added to logs name')
args = parser.parse_args()

# Read YAML configuration file
with open(args.config_file, 'r') as config_file:
    config = yaml.full_load(config_file)
config['experiment_tag'] = args.tag

# Run emcee
experiment = Experiment(config)
experiment.run_emcee()

import os

import numpy as np
import yaml
import pyccl as ccl

from env_config import PROJECT_PATH
from utils import struct, ISWTracer
from data_lotss import get_lotss_redshift_distribution


def get_theory_correlations(config, correlation_symbols, l_arr, omega_param=None):
    config = struct(**config)

    z_arr, n_arr = get_lotss_redshift_distribution(z_tail=config.z_tail)

    with open(os.path.join(PROJECT_PATH, 'cosmologies.yml'), 'r') as cosmology_file:
        cosmology_params = yaml.full_load(cosmology_file)[config.cosmology_name]
    cosmology_params['matter_power_spectrum'] = config.cosmology_matter_power_spectrum

    if omega_param:
        if omega_param[0] == 'Omega_c_b_frac':
            cosmology_params['Omega_c'] *= omega_param[1]
            cosmology_params['Omega_b'] *= omega_param[1]
        elif omega_param[0] == 'Omega_k':
            cosmology_params['Omega_k'] = omega_param[1]

    cosmology = ccl.Cosmology(**cosmology_params)

    bias_arr = config.bias * np.ones(len(z_arr))
    if config.scale_bias:
        bias_arr = bias_arr / ccl.growth_factor(cosmology, 1. / (1. + z_arr))

    tracers_dict = {
        'g': ccl.NumberCountsTracer(cosmology, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr)),
        'k': ccl.CMBLensingTracer(cosmology, 1091),
        't': ISWTracer(cosmology, z_max=6., n_chi=1024),
    }

    theory_correlations = {}
    for correlation_symbol in correlation_symbols:
        tracer_symbol_a = correlation_symbol[0]
        tracer_symbol_b = correlation_symbol[1]
        correlation_symbol = tracer_symbol_a + tracer_symbol_b
        theory_correlations[correlation_symbol] = ccl.angular_cl(cosmology, tracers_dict[tracer_symbol_a],
                                                                 tracers_dict[tracer_symbol_b], l_arr)

    return theory_correlations

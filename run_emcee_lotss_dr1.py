from experiment import Experiment

config = {
    'correlation_symbols': ['gg', 'gk'],
    'lss_survey': 'LoTSS DR1',
    'l_max': 510,
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
    'experiment_tag': 'prior-fix',
}

experiment = Experiment(config)
experiment.run_emcee()

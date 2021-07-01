from utils import get_config, save_correlations
from experiment import Experiment


for is_optical in [True, False]:
    for signal_to_noise in [0, 1, 2, 3, 4, 5]:
        for flux_min_cut in [0.5, 1, 2]:
            print(is_optical, signal_to_noise, flux_min_cut)

            config = get_config('LoTSS_DR1', configs_file='configs.yml')
            config['is_optical'] = is_optical
            config['signal_to_noise'] = signal_to_noise
            config['flux_min_cut'] = flux_min_cut

            experiment = Experiment(config, set_data=True, set_maps=True)

            experiment.set_correlations(with_covariance=False)

            save_correlations(experiment)

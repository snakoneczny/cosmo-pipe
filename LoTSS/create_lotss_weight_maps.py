import os

import healpy as hp

from env_config import DATA_PATH
from data_lotss import get_lotss_noise_weight_map

output_directory = 'LoTSS/DR{dr}/weight_maps/weight_map__pointing-mean_minflux-{s}_snr-{n}.fits'

for data_release in [2]:

    # Read noise map
    # data = get_lotss_data(data_release=data_release, flux_min_cut=None, optical=False)
    # _, _, noise_map = get_lotss_map(data, data_release=data_release, nside=256, cut_pixels=False, masked=False)
    noise_map = hp.read_map(
        os.path.join(DATA_PATH, 'LoTSS/DR{dr}/Alonso/maps_all_DR{dr}.fits'.format(dr=data_release)), field=1)
    noise_map *= 1E3

    for flux_min_cut in [0.5, 1, 1.5, 2]:
        for signal_to_noise in [0, 1, 2, 3, 4, 5, 6, 7, 7.5]:
            print('dr={}, flux={}, snr={}'.format(data_release, flux_min_cut, signal_to_noise))

            # Output filepath
            file_path = os.path.join(DATA_PATH, output_directory.format(
                dr=data_release, s=flux_min_cut, n=signal_to_noise))

            if os.path.exists(file_path):
                print('Already exists')

            else:
                # Get weight/probability map
                weight_map = get_lotss_noise_weight_map(noise_map, flux_cut=flux_min_cut, signal_to_noise=signal_to_noise)

                # Save map
                if not os.path.exists(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path))
                hp.fitsfunc.write_map(file_path, weight_map, dtype=float, overwrite=True)

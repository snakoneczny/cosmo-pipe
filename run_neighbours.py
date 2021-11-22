from multiprocessing import Pool
from functools import partial
import math
import os

import numpy as np
from tqdm import tqdm
import healpy as hp

from env_config import PROJECT_PATH
from data_lotss import get_lotss_data, get_biggest_optical_region


def get_closest_distances_for_idx(i, data, max_distance):
    distances = []

    i_ra = data.loc[i]['RA']
    i_dec = data.loc[i]['DEC']

    # Limit distance calculation to a square of max distance
    min_ra = i_ra - max_distance
    max_ra = i_ra + max_distance
    min_dec = i_dec - max_distance
    max_dec = i_dec + max_distance

    if min_ra > 0 and max_ra < 360:
        query_lon = (min_ra < data['RA']) & (data['RA'] < max_ra)
    elif min_ra < 0:
        query_lon = ((0 < data['RA']) & (data['RA'] < max_ra)) | ((min_ra + 360 < data['RA']) & (data['RA'] < 360))
    else:  # max_ra > 360
        query_lon = ((min_ra < data['RA']) & (data['RA'] < 360)) | ((0 < data['RA']) & (data['RA'] < max_ra - 360))

    query_lat = (min_dec < data['DEC']) & (data['DEC'] < max_dec)

    # Get distances of all the neighbors
    data_neighbours = data.loc[query_lon & query_lat].reset_index(drop=True)
    i_name = data.loc[i]['Source_Name']
    for j in range(data_neighbours.shape[0]):
        j_name = data_neighbours.loc[j]['Source_Name']
        if i_name == j_name:
            continue
        j_ra = data_neighbours.loc[j]['RA']
        j_dec = data_neighbours.loc[j]['DEC']
        distance = math.degrees(hp.rotator.angdist((i_ra, i_dec), (j_ra, j_dec), lonlat=True))
        if distance < max_distance:
            distances.append((i_name, j_name, distance))

    return distances


def get_closest_distances(data, max_distance=1):
    distances = []
    with Pool(12) as p:
        tasks = np.arange(data.shape[0])
        func = partial(get_closest_distances_for_idx, data=data, max_distance=max_distance)
        for distance_batch in tqdm(p.imap_unordered(func, tasks), total=len(tasks)):
            distances.extend(distance_batch)

    return distances


# Get distances
max_distance = 0.2
data = get_lotss_data(data_release=2, flux_min_cut=2, signal_to_noise=5, optical=False,
                      columns=['Source_Name', 'RA', 'DEC'])
data = get_biggest_optical_region(data).reset_index(drop=True)
distances = get_closest_distances(data, max_distance=max_distance)

# Save distances
folder_path = os.path.join(PROJECT_PATH, 'outputs/distances/LoTSS_DR2')
file_name = 'LoTSS_DR2_srl_2mJy_snr=5_max-dist={}__opt-big-patch.npy'.format(max_distance)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
np.save(os.path.join(folder_path, file_name), distances)

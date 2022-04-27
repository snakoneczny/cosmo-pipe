import os

import healpy as hp

from env_config import DATA_PATH
from data_lotss import get_dr2_inner_regions

for nside in [256, 512, 1024, 2056]:
    mask = get_dr2_inner_regions(nside)
    filepath = os.path.join(DATA_PATH, 'LoTSS/DR2/masks/mask_inner/mask_inner_nside={}.fits'.format(nside))
    hp.fitsfunc.write_map(filepath, mask)
    print('Mask at nside={} saved to {}'.format(nside, filepath))

import numpy as np
import healpy as hp
from utils import FluxPDF
import sys
from scipy.interpolate import interp1d
import os


if len(sys.argv) != 5:
    print("Usage: get_p_map.py fname_rms Icut q fname_out")
    exit(1)

fname_rms = sys.argv[1]
Icut = float(sys.argv[2])
q = float(sys.argv[3])
fname_out = sys.argv[4]

if os.path.isfile(fname_out):
    print(f"File {fname_out} exists")
    exit(0)
print(f"Computing {fname_out}")

fpdf = FluxPDF()

msk = hp.read_map(fname_rms, field=0)
rms_map = hp.read_map(fname_rms, field=1)*1E3  # in mJy

rms_min = np.amin(rms_map[rms_map > 0])
rms_max = np.amax(rms_map[rms_map > 0])
rms_arr = np.geomspace(rms_min, rms_max, 2048)
p_arr = fpdf.compute_p_map(q, rms_arr, Icut)
pf = interp1d(np.log(rms_arr), p_arr, fill_value=(p_arr[0], p_arr[-1]), bounds_error=False)
print("Mapping")
p_map = np.zeros_like(rms_map)
p_map[rms_map > 0] = pf(np.log(rms_map[rms_map > 0]))
p_map /= np.amax(p_map)
hp.write_map(fname_out, [p_map*msk, p_map, msk], overwrite=True,
             column_names=['p_map', 'p_map_comp', 'p_map_geom'])

import matplotlib.pyplot as plt
p_map[p_map == 0] = hp.UNSEEN
hp.mollzoom(rms_map)
hp.mollzoom(p_map)
hp.mollzoom(msk)
plt.show()

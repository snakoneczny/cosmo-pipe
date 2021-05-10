import numpy as np
import healpy as hp
from astropy.io import fits
from utils import FluxPDF
from scipy.interpolate import interp1d
import sys
import os
import matplotlib.pyplot as plt


if len(sys.argv) != 6:
    print("Usage: get_p_map_from_catalog.py fname_cat fname_mask Icut q fname_out")
    exit(1)

fname_cat = sys.argv[1]
fname_mask = sys.argv[2]
Icut = float(sys.argv[3])
q = float(sys.argv[4])
fname_out = sys.argv[5]

if os.path.isfile(fname_out):
    print(f"File {fname_out} exists")
    exit(0)
print(f"Computing {fname_out}")

fpdf = FluxPDF()

# Make RMS map from catalog
nside_lo = 256
npix_lo = hp.nside2npix(nside_lo)

cat = fits.open(fname_cat)[1].data
ipix = hp.ang2pix(nside_lo, cat['RA'], cat['DEC'], lonlat=True)

nc = np.bincount(ipix, minlength=npix_lo)
rms = np.bincount(ipix, minlength=npix_lo, weights=cat['Isl_rms'])
rms[nc > 0] = rms[nc > 0] / nc[nc > 0]
rms[nc <= 0] = 0
rmsp = rms.copy()
rmsp[nc <= 0] = hp.UNSEEN

# Translate to P-map
rms_min = np.amin(rms[rms > 0])
rms_max = np.amax(rms[rms > 0])
rms_arr = np.geomspace(rms_min, rms_max, 2048)
p_arr = fpdf.compute_p_map(q, rms_arr, Icut)
pf = interp1d(np.log(rms_arr), p_arr, fill_value=(p_arr[0], p_arr[-1]), bounds_error=False)
print("Mapping")
p_map = np.zeros_like(rms)
p_map[rms > 0] = pf(np.log(rms[rms > 0]))
p_map /= np.amax(p_map)

# Multiply by geometry and save
msk = hp.read_map(fname_mask, field=0)
nside = hp.npix2nside(len(msk))
p_map = hp.ud_grade(p_map, nside_out=nside)
hp.write_map(fname_out, [p_map*msk, p_map, msk], overwrite=True,
             column_names=['p_map', 'p_map_comp', 'p_map_geom'])
hp.mollzoom(msk)
hp.mollzoom(p_map)
plt.show()

from utils import Pointings
import healpy as hp
import numpy as np


nside = 2048
npix = hp.nside2npix(nside)

pt = Pointings('data/pointings.txt', '/mnt/extraspace/damonge/LensLotss')
n_pointings = len(pt.data['name'])

mask = np.zeros(npix)
ngood = np.zeros(npix)
vmean = np.zeros(npix)
vstd = np.zeros(npix)
for i_n, n in enumerate(pt.data['name']):
    n = n.decode()
    print(i_n, n_pointings, n, flush=True)
    fname = pt.prefix_out + f'/map_rms_{n}.fits.gz'
    v, v2, ng = hp.read_map(fname, field=[0, 1, 2])
    msk = ng / np.amax(ng)
    msk[msk > 0.99] = 1
    mask += msk
    ngood += ng
    vmean += v
    vstd += v2

mask[mask > 1] = 1
igood = ngood > 0
vmean[igood] = vmean[igood]/ngood[igood]
vstd[igood] = vstd[igood]/ngood[igood]
vstd = np.sqrt(vstd - vmean**2)

hp.write_map("/mnt/extraspace/damonge/LensLotss/maps_all.fits.gz",
             [mask, vmean, vstd, ngood],
             column_names=['mask', 'rms_mean', 'rms_std', 'n_good'])

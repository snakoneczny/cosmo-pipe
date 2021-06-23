from lotss_corr import Pointings
import healpy as hp
import numpy as np
import os


nside = 2048
npix = hp.nside2npix(nside)

pt_dr1 = Pointings('data/pointings_dr1.txt', '/mnt/extraspace/damonge/LensLotss/DR1_data/',
                   dr=1, bad_pointings='data/bad_pointings_dr1.txt')
pt_dr1_all = Pointings('data/pointings_dr1.txt', '/mnt/extraspace/damonge/LensLotss/DR1_data/',
                       dr=1)
pt_dr2 = Pointings('data/pointings.txt', '/mnt/extraspace/damonge/LensLotss/DR2_data/')

pts = [pt_dr1, pt_dr1_all, pt_dr2]
fn_maps = ['maps_good.fits.gz', 'maps_all.fits.gz', 'maps_all.fits.gz']

for pt, mname in zip(pts, fn_maps):
    fname_maps = pt.prefix_out + mname
    n_pointings = len(pt.data['name'])
    if os.path.isfile(fname_maps):
        continue
    mask = np.zeros(npix)
    ngood = np.zeros(npix)
    vmean = np.zeros(npix)
    vstd = np.zeros(npix)
    for i_n, n in enumerate(pt.data['name']):
        n = n.decode()
        print(i_n, n_pointings, n, flush=True)
        if n in pt.bad:
            print(" Skipping bad pointing")
            continue
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

    hp.write_map(fname_maps,
                 [mask, vmean, vstd, ngood],
                 column_names=['mask', 'rms_mean', 'rms_std', 'n_good'])

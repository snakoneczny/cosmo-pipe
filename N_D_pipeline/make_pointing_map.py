import sys
import flatmaps as fm
import healpy as hp
import numpy as np

fname_batch = sys.argv[1]

batch = open(fname_batch, 'r')
lines = batch.readlines()
for line in lines:
    fname_in, fname_out = line.split(' ')
    fname_out = fname_out.rstrip()
    print(fname_in, flush=True)
    fsk, mp = fm.read_flat_map(fname_in)
    npix = fsk.nx*fsk.ny
    ipix = np.arange(npix)
    ra, dec = fsk.pix2pos(ipix)
    nside = 2048
    npix_hp = hp.nside2npix(nside)
    ipix_hp = hp.ang2pix(nside, ra, dec, lonlat=True)
    i_good = ~np.isnan(mp)
    nc_good = np.bincount(ipix_hp[i_good], minlength=npix_hp)
    nc_all = np.bincount(ipix_hp, minlength=npix_hp)
    vsum = np.bincount(ipix_hp[i_good], minlength=npix_hp, weights=mp[i_good])
    v2sum = np.bincount(ipix_hp[i_good], minlength=npix_hp, weights=mp[i_good]**2)
    hp.write_map(fname_out, [vsum, v2sum, nc_good, nc_all],
                 column_names=['v_sum', 'v2_sum', 'n_good', 'n_all'])

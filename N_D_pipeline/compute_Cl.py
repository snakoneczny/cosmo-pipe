import numpy as np
import healpy as hp
import pymaster as nmt
import os
from utils import Bandpowers, Field
from astropy.io import fits
import sys


if len(sys.argv) != 9:
    print("Usage: compute_Cl.py output_dir catalog_name p_map_name kappa_name mask_kappa_name noise_kappa_name flux_threshold SN_threshold")
    exit(1)
dirsave = sys.argv[1]
cat_name = sys.argv[2]
p_map_name = sys.argv[3]
kappa_name = sys.argv[4]
mask_k_name = sys.argv[5]
noise_k_name = sys.argv[6]
I_thr = float(sys.argv[7])
q_thr = float(sys.argv[8])

nside = 2048
os.system(f'mkdir -p {dirsave}')

# Create bandpowers
bands = {
    "lsplit": 52,
    "nb_log": 28,
    "nlb": 20,
    "nlb_lin": 10,
    "type": "linlog"}
B = Bandpowers(nside, bands)

# Read catalog and make counts map
cat = fits.open(cat_name)[1].data
npix = hp.nside2npix(nside)
ipix = hp.ang2pix(nside, cat['RA'], cat['DEC'], lonlat=True)
nc = np.bincount(ipix[(cat['Total_flux'] > I_thr) &
                      (cat['Total_flux']/cat['Total_flux'] >= q_thr)],
                 minlength=npix)
ncounts_name = f'{dirsave}/counts_map_Icut{I_thr}_{q_thr}sig.fits.gz', nc)
hp.write_map(ncounts_name, nc, overwrite=True)

# Create fields
f_g = Field(ncounts_name, p_map_name, 'g', nside)
f_k = Field(kappa_name, mask_k_name, 'k', nside,
            fname_kappa_noise=noise_k_name)

# Mode-coupling matrices
def get_wsp(f1, f2):
    fname = f'{dirsave}/w{f1.kind}{f2.kind}.fits'
    w = nmt.NmtWorkspace()
    if not os.path.isfile(fname):
        w.compute_coupling_matrix(f1.get_field(), f2.get_field(), B.bn)
        w.write_to(fname)
    else:
        w.read_from(fname)
    return w
wgg = get_wsp(f_g, f_g)
wgk = get_wsp(f_g, f_k)
wkk = get_wsp(f_k, f_k)

# Power spectra
leff = B.bn.get_effective_ells()
clgg = wgg.decouple_cell(nmt.compute_coupled_cell(f_g.get_field(), f_g.get_field()))[0]
clgk = wgk.decouple_cell(nmt.compute_coupled_cell(f_g.get_field(), f_k.get_field()))[0]
clkk = wkk.decouple_cell(nmt.compute_coupled_cell(f_k.get_field(), f_k.get_field()))[0]

# Noise bias
nlgg = wgg.decouple_cell(f_g.get_nl_coupled())[0]
nlkk = wkk.decouple_cell(f_k.get_nl_coupled())[0]

# Covariance MCM
def get_cwsp(f1, f2, f3, f4):
    fname = f'{dirsave}/cw{f1.kind}{f2.kind}_{f3.kind}{f4.kind}.fits'
    cw = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(fname):
        cw.compute_coupling_coefficients(f1.get_field(), f2.get_field(),
                                         f3.get_field(), f4.get_field())
        cw.write_to(fname)
    else:
        cw.read_from(fname)
    return cw
cwgggg = get_cwsp(f_g, f_g, f_g, f_g)
cwgggk = get_cwsp(f_g, f_g, f_g, f_k)
cwgkgk = get_cwsp(f_g, f_k, f_g, f_k)


# Power spectra for covariance matrix
# For now we use the measurements themselves
from scipy.interpolate import interp1d

def interp_eval(l, cl, l_ev):
    clf = interp1d(l, cl, bounds_error=False, fill_value=(cl[0], cl[-1]))
    return np.array([clf(l_ev)])
ls = np.arange(3*nside)
clc_gg = interp_eval(leff, clgg, ls)
clc_gk = interp_eval(leff, clgk, ls)
clc_kk = interp_eval(leff, np.fabs(clkk), ls)

# This is commented out until we have a proper theory prediction
'''
# Theory power spectra
# CCL tracers
import pyccl as ccl
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.8)
t_g = f_g.get_tracer(cosmo, 1.5)
t_k = f_k.get_tracer(cosmo)
# Theory power spectra
tl_gg = ccl.angular_cl(cosmo, t_g, t_g, ls)
tl_gk = ccl.angular_cl(cosmo, t_g, t_k, ls)
tl_kk = ccl.angular_cl(cosmo, t_k, t_k, ls)
# Power spectra used for the covariance matrix
m_g = f_g.get_mask()
m_k = f_k.get_mask()
fsky_gg = np.mean(m_g*m_g)
fsky_gk = np.mean(m_g*m_k)
fsky_kk = np.mean(m_k*m_k)
tlc_gg = wgg.couple_cell([tl_gg])/fsky_gg
tlc_gk = wgk.couple_cell([tl_gk])/fsky_gk
tlc_kk = wkk.couple_cell([tl_kk])/fsky_kk
nlc_gg = f_g.get_nl_coupled()/fsky_gg
nlc_gk = np.zeros_like(nlc_gg)
nlc_kk = f_k.get_nl_coupled()/fsky_kk
clc_gg = tlc_gg+nlc_gg
clc_gk = tlc_gk+nlc_gk
clc_kk = tlc_kk+nlc_kk
'''

# Covariance matrices
cov_gg_gg = nmt.gaussian_covariance(cwgggg, 0, 0, 0, 0,
                                    clc_gg, clc_gg, clc_gg, clc_gg,
                                    wgg, wgg)
cov_gg_gk = nmt.gaussian_covariance(cwgggk, 0, 0, 0, 0,
                                    clc_gg, clc_gk, clc_gg, clc_gk,
                                    wgg, wgk)
cov_gk_gk = nmt.gaussian_covariance(cwgkgk, 0, 0, 0, 0,
                                    clc_gg, clc_gk, clc_gk, clc_kk,
                                    wgk, wgk)

np.savez(f"{dirsave}/cls_cov.npz",
         ls=leff,
         clgg=clgg, clgk=clgk, clkk=clkk,
         nlgg=nlgg, nlkk=nlkk,
         cov_gg_gg=cov_gg_gg,
         cov_gg_gk=cov_gg_gk,
         cov_gk_gk=cov_gk_gk)

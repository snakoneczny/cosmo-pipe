import numpy as np
import pyccl as ccl
import healpy as hp
import pymaster as nmt


class Pointings(object):
    def __init__(self, fname_pointings, prefix_out):
        self.prefix_out = prefix_out
        self.data = np.genfromtxt(fname_pointings,
                                  dtype='S256,<f8,<f8,S256,S256,S256,S256,S256',
                                  names=['name','ra','dec','fr_mosaic',
                                         'fr_rms','fr_res','lr_mosaic','cat'])

class Bandpowers(object):
    """
    A class encoding the information about a set of bandpowers (i.e. bins
    of multipoles ell.
    Args:
        nside (int): HEALPix resolution parameter.
        d (dict): dictionary containing all arguments in the \'bandpowers\'
            section of the input parameter file.
    """
    def __init__(self, nside, d):
        if d['type'] == 'linlog':  # Check if using hybrid binning
            # Set up linear part
            l_edges_lin = np.linspace(2, d['lsplit'],
                                      (d['lsplit']-2)//d['nlb_lin']+1)
            l_edges_lin = l_edges_lin.astype(int)
            # Set up log part
            l_edges_log = np.unique(np.logspace(np.log10(d['lsplit']),
                                                np.log10(3*nside-1),
                                                d['nb_log']).astype(int))
            # Join
            l_edges = np.concatenate((l_edges_lin, l_edges_log[1:]))

            # Give bandpower indices and weights to each multipole
            larr = np.arange(3*nside)
            weights = np.ones(len(larr))
            bpws = -1+np.zeros(len(larr), dtype=int)
            for i in range(len(l_edges)-1):
                bpws[l_edges[i]:l_edges[i+1]] = i

            # Create binning scheme
            self.bn = nmt.NmtBin(nside, ells=larr, bpws=bpws, weights=weights)
        elif d['type'] == 'lin':  # Check if using linear binning
            # Create binning scheme
            self.bn = nmt.NmtBin(nside, nlb=d['nlb'])
        else:
            raise ValueError("Unrecognised binning scheme "+d['type'])


class Field(object):
    def __init__(self, fname_map, fname_mask, kind, nside,
                 fname_kappa_noise=None):
        self.fname_map = fname_map
        self.fname_mask = fname_mask
        self.kind = kind
        if self.kind == 'k':
            self.fname_kappa_noise = fname_kappa_noise
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.f = None
        self.nl_coupled = None
        self.cosmo = None
        self.tracer = None
        self.msk = None

    def _read_maps(self):
        mp = self._read_map(self.fname_map, self.nside)
        msk = self._read_map(self.fname_mask, self.nside)
        if self.kind == 'g':
            # Scale mask
            msk = msk / np.amax(msk)
            # Compute mean and delta
            self.mean_n = np.sum(mp[msk > 0])/np.sum(msk[msk > 0])
            mp[msk > 0] = mp[msk > 0] / (self.mean_n * msk[msk > 0]) - 1
            mp[msk <= 0] = 0
        return mp, msk

    def _read_map(self, fname, nside):
        return hp.ud_grade(hp.read_map(fname, verbose=False),
                           nside_out=nside)

    def get_field(self, n_iter=0):
        if self.f is None:
            mp, self.msk = self._read_maps()
            self.f = nmt.NmtField(self.msk, [mp], n_iter=n_iter)
        return self.f

    def get_mask(self):
        if self.msk is None:
            _, self.msk = self._read_maps()
        return self.msk

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            if self.kind == 'g':
                mp, self.msk = self._read_maps()
                ndens = self.mean_n * self.npix / (4*np.pi)
                self.nl_coupled = (np.mean(self.msk) *
                                   np.ones([1, 3*self.nside]) / ndens)
            else:
                from scipy.interpolate import interp1d
                l, nl, _ = np.loadtxt(self.fname_kappa_noise, unpack=True)
                nlf = interp1d(l, nl, bounds_error=False,
                               fill_value=(nl[0], 0))
                nll = nlf(np.arange(3*self.nside))
                nll[:2] = 0
                self.msk = self.get_mask()
                self.nl_coupled = np.array([nll]) * np.mean(self.msk**2)
        return self.nl_coupled

    def _nz_fit(self, z, ztail=1.5):
        z0 = 0.1
        gamma = 3.5
        x = z/z0
        return x**2/(1+x**2)/(1+(z/ztail)**gamma)

    def get_tracer(self, cosmo, bias=None, ztail=1.5):
        if self.tracer is None:
            if self.kind == 'g':
                z = np.linspace(0, 5, 1024)
                bz = bias / ccl.growth_factor(cosmo, 1./(1+z))
                nz = self._nz_fit(z, ztail=ztail)
                self.tracer = ccl.NumberCountsTracer(cosmo, False, (z, nz),
                                                     bias=(z, bz))
            else:
                self.tracer = ccl.CMBLensingTracer(cosmo, z_source=1100.)
        return self.tracer

import numpy as np
import pyccl as ccl
import healpy as hp
import pymaster as nmt
from scipy.special import erf


class Pointings(object):
    def __init__(self, fname_pointings, prefix_out, dr=2, bad_pointings=None):
        self.prefix_out = prefix_out
        self.dr = dr
        if self.dr == 2:
            self.data = np.genfromtxt(fname_pointings,
                                      dtype='S256,<f8,<f8,S256,S256,S256,S256,S256',
                                      names=['name','ra','dec','fr_mosaic',
                                             'fr_rms','fr_res','lr_mosaic','cat'])
        elif self.dr == 1:
            self.data = np.genfromtxt(fname_pointings,
                                      dtype='S256,<f8,<f8,S256,S256,S256,S256,S256',
                                      names=['name','ra','dec','fr_mosaic',
                                             'fr_rms','fr_res','lr_mosaic','lr_res'])
        else:
            raise ValueError("'dr' can be 1 or 2")

        if bad_pointings is None:
            self.bad = {}
        else:
            dbad = np.genfromtxt(bad_pointings,
                                 dtype='S256,<f8,<f8',
                                 names=['name', 'ra', 'dec'])
            self.bad = {d['name'].decode(): [d['ra'], d['dec']]
                        for d in dbad}


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
                 fname_kappa_noise=None, mask_thr=0):
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
        self.msk_thr = mask_thr

    def _read_maps(self):
        mp = self._read_map(self.fname_map, self.nside)
        msk = self._read_map(self.fname_mask, self.nside)
        if self.kind == 'g':
            # Scale mask
            msk = msk / np.amax(msk)
            mp[msk <= self.msk_thr] = 0
            msk[msk <= self.msk_thr] = 0
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


class FluxPDF(object):
    def __init__(self, fname_in="data/skads_flux_counts.result"):
        from scipy.interpolate import interp1d
        # Read flux distribution from SKADS' S3-SEX simulation
        self.log_flux, counts = np.loadtxt(fname_in, unpack=True,
                                           delimiter=',', skiprows=1)
        self.log_flux += 3  # Use mJy instead of Jy
        # Assuming equal spacing
        self.dlog_flux = np.mean(np.diff(self.log_flux))
        self.log_flux = self.log_flux[counts >= 0]
        counts = counts[counts >= 0]
        self.probs = counts / np.sum(counts)
        # Cut to non-zero counts
        self.lpdf = interp1d(self.log_flux, np.log10(counts),
                             fill_value=-500, bounds_error=False)

    def plot_pdf(self, log_flux_min=-6, log_flux_max=6,
                 n_log_flux=256):
        lf = np.linspace(log_flux_min, log_flux_max, n_log_flux)
        dlf = np.mean(np.diff(lf))
        p = 10.**self.lpdf(lf)
        p /= np.sum(p) * dlf
        plt.figure()
        plt.plot(10.**lf, p, 'k-')
        plt.loglog()
        plt.xlabel(r'$I_{1400}\,{\rm mJy}$', fontsize=14)
        plt.ylabel(r'$dp/d\log_{10}I_{1400}$', fontsize=14)
        plt.show()

    def compute_p_map(self, q, std_map, Imin, alpha=-0.7):
        lf = self.log_flux + alpha * np.log10(144. / 1400.)
        p_map = np.zeros(len(std_map))
        for ip, std in enumerate(std_map):
            if std > 0:
                Ithr = max(q * std, Imin)
                x = (Ithr - 10.**lf) / (np.sqrt(2.) * std)
                comp = 0.5 * (1 - erf(x))
                p_map[ip] = np.sum(self.probs * comp)
        return p_map

    def draw_random_fluxes(self, n, alpha=-0.7, lf_thr_low=-3.5):
        msk = self.log_flux >= lf_thr_low
        lf_ax = self.log_flux[msk]
        p_ax = self.probs[msk]
        p_ax /= np.sum(p_ax)
        lf = np.random.choice(lf_ax, size=n, p=p_ax)
        lf += self.dlog_flux * (np.random.random(n)-0.5)
        # Extrapolate to 144 MHz
        # Assumption: I_nu = I_1400 * (nu / 1400)^alpha
        if alpha != 0:
            lf += alpha * np.log10(144. / 1400.)
        return lf


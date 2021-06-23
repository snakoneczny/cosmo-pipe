from astropy.io import fits
import healpy as hp
import numpy as np
from .mask import Mask
import os
import pymaster as nmt
import yaml
from scipy.interpolate import interp1d
import sacc


class ClCalculator(object):
    def __init__(self, config):
        self.config = config
        self.nside = self.config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.n_iter = self.config.get('n_iter', 0)
        self.cls_compute = self.config.get('compute', ['gg', 'gk', 'kk'])
        self.outdir = config['output_dir']
        os.system('mkdir -p ' + self.outdir)
        fname_dump = os.path.join(self.outdir, 'params.yml')
        with open(fname_dump, 'w') as f:
            yaml.dump(self.config, f)
        self.toe_cls = self._get_toeplitz_dict('cls')
        self.toe_cov = self._get_toeplitz_dict('cov')

        self.cat = None
        self.msk_d = None
        self.map_d = None
        self.f_d = None
        self.msk_k = None
        self.map_k = None
        self.f_k = None
        self.wsp = {'gg': None, 'gk': None, 'kk': None}
        self.bpw = None
        self.cls = {'gg': None, 'gk': None, 'kk': None}
        self.cov = {'gggg': None, 'gggk': None,
                    'ggkk': None, 'gkgk': None,
                    'gkkk': None, 'kkkk': None}
        self.sacc = None
        self.sacc_noise = None

    def _get_toeplitz_dict(self, section):
        d = {'l_toeplitz': -1,
             'l_exact': -1,
             'dl_band': -1}
        if 'toeplitz' in self.config:
            if section in self.config['toeplitz']:
                dd = self.config['toeplitz'][section]
                for k in d:
                    d[k] = dd.get(k, -1)
        return d

    def _get_fname_mask(self, typ='g'):
        if typ == 'g':
            return os.path.join(self.config['output_dir'],
                                'mask_delta.fits.gz')

    def get_bandpowers(self):
        if self.bpw is None:
            d = self.config['bandpowers']
            if d['type'] == 'linlog':  # Hybrid binning
                # Set up linear part
                l_edges_lin = np.linspace(2, d['lsplit'],
                                          (d['lsplit']-2)//d['nlb_lin']+1)
                l_edges_lin = l_edges_lin.astype(int)
                # Set up log part
                l_edges_log = np.unique(np.logspace(np.log10(d['lsplit']),
                                                    np.log10(3*self.nside-1),
                                                    d['nb_log']).astype(int))
                # Join
                l_edges = np.concatenate((l_edges_lin, l_edges_log[1:]))

                # Give bandpower indices and weights to each multipole
                larr = np.arange(3*self.nside)
                weights = np.ones(len(larr))
                bpws = -1+np.zeros(len(larr), dtype=int)
                for i in range(len(l_edges)-1):
                    bpws[l_edges[i]:l_edges[i+1]] = i

                # Create binning scheme
                self.bpw = nmt.NmtBin(self.nside, ells=larr, bpws=bpws, weights=weights)
            elif d['type'] == 'lin':  # Check if using linear binning
                # Create binning scheme
                self.bpw = nmt.NmtBin(self.nside, nlb=d['nlb'])
            else:
                raise ValueError("Unrecognised binning scheme "+d['type'])
        return self.bpw

    def get_catalog(self):
        if self.cat is None:
            I_thr = self.config['lotss']['I_thr']
            q_thr = self.config['lotss']['q_thr']
            self.cat = fits.open(self.config['lotss']['fname_catalog'])[1].data
            msk = ((self.cat['Total_flux'] > I_thr) &
                   (self.cat['Total_flux']/self.cat['E_Total_flux'] >= q_thr))
            self.cat = self.cat[msk]
        return self.cat

    def get_delta_mask(self):
        if self.msk_d is None:
            msk = Mask(self.config['lotss']['fname_rms'],
                       self.config['lotss']['I_thr'],
                       self.config['lotss']['q_thr'],
                       self._get_fname_mask(typ='g'),
                       from_cat=self.config['lotss'].get('mask_from_catalog', False),
                       fname_cat=self.config['lotss']['fname_catalog'])
            self.msk_d = msk.get_mask()
            self.msk_d = hp.ud_grade(self.msk_d, nside_out=self.nside)
            self.msk_d = self.msk_d / np.amax(self.msk_d)
            msk_thr = self.config['lotss'].get('mask_threshold', 0.)
            self.msk_d[self.msk_d <= msk_thr] = 0
        return self.msk_d

    def get_delta_map(self):
        if self.map_d is None:
            cat = self.get_catalog()
            ipix = hp.ang2pix(self.nside, cat['RA'], cat['DEC'], lonlat=True)
            nc = np.bincount(ipix, minlength=self.npix)
            msk = self.get_delta_mask()
            self.mean_n = np.sum(nc[msk > 0]) / np.sum(msk[msk > 0])
            self.map_d = np.zeros_like(msk)
            self.map_d[msk > 0] = nc[msk > 0] / (self.mean_n * msk[msk > 0]) - 1
        return self.map_d

    def get_delta_field(self):
        if self.f_d is None:
            self.get_delta_mask()
            self.get_delta_map()
            self.f_d = nmt.NmtField(self.msk_d, [self.map_d],
                                    n_iter=self.n_iter)
        return self.f_d

    def get_kappa_mask(self):
        if self.msk_k is None:
            self.msk_k = hp.read_map(self.config['kappa']['fname_mask'],
                                     verbose=False)
            self.msk_k = hp.ud_grade(self.msk_k, nside_out=self.nside)
        return self.msk_k

    def get_kappa_map(self):
        if self.map_k is None:
            self.map_k = hp.read_map(self.config['kappa']['fname_map'],
                                     verbose=False)
            self.map_k = hp.ud_grade(self.map_k, nside_out=self.nside)
        return self.map_k

    def get_kappa_field(self):
        if self.f_k is None:
            self.get_kappa_mask()
            self.get_kappa_map()
            self.f_k = nmt.NmtField(self.msk_k, [self.map_k],
                                    n_iter=self.n_iter)
        return self.f_k

    def get_clfile_name(self, typ, prefix, ext):
        return os.path.join(self.config['output_dir'],
                            f'{prefix}_{typ}.{ext}')

    def get_workspace(self, typ):
        if self.wsp[typ] is None:
            fname = self.get_clfile_name(typ, 'wsp', 'fits')
            b = self.get_bandpowers()
            self.wsp[typ] = nmt.NmtWorkspace()
            if not os.path.isfile(fname):
                if 'g' in typ:
                    self.get_delta_field()
                if 'k' in typ:
                    self.get_kappa_field()
                fl = {'g': self.f_d, 'k': self.f_k}
                t1, t2 = typ
                self.wsp[typ].compute_coupling_matrix(fl[t1],
                                                      fl[t2],
                                                      b, **(self.toe_cls))
                self.wsp[typ].write_to(fname)
            else:
                self.wsp[typ].read_from(fname)
        return self.wsp[typ]

    def get_nl_coupled(self, typ):
        if typ == 'gg':
            self.get_delta_map()
            ndens = self.mean_n * self.npix / (4*np.pi)
            return np.mean(self.msk_d) * np.ones([1, 3*self.nside]) / ndens
        elif typ == 'gk':
            return np.zeros([1, 3*self.nside])
        elif typ == 'kk':
            l, nl, _ = np.loadtxt(self.config['kappa']['fname_noise'], unpack=True)
            nlf = interp1d(l, nl, bounds_error=False,
                           fill_value=(nl[0], 0))
            nll = nlf(np.arange(3*self.nside))
            nll[:2] = 0
            self.get_kappa_mask()
            return np.array([nll]) * np.mean(self.msk_k**2)

    def get_cl(self, typ):
        if self.cls[typ] is None:
            fname = self.get_clfile_name(typ, 'cl', 'npz')
            if not os.path.isfile(fname):
                if 'g' in typ:
                    self.get_delta_field()
                if 'k' in typ:
                    self.get_kappa_field()
                fl = {'g': self.f_d, 'k': self.f_k}
                t1, t2 = typ
                wsp = self.get_workspace(typ)
                cl = wsp.decouple_cell(nmt.compute_coupled_cell(fl[t1],
                                                                fl[t2]))
                leff = self.bpw.get_effective_ells()
                nlc = self.get_nl_coupled(typ)
                nl = wsp.decouple_cell(nlc)
                self.cls[typ] = {'ls': leff,
                                 'cl': cl,
                                 'nl': nl,
                                 'nlc': nlc,
                                 'win': wsp.get_bandpower_windows()}
                np.savez(fname, **(self.cls[typ]))
            else:
                d = np.load(fname)
                self.cls[typ] = {k: d[k]
                                 for k in ['ls', 'cl', 'nl', 'nlc', 'win']}
        return self.cls[typ]

    def get_cls(self):
        for t in self.cls_compute:
            self.get_cl(t)
        return self.cls

    def get_covariance_workspace(self, typ1, typ2):
        tttt = typ1 + typ2
        fname = self.get_clfile_name(tttt, 'cwsp', 'fits')
        cwsp = nmt.NmtCovarianceWorkspace()
        if not os.path.isfile(fname):
            if 'g' in tttt:
                self.get_delta_field()
            if 'k' in tttt:
                self.get_kappa_field()
            fl = {'g': self.f_d, 'k': self.f_k}
            t1, t2, t3, t4 = tttt
            cwsp.compute_coupling_coefficients(fl[t1], fl[t2],
                                               fl[t3], fl[t4],
                                               **(self.toe_cov))
            cwsp.write_to(fname)
        else:
            cwsp.read_from(fname)
        return cwsp

    def get_interpolated_cl(self, typ):
        lev = np.arange(3*self.nside)
        # Swap if needed
        if typ not in self.cls:
            ttyp = typ[::-1]
        else:
            ttyp = typ
        d = self.get_cl(ttyp)
        clf = interp1d(d['ls'], d['cl'][0],
                       bounds_error=False,
                       fill_value=(d['cl'][0][0],
                                   d['cl'][0][-1]))
        return np.array([clf(lev)])

    def get_covariance(self, typ1, typ2):
        tttt = typ1 + typ2
        if self.cov[tttt] is None:
            fname = self.get_clfile_name(tttt, 'cov', 'npz')
            if not os.path.isfile(fname):
                t1, t2, t3, t4 = tttt
                clt = {}
                cwsp = self.get_covariance_workspace(typ1, typ2)
                w1 = self.get_workspace(typ1)
                w2 = self.get_workspace(typ2)
                for tt in [t1+t3, t1+t4, t2+t3, t2+t4]:
                    clt[tt] = self.get_interpolated_cl(tt)
                self.cov[tttt] = nmt.gaussian_covariance(cwsp,
                                                         0, 0, 0, 0,
                                                         clt[t1+t3],
                                                         clt[t1+t4],
                                                         clt[t2+t3],
                                                         clt[t2+t4],
                                                         w1, wb=w2)
                np.savez(fname, cov=self.cov[tttt])
            else:
                d = np.load(fname)
                self.cov[tttt] = d['cov']
        return self.cov[tttt]

    def get_covariances(self):
        for i1, typ1 in enumerate(self.cls_compute):
            for typ2 in self.cls_compute[i1:]:
                self.get_covariance(typ1, typ2)

    def get_everything(self):
        if self.sacc is None:
            fname = os.path.join(self.config['output_dir'],
                                 f'cls_all.fits')
            fname_n = os.path.join(self.config['output_dir'],
                                   f'cls_noise.fits')
            if not os.path.isfile(fname):
                self.get_cls()
                self.get_covariances()
                s = sacc.Sacc()
                sn = sacc.Sacc()
                # Add tracers
                s.add_tracer('NZ', 'LoTSS',
                             quantity='galaxy_density',
                             spin=0,
                             z=np.linspace(0, 5, 512),
                             nz=np.ones(512))  # We will need a better N(z)
                s.add_tracer('Map', 'Planck18',
                             quantity='cmb_convergence',
                             spin=0, ell=np.arange(3*self.nside),
                             beam=np.ones(3*self.nside))
                sn.add_tracer('NZ', 'LoTSS',
                              quantity='galaxy_density',
                              spin=0,
                              z=np.linspace(0, 5, 512),
                              nz=np.ones(512))  # We will need a better N(z)
                sn.add_tracer('Map', 'Planck18',
                              quantity='cmb_convergence',
                              spin=0, ell=np.arange(3*self.nside),
                              beam=np.ones(3*self.nside))
                              
                order = []
                nls = []
                ells = np.arange(3*self.nside)
                if 'gg' in self.cls_compute:
                    d = self.cls['gg']
                    wins = sacc.BandpowerWindow(ells, d['win'][0, :, 0, :].T)
                    s.add_ell_cl('cl_00', 'LoTSS', 'LoTSS',
                                 d['ls'], d['cl'][0]-d['nl'][0],
                                 window=wins)
                    sn.add_ell_cl('cl_00', 'LoTSS', 'LoTSS',
                                  d['ls'], d['nl'][0])
                    order.append('gg')
                    nls.append(len(d['ls']))
                if 'gk' in self.cls_compute:
                    d = self.cls['gk']
                    wins = sacc.BandpowerWindow(ells, d['win'][0, :, 0, :].T)
                    s.add_ell_cl('cl_00', 'LoTSS', 'Planck18',
                                 d['ls'], d['cl'][0]-d['nl'][0],
                                 window=wins)
                    sn.add_ell_cl('cl_00', 'LoTSS', 'Planck18',
                                  d['ls'], d['nl'][0])
                    order.append('gk')
                    nls.append(len(d['ls']))
                if 'kk' in self.cls_compute:
                    d = self.cls['kk']
                    wins = sacc.BandpowerWindow(ells, d['win'][0, :, 0, :].T)
                    s.add_ell_cl('cl_00', 'Planck18', 'Planck18',
                                 d['ls'], d['cl'][0]-d['nl'][0],
                                 window=wins)
                    sn.add_ell_cl('cl_00', 'Planck18', 'Planck18',
                                  d['ls'], d['nl'][0])
                    order.append('kk')
                    nls.append(len(d['ls']))
                ncls = len(nls)
                cov = np.zeros([ncls, nls[0], ncls, nls[0]])
                for i1, t1 in enumerate(order):
                    for i2, t2 in enumerate(order):
                        if t1+t2 in self.cov:
                            cv = self.cov[t1+t2]
                        else:
                            cv = self.cov[t2+t1].T
                        cov[i1, :, i2, :] = cv
                cov = cov.reshape([ncls*nls[0], ncls*nls[0]])
                s.add_covariance(cov)
                s.save_fits(fname)
                sn.save_fits(fname_n)
                self.sacc = s
                self.sacc_noise = sn
            else:
                self.sacc = sacc.Sacc.load_fits(fname)
                self.sacc_noise = sacc.Sacc.load_fits(fname_n)
        return self.sacc, self.sacc_noise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    args = parser.parse_args()

    with open(args.INPUT) as f:
        config = yaml.safe_load(f)

    clc = ClCalculator(config)
    clc.get_everything()

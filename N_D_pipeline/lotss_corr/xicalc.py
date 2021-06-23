from astropy.io import fits
import numpy as np
from .mask import Mask
import os
import treecorr as tcr
import yaml
import sacc


class XiCalculator(object):
    def __init__(self, config):
        self.config = config
        self.xi_compute = self.config.get('compute', ['gg'])
        self.outdir = config['output_dir']
        os.system('mkdir -p ' + self.outdir)
        fname_dump = os.path.join(self.outdir, 'params.yml')
        with open(fname_dump, 'w') as f:
            yaml.dump(self.config, f)

        self.n_data = 0
        self.cat = None
        self.ran = None
        self.msk_d = None
        self.xis = {'gg': None}
        self.cov = {'gggg': None}
        self.sacc = None

    def _get_npatches(self):
        b = self._get_binning()
        return b['nbins']*10

    def _get_binning(self):
        bn = {'nbins': 20,
              'min_sep': 0.01,
              'max_sep': 2.,
              'sep_units': 'deg'}
        if 'treecorr' in self.config:
            for k in self.config['treecorr']:
                bn[k] = self.config['treecorr'][k]

        return bn

    def _get_fname_mask(self, typ='g'):
        if typ == 'g':
            return os.path.join(self.config['output_dir'],
                                'mask_delta.fits.gz')

    def get_catalog(self):
        if self.cat is None:
            I_thr = self.config['lotss']['I_thr']
            q_thr = self.config['lotss']['q_thr']
            cat = fits.open(self.config['lotss']['fname_catalog'])[1].data
            msk = ((cat['Total_flux'] > I_thr) &
                   (cat['Total_flux']/cat['E_Total_flux'] >= q_thr))
            cat = cat[msk]
            self.n_data = len(cat)
            self.cat = tcr.Catalog(ra=cat['RA'], dec=cat['DEC'],
                                   ra_units='deg', dec_units='deg',
                                   npatch=self._get_npatches())
        return self.cat

    def get_delta_mask(self):
        if self.msk_d is None:
            self.msk_d = Mask(self.config['lotss']['fname_rms'],
                              self.config['lotss']['I_thr'],
                              self.config['lotss']['q_thr'],
                              self._get_fname_mask(typ='g'),
                              from_cat=self.config['lotss'].get('mask_from_catalog', False),
                              fname_cat=self.config['lotss']['fname_catalog'])
            self.msk_d.get_mask()
        return self.msk_d

    def get_random(self):
        if self.ran is None:
            self.get_catalog()
            self.get_delta_mask()
            nrand = int(self.n_data*self.config['lotss'].get('nr_factor', 10))
            ran = self.msk_d.get_random_catalog(nrand)
            self.ran = tcr.Catalog(ra=ran['RA'], dec=ran['DEC'],
                                   ra_units='deg', dec_units='deg',
                                   patch_centers=self.cat.patch_centers)
        return self.ran

    def get_xifile_name(self, typ, prefix, ext):
        return os.path.join(self.config['output_dir'],
                            f'{prefix}_{typ}.{ext}')

    def get_xi_gg(self):
        if self.xis['gg'] is None:
            fname = self.get_xifile_name('gg', 'xi', 'npz')
            if not os.path.isfile(fname):
                self.get_catalog()
                self.get_delta_mask()
                self.get_random()
                bn = self._get_binning()
                dd = tcr.NNCorrelation(config=bn, var_method='jackknife')
                dd.process(self.cat)
                dr = tcr.NNCorrelation(config=bn)
                dr.process(self.cat, self.ran)
                rr = tcr.NNCorrelation(config=bn)
                rr.process(self.ran)
                xi, varxi = dd.calculateXi(rr, dr)
                self.xis['gg'] = {'th': dd.rnom,
                                  'th_mean': dd.meanr,
                                  'DD': dd.npairs,
                                  'DR': dr.npairs,
                                  'RR': rr.npairs,
                                  'xi': xi,
                                  'cov': dd.cov,
                                  'var': varxi}
                np.savez(fname, **(self.xis['gg']))
            else:
                d = np.load(fname)
                self.xis['gg'] = {k: d[k]
                                  for k in ['th', 'th_mean',
                                            'DD', 'DR', 'RR',
                                            'xi', 'cov', 'var']}
        return self.xis['gg']

    def get_xis(self):
        for t in self.xis_compute:
            if t == 'gg':
                self.get_xi_gg()
            else:
                raise NotImplementedError(f"{t} correlation not ready")
        return self.xis

    def get_covariance(self, typ1, typ2):
        raise NotImplementedError("Covariance not ready")

    def get_covariances(self):
        raise NotImplementedError("Covariance not ready")

    def get_everything(self):
        raise NotImplementedError("Sacc not ready")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Xis and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    args = parser.parse_args()

    with open(args.INPUT) as f:
        config = yaml.safe_load(f)

    xic = XiCalculator(config)
    xic.get_xi_gg()

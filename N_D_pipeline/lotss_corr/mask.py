import healpy as hp
from scipy.interpolate import interp1d
import os
from astropy.io import fits
from .flux_pdf import FluxPDF
import numpy as np


class Mask(object):
    def __init__(self, fname_rms, Icut, q, fname_mask, from_cat=False, fname_cat=None):
        self.fname_rms = fname_rms
        self.Icut = Icut
        self.q = q
        self.fname_mask = fname_mask
        self.from_cat = from_cat
        self.fname_cat = fname_cat

        self.mask = None

    def _rms_to_p(self, rms_map):
        fpdf = FluxPDF()
        rms_min = np.amin(rms_map[rms_map > 0])
        rms_max = np.amax(rms_map[rms_map > 0])
        rms_arr = np.geomspace(rms_min, rms_max, 2048)
        p_arr = fpdf.compute_p_values(self.q, rms_arr, self.Icut)
        pf = interp1d(np.log(rms_arr),
                      p_arr,
                      fill_value=(p_arr[0], p_arr[-1]),
                      bounds_error=False)
        p_map = np.zeros_like(rms_map)
        p_map[rms_map > 0] = pf(np.log(rms_map[rms_map > 0]))
        p_map /= np.amax(p_map)
        return p_map
        
    def _compute_mask(self):
        if not os.path.isfile(self.fname_mask):
            if self.from_cat:
                msk, p_map = self._compute_mask_from_catalog()
            else:
                msk, p_map = self._compute_mask_from_rmsmap()
            hp.write_map(self.fname_mask, [p_map*msk, p_map, msk],
                         overwrite=True,
                         column_names=['p_map', 'p_map_comp', 'p_map_geom'])
        return hp.read_map(self.fname_mask, field=0, verbose=False)

    def _compute_mask_from_catalog(self):
        # Read mask
        msk = hp.read_map(self.fname_rms, field=0, verbose=False)

        # Make RMS map from catalog
        nside_lo = 256
        npix_lo = hp.nside2npix(nside_lo)
        cat = fits.open(self.fname_cat)[1].data
        ipix = hp.ang2pix(nside_lo, cat['RA'], cat['DEC'], lonlat=True)
        nc = np.bincount(ipix, minlength=npix_lo)
        rms = np.bincount(ipix, minlength=npix_lo, weights=cat['Isl_rms'])
        rms[nc > 0] = rms[nc > 0] / nc[nc > 0]
        rms[nc <= 0] = 0

        # Transform to p
        p_map = self._rms_to_p(rms)
        nside = hp.npix2nside(len(msk))
        p_map = hp.ud_grade(p_map, nside_out=nside)
        return msk, p_map

    def _compute_mask_from_rmsmap(self):
        # Read mask
        msk = hp.read_map(self.fname_rms, field=0, verbose=False)

        # Read rms map
        rms_map = hp.read_map(self.fname_rms, field=1, verbose=False)*1E3  # in mJy

        # Transform to p
        p_map = self._rms_to_p(rms_map)
        return msk, p_map

    def get_mask(self):
        if self.mask is None:
            self.mask = self._compute_mask()
        return self.mask

import numpy as np


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

import numpy as np


class Pointings(object):
    def __init__(self, fname_pointings, prefix_out):
        self.prefix_out = prefix_out
        self.data = np.genfromtxt(fname_pointings,
                                  dtype='S256,<f8,<f8,S256,S256,S256,S256,S256',
                                  names=['name','ra','dec','fr_mosaic',
                                         'fr_rms','fr_res','lr_mosaic','cat'])

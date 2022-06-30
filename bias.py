import numpy as np


def get_sherwin_qso_bias():
    # Values from the Shen et al. 2009
    z_arr = np.array([0.5, 1.13, 1.68, 2.18, 3.17, 3.84])
    b_arr = np.array([1.32, 2.31, 2.96, 4.69, 7.76, 12.96])

    # Amplitude from Sherwin et al. 2012: 1.02 ± 0.24
    # https: // arxiv.org / pdf / 1207.4543.pdf
    b_err = 0.24 * b_arr
    b_arr = 1.02 * b_arr

    # Make interpolation
    # f = interp1d(z_arr_qso, b_arr_qso)
    # z_arr = None
    # bias_arr = None * f(z_arr)

    return z_arr, b_arr, b_err

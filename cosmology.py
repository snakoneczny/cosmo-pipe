import math

import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import pyccl as ccl


def get_covariance_gg_gg(field, gg_theory, gg_workspace, n_bands, covariance_workspace=None):
    if not covariance_workspace:
        covariance_workspace = nmt.NmtCovarianceWorkspace()
        covariance_workspace.compute_coupling_coefficients(field, field, field, field)

    covar_gg_gg = nmt.gaussian_covariance(
        covariance_workspace,
        0, 0, 0, 0,  # Spins of the 4 fields
        [gg_theory],  # GG
        [gg_theory],  # GG
        [gg_theory],  # GG
        [gg_theory],  # GG
        gg_workspace,
        wb=gg_workspace
    )
    covar_gg_gg = covar_gg_gg.reshape([n_bands, 1, n_bands, 1])
    return covar_gg_gg[:, 0, :, 0]


def get_theory_clustering_correlation(l_arr, z_arr, n_arr, bias_arr):
    cosmology = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96)
    number_counts_tracer = ccl.NumberCountsTracer(cosmology, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr))
    correlation = ccl.angular_cl(cosmology, number_counts_tracer, number_counts_tracer, l_arr)
    return correlation


def get_auto_correlation(map, mask, nside, l_max=None, normalize_map=True, with_shot_noise=True, mask_aposize=1.0,
                         ells_per_bandpower=4):
    # Get shot noise for discrete objects
    shot_noise = 0
    if with_shot_noise:
        shot_noise = get_shot_noise(map, mask)
        print('Shot noise: {}'.format(shot_noise))
    # Normalize counts for discrete objects
    if normalize_map:
        sky_mean = np.mean(map[np.nonzero(mask)])
        map_to_correlate = (map - sky_mean) / sky_mean
    else:
        map_to_correlate = map
    # Apodize mask  # TODO: should the apodization influence any other mask usage?
    if mask_aposize:
        mask = nmt.mask_apodization(mask, mask_aposize, apotype='Smooth')
    # Get field
    field = nmt.NmtField(mask, [map_to_correlate])
    # Initialize binning scheme
    if l_max:
        ells = np.arange(l_max, dtype='int32')
        weights = 1.0 / ells_per_bandpower * np.ones_like(ells)
        bandpower_indices = -1 + np.zeros_like(ells)
        i = 0
        while ells_per_bandpower * (i + 1) + 2 < l_max:
            bandpower_indices[ells_per_bandpower * i + 2:ells_per_bandpower * (i + 1) + 2] = i
            i += 1
        binning = nmt.NmtBin(nside=nside, bpws=bandpower_indices, ells=ells, weights=weights)
    else:
        binning = nmt.NmtBin.from_nside_linear(nside, ells_per_bandpower)

    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, binning)
    cl_coupled = nmt.compute_coupled_cell(field, field)
    cl_decoupled = workspace.decouple_cell(cl_coupled)

    # Substract shot noise
    cl_coupled = cl_coupled[0] - shot_noise
    cl_decoupled = cl_decoupled[0] - shot_noise

    # Return correlation with substracted shot noise
    return cl_coupled, cl_decoupled, workspace, binning, shot_noise, field


def plot_correlation(binning, correlation, model_correlation=None, covariance_matrix=None, x_max=None, y_min=None,
                     x_scale='linear', y_scale='linear'):
    ell_arr = binning.get_effective_ells()
    to_plot = np.fabs(correlation)

    if covariance_matrix is not None:
        y_err = [covariance_matrix[i, i] for i in range(covariance_matrix.shape[0])]
        plt.errorbar(ell_arr, to_plot, yerr=y_err, fmt='ob', label='GG', markersize=2)
    else:
        plt.plot(ell_arr, to_plot, 'ob', label='GG', markersize=2)

    if model_correlation is not None:
        plt.plot(ell_arr, model_correlation, 'r', label='theory', markersize=2)

    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.xlim(xmax=x_max)
    plt.ylim(ymin=y_min)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$C_\\ell$', fontsize=16)
    plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
    plt.show()


def get_shot_noise(map, mask):
    sky_frac = np.sum(mask) / np.shape(mask)[0]
    n_obj = np.sum(map[np.nonzero(mask)])
    shot_noise = 4.0 * math.pi * sky_frac / n_obj
    return shot_noise

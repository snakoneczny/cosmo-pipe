import math

import numpy as np
import pymaster as nmt
import pyccl as ccl


def get_chi_squared(data, theory, covariance):
    diff = data - theory
    cov_inv = np.linalg.inv(covariance)
    return diff.dot(cov_inv).dot(diff)


def get_covariance_matrix(field, cl_theory, cl_workspace, n_bands, covariance_workspace=None):
    if not covariance_workspace:
        covariance_workspace = nmt.NmtCovarianceWorkspace()
        covariance_workspace.compute_coupling_coefficients(field, field, field, field)

    covariance = nmt.gaussian_covariance(
        covariance_workspace,
        0, 0, 0, 0,  # Spins of the 4 fields
        [cl_theory],  # GG
        [cl_theory],  # GG
        [cl_theory],  # GG
        [cl_theory],  # GG
        cl_workspace,
        wb=cl_workspace
    )
    covariance = covariance.reshape([n_bands, 1, n_bands, 1])
    return covariance[:, 0, :, 0]


def get_theory_correlations(l_arr, z_arr, n_arr, bias, scale_bias=False):
    cosmology = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96)
    bias_arr = bias * np.ones(len(z_arr))
    if scale_bias:
        bias_arr = bias_arr / ccl.growth_factor(cosmology, 1. / (1 + z_arr))
    number_counts_tracer = ccl.NumberCountsTracer(cosmology, has_rsd=False, dndz=(z_arr, n_arr), bias=(z_arr, bias_arr))
    lensing_tracer = ccl.WeakLensingTracer(cosmology, dndz=(z_arr, n_arr))
    cmb_lensing_tracer = ccl.CMBLensingTracer(cosmology, 1091)
    cl_gg = ccl.angular_cl(cosmology, number_counts_tracer, number_counts_tracer, l_arr)
    cl_gk = ccl.angular_cl(cosmology, lensing_tracer, cmb_lensing_tracer, l_arr)
    cl_kk = ccl.angular_cl(cosmology, cmb_lensing_tracer, cmb_lensing_tracer, l_arr)
    return cl_gg, cl_gk, cl_kk


def get_data_correlations(map_counts, map_g, mask_g, map_cmb_k, mask_cmb_k, nside, l_max=None, with_shot_noise=True,
                          mask_aposize=1.0, ells_per_bandpower=4):
    # Get shot noise for discrete objects
    shot_noise = 0
    if with_shot_noise:
        shot_noise = get_shot_noise(map_counts, mask_g)
    # Apodize mask  # TODO: should the apodization influence any other mask usage?
    if mask_aposize:
        mask_g = nmt.mask_apodization(mask_g, mask_aposize, apotype='Smooth')
    # Get field
    field_g = nmt.NmtField(mask_g, [map_g])
    field_cmb_k = nmt.NmtField(mask_cmb_k, [map_cmb_k])
    # field_cmb_t = nmt.NmtField(mask_cmb, [map_cmb_t])
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

    # Get all correlations
    cl_coupled_gg, cl_decoupled_gg, workspace_gg = compute_master(field_g, field_g, binning)
    cl_coupled_gk, cl_decoupled_gk, workspace_gk = compute_master(field_g, field_cmb_k, binning)
    cl_coupled_kk, cl_decoupled_kk, workspace_kk = compute_master(field_cmb_k, field_cmb_k, binning)

    # Substract shot noise
    cl_coupled_gg = cl_coupled_gg - shot_noise
    cl_decoupled_gg = cl_decoupled_gg - shot_noise

    # Return correlation with substracted shot noise
    return field_g, field_cmb_k, cl_coupled_gg, cl_decoupled_gg, workspace_gg, cl_coupled_gk, cl_decoupled_gk, \
           workspace_gk, cl_coupled_kk, cl_decoupled_kk, workspace_kk, binning, shot_noise


def get_shot_noise(map, mask):
    sky_frac = np.sum(mask) / np.shape(mask)[0]
    n_obj = np.sum(map[np.nonzero(mask)])
    shot_noise = 4.0 * math.pi * sky_frac / n_obj
    return shot_noise


def compute_master(field_a, field_b, binning):
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field_a, field_b, binning)
    cl_coupled = nmt.compute_coupled_cell(field_a, field_b)
    cl_decoupled = workspace.decouple_cell(cl_coupled)
    return cl_coupled[0], cl_decoupled[0], workspace

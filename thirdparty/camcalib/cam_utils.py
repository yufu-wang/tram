# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np
from scipy.stats import norm
import torch.nn.functional as F
# from pare.models.layers.softargmax import softargmax1d


def _softmax(tensor, temperature, dim=-1):
    return F.softmax(tensor * temperature, dim=dim)


def softargmax1d(
        heatmaps,
        temperature=None,
        normalize_keypoints=True,
):
    dtype, device = heatmaps.dtype, heatmaps.device
    if temperature is None:
        temperature = torch.tensor(1.0, dtype=dtype, device=device)
    batch_size, num_channels, dim = heatmaps.shape
    points = torch.arange(0, dim, device=device, dtype=dtype).reshape(1, 1, dim).expand(batch_size, -1, -1)
    # y = torch.arange(0, height, device=device, dtype=dtype).reshape(1, 1, height, 1).expand(batch_size, -1, -1, width)
    # Should be Bx2xHxW

    # points = torch.cat([x, y], dim=1)
    normalized_heatmap = _softmax(
        heatmaps.reshape(batch_size, num_channels, -1),
        temperature=temperature.reshape(1, -1, 1),
        dim=-1)

    # Should be BxJx2
    keypoints = (normalized_heatmap.reshape(batch_size, -1, dim) * points).sum(dim=-1)

    if normalize_keypoints:
        # Normalize keypoints to [-1, 1]
        keypoints = (keypoints / (dim - 1) * 2 - 1)

    return keypoints, normalized_heatmap.reshape(
        batch_size, -1, dim)


def get_bins(minval, maxval, sigma, alpha, beta, kappa):
    """Remember, bin 0 = below value! last bin mean >= maxval"""
    x = np.linspace(minval, maxval, 255)

    rv = norm(0, sigma)
    pdf = rv.pdf(x)
    pdf /= (pdf.max())
    pdf *= alpha
    pdf = pdf.max()*beta - pdf
    cumsum = np.cumsum(pdf)
    cumsum = cumsum / cumsum.max() * kappa
    cumsum -= cumsum[pdf.size//2]

    return cumsum


pitch_bins = np.linspace(-0.6, 0.6, 255)
pitch_bins_centers = pitch_bins.copy()
pitch_bins_centers[:-1] += np.diff(pitch_bins_centers)/2
pitch_bins_centers = np.append(pitch_bins_centers, pitch_bins[-1])

horizon_bins = np.linspace(-0.5, 1.5, 255)
horizon_bins_centers = horizon_bins.copy()
horizon_bins_centers[:-1] += np.diff(horizon_bins_centers)/2
horizon_bins_centers = np.append(horizon_bins_centers, horizon_bins[-1])

roll_bins = get_bins(-np.pi/6, np.pi/6, 0.5, 0.04, 1.1, np.pi)
# roll_bins = get_bins(-np.pi/6, np.pi/6, 0.2, 0.04, 1.1, np.pi/3)
roll_bins_centers = roll_bins.copy()
roll_bins_centers[:-1] += np.diff(roll_bins_centers)/2
roll_bins_centers = np.append(roll_bins_centers, roll_bins[-1])

vfov_bins = np.linspace(0.2617, 2.1, 255)
vfov_bins_centers = vfov_bins.copy()
vfov_bins_centers[:-1] += np.diff(vfov_bins_centers)/2
vfov_bins_centers = np.append(vfov_bins_centers, vfov_bins[-1])

roll_new_bins = np.linspace(-0.6, 0.6, 255)
roll_new_bins_centers = roll_new_bins.copy()
roll_new_bins_centers[:-1] += np.diff(roll_new_bins_centers)/2
roll_new_bins_centers = np.append(roll_new_bins_centers, roll_new_bins[-1])


def bins2horizon(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return horizon_bins_centers[idxes]


def bins2pitch(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return pitch_bins_centers[idxes]


def bins2roll(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return roll_bins_centers[idxes]


def bins2vfov(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return vfov_bins_centers[idxes]


def vfov2soft_idx(angle):
    return angle_to_soft_idx(angle, min=np.min(vfov_bins), max=np.max(vfov_bins))


def pitch2soft_idx(angle):
    return angle_to_soft_idx(angle, min=np.min(pitch_bins), max=np.max(pitch_bins))


def roll2soft_idx(angle):
    return angle_to_soft_idx(angle, min=-0.6, max=0.6)


def angle_to_soft_idx(angle, min, max):
    return 2 * ((angle - min) / (max - min)) - 1


def soft_idx_to_angle(soft_idx, min, max):
    return (max - min) * ((soft_idx + 1) / 2) + min


def get_softargmax(pred):
    pred = pred.unsqueeze(1)  # (N, 1, 256)
    pred_argmax, _ = softargmax1d(pred, normalize_keypoints=True)  # (N, 1, 1)
    pred_argmax = pred_argmax.reshape(-1)
    return pred_argmax


@torch.no_grad()
def convert_preds_to_angles(pred_vfov, pred_pitch, pred_roll, loss_type='kl', return_type='torch', legacy=False):
    if loss_type in ('kl', 'ce'):
        pred_vfov = bins2vfov(pred_vfov)
        pred_pitch = bins2pitch(pred_pitch)
        pred_roll = bins2roll(pred_roll)
    elif loss_type in ('softargmax_l2', 'softargmax_biased_l2'):
        pred_vfov = soft_idx_to_angle(get_softargmax(pred_vfov),
                                      min=np.min(vfov_bins), max=np.max(vfov_bins))
        pred_pitch = soft_idx_to_angle(get_softargmax(pred_pitch),
                                       min=np.min(pitch_bins), max=np.max(pitch_bins))
        if not legacy:
            pred_roll = soft_idx_to_angle(get_softargmax(pred_roll), min=-0.6, max=0.6)
        else:
            pred_roll = bins2roll(pred_roll)

    if return_type == 'np' and isinstance(pred_vfov, torch.Tensor):
        return pred_vfov.cpu().numpy(), \
               pred_pitch.cpu().numpy(), \
               pred_roll.cpu().numpy()

    if return_type == 'torch' and isinstance(pred_vfov, np.ndarray):
        return torch.from_numpy(pred_vfov), torch.from_numpy(pred_pitch), torch.from_numpy(pred_roll)

    return pred_vfov, pred_pitch, pred_roll
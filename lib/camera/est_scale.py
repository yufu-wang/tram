import numpy as np
import cv2
import torch
from torchmin import minimize


def est_scale_iterative(slam_depth, pred_depth, iters=10, msk=None):
    """ Simple depth-align by iterative median and thresholding """
    s = pred_depth / slam_depth
    
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<10)
    s_est = s[robust]
    scale = np.median(s_est)
    scales_ = [scale]

    for _ in range(iters):
        slam_depth_0 = slam_depth * scale
        robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<10) * (0<pred_depth) * (pred_depth<10)
        s_est = s[robust]
        scale = np.median(s_est)
        scales_.append(scale)

    return scale


def est_scale_gmof(slam_depth, pred_depth, lr=1, sigma=0.5, iters=500, msk=None):
    """ Simple depth-align by robust least-square """
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<10)
    pm = torch.from_numpy(pred_depth[robust])
    sm = torch.from_numpy(slam_depth[robust])

    scale = torch.tensor([1.], requires_grad=True)
    optim = torch.optim.Adam([scale], lr=lr)
    losses = []
    for i in range(iters):
        loss = sm * scale - pm
        loss = gmof(loss, sigma=sigma).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    scale = scale.detach().cpu().item()

    return scale


def est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=None, 
                     far_thresh=10):
    """ Depth-align by iterative + robust least-square """
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # Stage 1: Iterative steps
    s = pred_depth / slam_depth

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<10)
    s_est = s[robust]
    scale = np.median(s_est)

    for _ in range(10):
        slam_depth_0 = slam_depth * scale
        robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (0<pred_depth) * (pred_depth<far_thresh)
        s_est = s[robust]
        scale = np.median(s_est)


    # Stage 2: Robust optimization
    robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (0<pred_depth) * (pred_depth<far_thresh)
    pm = torch.from_numpy(pred_depth[robust])
    sm = torch.from_numpy(slam_depth[robust])

    def f(x):
        loss = sm * x - pm
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([scale])
    result = minimize(f, x0,  method='bfgs')
    scale = result.x.detach().cpu().item()

    return scale


def scale_shift_align(smpl_depth, pred_depth, sigma=0.5):
    """ Align pred_depth to smpl depth """
    smpl = torch.from_numpy(smpl_depth)
    pred = torch.from_numpy(pred_depth)

    def f(x):
        loss = smpl - (pred * x[0] + x[1])
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([1., 0.])
    result = minimize(f, x0,  method='bfgs')
    scale_shift = result.x.detach().cpu().numpy()

    return scale_shift


def shift_align(smpl_depth, pred_depth, sigma=0.5):
    """ Align pred_depth to smpl depth by only shift """
    smpl = torch.from_numpy(smpl_depth)
    pred = torch.from_numpy(pred_depth)

    def f(x):
        loss = smpl - (pred + x)
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([0.])
    result = minimize(f, x0,  method='bfgs')
    scale_shift = result.x.detach().cpu().numpy()

    return scale_shift


def gmof(x, sigma=100):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from scipy.ndimage import gaussian_filter

from .tools import checkerboard_geometry
from lib.models.smpl import SMPL
from lib.utils.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion


def traj_filter(pred_vert_w, pred_j3d_w, sigma=3):
    """ Smooth the root trajetory (xyz) """
    root = pred_j3d_w[:, 0]
    root_smooth = torch.from_numpy(gaussian_filter(root, sigma=sigma, axes=0))

    pred_vert_w = pred_vert_w + (root_smooth - root)[:, None]
    pred_j3d_w = pred_j3d_w + (root_smooth - root)[:, None]
    return pred_vert_w, pred_j3d_w

def cam_filter(cam_r, cam_t, r_sigma=3, t_sigma=15):
    """ Smooth camera trajetory (SO3) """
    cam_q = matrix_to_quaternion(cam_r)
    r_smooth = torch.from_numpy(gaussian_filter(cam_q, sigma=r_sigma, axes=0))
    t_smooth = torch.from_numpy(gaussian_filter(cam_t, sigma=t_sigma, axes=0))

    r_smooth = r_smooth / r_smooth.norm(dim=1, keepdim=True)
    r_smooth = quaternion_to_matrix(r_smooth)
    return r_smooth,  t_smooth

def fit_to_ground_easy(pred_vert_w, pred_j3d_w, idx=-1):
    """
    Transform meshes to a y-up ground plane
    pred_vert_w (B, N, 3)
    pred_j3d_w (B, J, 3)
    """
    # fit a ground plane
    toes = pred_j3d_w[:, [10, 11]]
    toes = toes.reshape(1, -1, 3)
    pl = fit_plane(toes, idx)

    normal = pl[0, :3]
    offset = pl[0, -1]
    person_up = (pred_j3d_w[:, 3] - pred_j3d_w[:, 0]).mean(dim=0)
    if (person_up @ normal).sign() < 0:
        normal = -normal
        offset = -offset

    yup = torch.tensor([0, 1., 0])
    R = align_a2b(normal, yup)

    pred_vert_gr = torch.einsum('ij,bnj->bni', R, pred_vert_w)
    pred_j3d_gr = torch.einsum('ij,bnj->bni', R, pred_j3d_w)
    offset = pred_vert_gr[:, :, 1].min()

    return R, offset

def fit_to_ground_spine(pred_vert_w, pred_j3d_w, start=0, end=15, lowest=None):
    """
    Transform to a y-up ground plane using the spine direction
    pred_vert_w (B, N, 3)
    pred_j3d_w (B, J, 3)
    """
    # fit a ground plane
    person_up = (pred_j3d_w[start:end, 6] - pred_j3d_w[start:end, 3]).mean(dim=0)
    person_up /= person_up.norm()
    yup = torch.tensor([0, 1., 0])
    R = align_a2b(person_up, yup)

    pred_vert_gr = torch.einsum('ij,bnj->bni', R, pred_vert_w)
    pred_j3d_gr = torch.einsum('ij,bnj->bni', R, pred_j3d_w)

    if lowest is None:
        lowest = end
    offset = pred_vert_gr[0:lowest, :, 1].min()
    
    pred_vert_gr[...,1] -= offset
    pred_j3d_gr[...,1] -= offset

    return pred_vert_gr, pred_j3d_gr

def fit_plane(points, idx=-1):
    """
    From SLAHMR
    :param points (*, N, 3)
    returns (*, 3) plane parameters (returns in normal * offset format)
    """
    *dims, N, D = points.shape
    mean = points.mean(dim=-2, keepdim=True)
    # (*, N, D), (*, D), (*, D, D)
    U, S, Vh = torch.linalg.svd(points - mean)
    normal = Vh[..., idx, :]  # (*, D)
    offset = torch.einsum("...ij,...j->...i", points, normal)  # (*, N)
    offset = offset.mean(dim=-1, keepdim=True)
    return torch.cat([normal, offset], dim=-1)

def get_floor_mesh(pred_vert_gr, z_start=0, z_end=-1, scale=1.5):
    """ Return the geometry of the floor mesh """
    verts = pred_vert_gr.clone()

    # Scale of the scene
    sx, sz = (verts.mean(1).max(0)[0] - verts.mean(1).min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item()) * scale

    # Center X
    cx = (verts.mean(1).max(0)[0] + verts.mean(1).min(0)[0])[[0]] / 2.0
    cx = cx.item()

    # Center Z: optionally using only a subsection
    verts = verts[z_start:z_end]
    cz = (verts.mean(1).max(0)[0] + verts.mean(1).min(0)[0])[[2]] / 2.0
    cz = cz.item()

    v, f, vc, fc = checkerboard_geometry(length=scale, c1=cx, c2=cz, up="y")
    vc = vc[:, :3] * 255
  
    return [v, f, vc]

def get_mesh_world(pred_smpl, pred_cam=None, scale=None):
    """ Transform smpl from canonical to world frame """
    smpl = SMPL()

    pred_rotmat = pred_smpl['pred_rotmat']
    pred_shape = pred_smpl['pred_shape']
    pred_trans = pred_smpl['pred_trans']

    pred = smpl(body_pose=pred_rotmat[:,1:], 
                global_orient=pred_rotmat[:,[0]], 
                betas=pred_shape, 
                transl=pred_trans.squeeze(),
                pose2rot=False, 
                default_smpl=True)
    pred_vert = pred.vertices
    pred_j3d = pred.joints[:, :24]

    if pred_cam is not None:
        pred_traj = pred_cam['traj']
        pred_camt = torch.tensor(pred_traj[:, :3]) * scale
        pred_camq = torch.tensor(pred_traj[:, 3:])
        pred_camr = quaternion_to_matrix(pred_camq[:,[3,0,1,2]])

        pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:,None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:,None]
    else:
        pred_vert_w = pred_vert
        pred_j3d_w = pred_j3d

    return pred_vert_w, pred_j3d_w
    
def align_a2b(a, b):
    # Find a rotation that align a to b
    v = torch.cross(a, b)
    # s = v.norm()
    c = torch.dot(a, b)
    R = torch.eye(3) + skew(v) + skew(v)@skew(v) * (1/(1+c))
    return R

def skew(a):
    v1, v2, v3 = a
    m = torch.tensor([[0, -v3, v2],
                      [v3, 0, -v1],
                      [-v2, v1, 0]]).float()
    return m

def vis_traj(traj_1, traj_2, savefolder, grid=5):
    """ Plot & compare the trajetories in the xy plane """
    os.makedirs(savefolder, exist_ok=True)

    for seq in traj_1:
        traj_gt = traj_1[seq]['gt']
        traj_1 = traj_1[seq]['pred']
        traj_w = traj_2[seq]['pred']

        vis_center = traj_gt[0]
        traj_1 = traj_1 - vis_center
        traj_w = traj_w - vis_center
        traj_gt = traj_gt - vis_center
        
        length = len(traj_gt)
        step = int(length/100)

        a1 = np.linspace(0.3, 0.90, len(traj_gt[0::step,0]))
        a2 = np.linspace(0.3, 0.90, len(traj_w[0::step,0]))

        plt.rcParams['figure.figsize']=4,3
        fig, ax = plt.subplots()
        colors = ['tab:green', 'tab:blue', 'tab:orange']
        ax.scatter(traj_gt[0::step,0], traj_gt[0::step,2], s=10, c='tab:grey', alpha=a1, edgecolors='none')
        ax.scatter(traj_w[0::step,0], traj_w[0::step,2], s=10, c='tab:blue', alpha=a2, edgecolors='none')
        ax.scatter(traj_1[0::step,0], traj_1[0::step,2], s=10, c='tab:orange', alpha=a1, edgecolors='none')
        ax.set_box_aspect(1)
        ax.set_aspect(1, adjustable='datalim')
        ax.grid(linewidth=0.4, linestyle='--')

        ax.tick_params(axis='both', labelsize=8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(grid)) 
        fig.savefig(f'{savefolder}/{seq}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)



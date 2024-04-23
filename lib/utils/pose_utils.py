"""
Code adapted from: https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple

from lib.core import constants


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re.cpu().numpy()

def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    # Reconstuction_error
    r_error = reconstruction_error(pred_joints.cpu(), gt_joints.cpu())
    return 1000 * mpjpe, 1000 * r_error

class Evaluator:

    def __init__(self, dataset_length=None, seq_len=None):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            dataset_length (int): Total dataset length.
            keypoint_list [List]: List of keypoints used for evaluation.
            pelvis_ind (int): Index of pelvis keypoint; used for aligning the predictions and ground truth.
            metrics [List]: List of evaluation metrics to record.
        """
        self.dataset_length = dataset_length
        self.seq_len = seq_len
        
        self.mpjpe = np.zeros((dataset_length,))
        self.re = np.zeros((dataset_length,))
        self.pve = np.zeros((dataset_length,))
        self.acc = np.zeros((dataset_length,))
        self.counter = 0

        self.J24_TO_J17 = constants.J24_TO_J17
        self.J24_TO_J14 = constants.J24_TO_J14
        self.H36M_TO_J17 = constants.H36M_TO_J17
        self.H36M_TO_J14 = constants.H36M_TO_J14
        self.all_acc = []


    def __call__(self, gt_keypoints_3d, pred_keypoints_3d, dataset='3dpw', 
                gt_verts=None, pred_verts=None):

        batch_size = gt_keypoints_3d.shape[0]

        gt_keypoints_3d = gt_keypoints_3d[:, :, :3].detach()
        pred_keypoints_3d = pred_keypoints_3d[:, :, :3].detach()
        num_j = gt_keypoints_3d.shape[1]
        
        gt_valid, pred_valid = self.get_valid_joints(gt_keypoints_3d, 
                                                     pred_keypoints_3d, 
                                                     dataset)


        # Compute joint errors
        mpjpe, re = eval_pose(pred_valid, gt_valid)

        self.mpjpe[self.counter:self.counter+batch_size] = mpjpe
        self.re[self.counter:self.counter+batch_size] = re
        
        if gt_verts is not None and pred_verts is not None:
            pve = (pred_verts - gt_verts).norm(dim=-1).mean(dim=-1).cpu().numpy()
            self.pve[self.counter:self.counter+batch_size] = pve * 1000

        if self.seq_len is not None:
            gt = gt_keypoints_3d.reshape(-1, self.seq_len, num_j, 3).cpu()
            pred = pred_keypoints_3d.reshape(-1, self.seq_len, num_j, 3).cpu()
            acc = 0
            for i in range(len(gt)):
                acc += compute_error_accel(gt[i], pred[i]).mean() / len(gt)
            
            self.acc[self.counter:self.counter+batch_size] = acc * 1000 #(30**2)
            
        self.counter += batch_size


    def get_valid_joints(self, gt_keypoints_3d, pred_keypoints_3d, dataset):
        if 'emdb' in dataset:
            gt_valid = gt_keypoints_3d
            pred_valid = pred_keypoints_3d

        else:
            j_mapper = self.get_gt_mapper(dataset)
            gt_valid = gt_keypoints_3d[:, j_mapper]

            j_mapper = self.get_pred_mapper(dataset)
            pred_valid = pred_keypoints_3d[:, j_mapper]

        return gt_valid, pred_valid


    def get_gt_mapper(self, dataset):

        if dataset == 'mpi-inf-3dhp':
            j_mapper = self.J24_TO_J17

        elif dataset == 'h36m':
            j_mapper = self.J24_TO_J14

        elif dataset == '3dpw':
            j_mapper = self.H36M_TO_J14

        return j_mapper


    def get_pred_mapper(self, dataset):
        
        if dataset == 'mpi-inf-3dhp':
            j_mapper = self.H36M_TO_J17

        elif dataset == 'h36m':
            j_mapper = self.H36M_TO_J14

        elif dataset == '3dpw':
            j_mapper = self.H36M_TO_J14

        return j_mapper


    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return

        print(f'{self.counter} / {self.dataset_length} samples')
        print(f're: {self.re[:self.counter].mean()} mm')
        print(f'mpjpe: {self.mpjpe[:self.counter].mean()} mm')
        print(f'pve: {self.pve[:self.counter].mean()} mm')
        print(f'accel: {self.acc[:self.counter].mean()} mm')
        print('***')



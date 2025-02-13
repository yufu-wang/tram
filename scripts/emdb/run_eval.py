import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob
from tqdm import tqdm
from collections import defaultdict

from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.vis.traj import *
from lib.camera.slam_utils import eval_slam

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--input_dir', type=str, default='results/emdb')
args = parser.parse_args()
input_dir = args.input_dir

# EMDB dataset and splits
roots = []
for p in range(10):
    folder = f'/mnt/kostas-graid/datasets/yufu/emdb/P{p}'
    root = sorted(glob(f'{folder}/*'))
    roots.extend(root)

emdb = []
spl = args.split
for root in roots:
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))
    if ann[f'emdb{spl}']:
        emdb.append(root)

# SMPL
smpl = SMPL()
smpls = {g:SMPL(gender=g) for g in ['neutral', 'male', 'female']}


# Evaluations: world-coordinate SMPL
accumulator = defaultdict(list)
m2mm = 1e3
human_traj = {}
total_invalid = 0

for root in tqdm(emdb):
    # GT
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))

    ext = ann['camera']['extrinsics']  # in the forms of R_cw, t_cw
    intr = ann['camera']['intrinsics']
    img_focal = (intr[0,0] +  intr[1,1]) / 2.
    img_center = intr[:2, 2]

    valid = ann['good_frames_mask']
    gender = ann['gender']
    poses_body = ann["smpl"]["poses_body"]
    poses_root = ann["smpl"]["poses_root"]
    betas = np.repeat(ann["smpl"]["betas"].reshape((1, -1)), repeats=ann["n_frames"], axis=0)
    trans = ann["smpl"]["trans"]
    total_invalid += (~valid).sum()

    tt = lambda x: torch.from_numpy(x).float()
    gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root), betas=tt(betas), transl=tt(trans),
                    pose2rot=True, default_smpl=True)
    gt_vert = gt.vertices
    gt_j3d = gt.joints[:,:24] 
    gt_ori = axis_angle_to_matrix(tt(poses_root))

    # Groundtruth local motion
    poses_root_cam = matrix_to_axis_angle(tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root)))
    gt_cam = smpls[gender](body_pose=tt(poses_body), global_orient=poses_root_cam, betas=tt(betas),
                           pose2rot=True, default_smpl=True)
    gt_vert_cam = gt_cam.vertices
    gt_j3d_cam = gt_cam.joints[:,:24] 
    
    # PRED
    seq = root.split('/')[-1]
    pred_cam = dict(np.load(f'{input_dir}/camera/{seq}.npz'))
    pred_smpl = dict(np.load(f'{input_dir}/smpl/{seq}.npz'))

    pred_rotmat = torch.tensor(pred_smpl['pred_rotmat'])
    pred_shape = torch.tensor(pred_smpl['pred_shape'])
    pred_trans = torch.tensor(pred_smpl['pred_trans'])

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.repeat(len(pred_shape), 1)

    pred = smpls['neutral'](body_pose=pred_rotmat[:,1:], 
                            global_orient=pred_rotmat[:,[0]], 
                            betas=pred_shape, 
                            transl=pred_trans.squeeze(),
                            pose2rot=False, 
                            default_smpl=True)
    pred_vert = pred.vertices
    pred_j3d = pred.joints[:, :24]

    pred_camt = torch.tensor(pred_cam['pred_cam_T']) 
    pred_camr = torch.tensor(pred_cam['pred_cam_R'])
   
    pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:,None]
    pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:,None]
    pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, pred_rotmat[:,0])
    pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)

    # Valid mask
    gt_j3d = gt_j3d[valid]
    gt_ori = gt_ori[valid]
    pred_j3d_w  = pred_j3d_w[valid]
    pred_ori_w = pred_ori_w[valid]

    gt_j3d_cam = gt_j3d_cam[valid]
    gt_vert_cam = gt_vert_cam[valid]
    pred_j3d = pred_j3d[valid]
    pred_vert = pred_vert[valid]

    # <======= Evaluation on the local motion
    pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam = batch_align_by_pelvis(
        [pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam], pelvis_idxs=[1,2]
    )
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, gt_j3d_cam)
    pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
    mpjpe = torch.sqrt(((pred_j3d - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
    pve = torch.sqrt(((pred_vert - gt_vert_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

    accel = compute_error_accel(joints_pred=pred_j3d.cpu(), joints_gt=gt_j3d_cam.cpu())[1:-1]
    accel = accel * (30 ** 2)       # per frame^s to per s^2

    accumulator['pa_mpjpe'].append(pa_mpjpe)
    accumulator['mpjpe'].append(mpjpe)
    accumulator['pve'].append(pve)
    accumulator['accel'].append(accel)
    # =======>

    # <======= Evaluation on the global motion
    chunk_length = 100
    w_mpjpe, wa_mpjpe = [], []
    for start in range(0, valid.sum() - chunk_length, chunk_length):
        end = start + chunk_length
        if start + 2 * chunk_length > valid.sum(): end = valid.sum() - 1
        
        target_j3d = gt_j3d[start:end].clone().cpu()
        pred_j3d = pred_j3d_w[start:end].clone().cpu()
        
        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)
        
        w_jpe = compute_jpe(target_j3d, w_j3d)
        wa_jpe = compute_jpe(target_j3d, wa_j3d)
        w_mpjpe.append(w_jpe)
        wa_mpjpe.append(wa_jpe)

    w_mpjpe = np.concatenate(w_mpjpe) * m2mm
    wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
    # =======>

    # <======= Evaluation on the entier global motion
    # RTE: root trajectory error
    pred_j3d_align = first_align_joints(gt_j3d, pred_j3d_w)
    rte_align_first= compute_jpe(gt_j3d[:,[0]], pred_j3d_align[:,[0]])
    rte_align_all = compute_rte(gt_j3d[:,0], pred_j3d_w[:,0]) * 1e2 

    # ERVE: Ego-centric root velocity error
    erve = computer_erve(gt_ori, gt_j3d, pred_ori_w, pred_j3d_w) * m2mm
    # =======>

    # <======= Record human trajectory
    human_traj[seq] = {'gt': gt_j3d[:,0], 'pred': pred_j3d_align[:, 0]}
    # =======>

    accumulator['wa_mpjpe'].append(wa_mpjpe)
    accumulator['w_mpjpe'].append(w_mpjpe)
    accumulator['rte'].append(rte_align_all)
    accumulator['erve'].append(erve)
    
for k, v in accumulator.items():
    accumulator[k] = np.concatenate(v).mean()


# Evaluation: Camera motion
results = {}
for root in emdb:
    # Annotation
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))

    ext = ann['camera']['extrinsics']
    cam_r = ext[:,:3,:3].transpose(0,2,1)
    cam_t = np.einsum('bij, bj->bi', cam_r, -ext[:, :3, -1])
    cam_q = matrix_to_quaternion(torch.from_numpy(cam_r)).numpy()

    # PRED
    seq = root.split('/')[-1]
    pred_cam = dict(np.load(f'{input_dir}/camera/{seq}.npz'))

    pred_camt = torch.tensor(pred_cam['pred_cam_T'])
    pred_camr = torch.tensor(pred_cam['pred_cam_R'])
    pred_camq = matrix_to_quaternion(pred_camr)
    pred_traj = torch.concat([pred_camt, pred_camq], dim=-1).numpy()

    stats_slam, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=True)
    stats_metric, traj_ref, traj_est = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=False)
  
    # Save results
    re = {'traj_gt': traj_ref.positions_xyz,
          'traj_est': traj_est.positions_xyz, 
          'traj_gt_q': traj_ref.orientations_quat_wxyz,
          'traj_est_q': traj_est.orientations_quat_wxyz,
          'stats_slam': stats_slam,
          'stats_metric': stats_metric}
    
    results[seq] = re

ate = np.mean([re['stats_slam']['mean'] for re in results.values()])
ate_s = np.mean([re['stats_metric']['mean'] for re in results.values()])
accumulator['ate'] = ate
accumulator['ate_s'] = ate_s

# Save evaluation results
for k, v in accumulator.items():
    print(k, accumulator[k])

df = pd.DataFrame(list(accumulator.items()), columns=['Metric', 'Value'])
df.to_excel(f"{args.input_dir}/evaluation.xlsx", index=False)

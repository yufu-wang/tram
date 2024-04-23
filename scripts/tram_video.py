import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
import os
import torch
import cv2
from tqdm import tqdm
from glob import glob
import imageio

from lib.vis.traj import *
from lib.models.smpl import SMPL
from lib.vis.renderer import Renderer
from lib.utils.rotation_conversions import quaternion_to_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./example_video.mov')
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

##### Read results from SLAM and HPS #####
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'{root}/{seq}'
img_folder = f'{seq_folder}/images'
hps_folder = f'{seq_folder}/hps'
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
hps_files = sorted(glob(f'{hps_folder}/*.npy'))

slam = dict(np.load(f'{seq_folder}/masked_droid_slam.npz'))
img_focal = slam['img_focal'].tolist()
img_center = slam['img_center'].tolist()

smpl = SMPL()
colors = np.loadtxt('data/colors.txt')/255
colors = torch.from_numpy(colors).float()

max_track = len(hps_files)
tstamp =  [t for t in range(len(imgfiles))]
track_verts = {i:[] for i in tstamp}
track_joints = {i:[] for i in tstamp}
track_tid = {i:[] for i in tstamp}
locations = []

##### TRAM + VIMO #####
pred_cam = dict(np.load(f'{seq_folder}/masked_droid_slam.npz', allow_pickle=True))
img_focal = pred_cam['img_focal'].item()

for i in range(max_track):
    hps_file = hps_files[i]

    pred_smpl = np.load(hps_file, allow_pickle=True).item()
    pred_rotmat = pred_smpl['pred_rotmat']
    pred_shape = pred_smpl['pred_shape']
    pred_trans = pred_smpl['pred_trans']
    frame = pred_smpl['frame']

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.repeat(len(pred_shape), 1)

    pred = smpl(body_pose=pred_rotmat[:,1:], 
                global_orient=pred_rotmat[:,[0]], 
                betas=pred_shape, 
                transl=pred_trans.squeeze(),
                pose2rot=False, 
                default_smpl=True)
    pred_vert = pred.vertices
    pred_j3d = pred.joints[:, :24]

    pred_traj = pred_cam['traj']
    pred_camt = torch.tensor(pred_traj[frame, :3]) * pred_cam['scale']
    pred_camq = torch.tensor(pred_traj[frame, 3:])
    pred_camr = quaternion_to_matrix(pred_camq[:,[3,0,1,2]])

    pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:,None]
    pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:,None]
    pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)
    locations.append(pred_j3d_w.mean(1))

    for j, f in enumerate(frame.tolist()):
        track_tid[f].append(i)
        track_verts[f].append(pred_vert_w[j])
        track_joints[f].append(pred_j3d_w[j])

##### Fit to Ground #####
grounding_verts = []
grounding_joints = []
for t in tstamp[:10] + tstamp[-10:]:
    verts = torch.stack(track_verts[t])
    joints = torch.stack(track_joints[t])
    grounding_verts.append(verts)
    grounding_joints.append(joints)
    
grounding_verts = torch.cat(grounding_verts)
grounding_joints = torch.cat(grounding_joints)

R, offset = fit_to_ground_easy(grounding_verts, grounding_joints)
offset = torch.tensor([0, offset, 0])

locations = torch.cat(locations)
locations = torch.einsum('ij,bj->bi', R, locations) - offset
cx, cz = (locations.max(0)[0] + locations.min(0)[0])[[0, 2]] / 2.0
sx, sz = (locations.max(0)[0] - locations.min(0)[0])[[0, 2]]
scale = max(sx.item(), sz.item()) * 2

##### Viewing Camera #####
pred_cam = dict(np.load(f'{seq_folder}/masked_droid_slam.npz', allow_pickle=True))
pred_traj = pred_cam['traj']
pred_camt = torch.tensor(pred_traj[:, :3]) * pred_cam['scale']
pred_camq = torch.tensor(pred_traj[:, 3:])
pred_camr = quaternion_to_matrix(pred_camq[:,[3,0,1,2]])

cam_R = torch.einsum('ij,bjk->bik', R, pred_camr)
cam_T = torch.einsum('ij,bj->bi', R, pred_camt) - offset
cam_R = cam_R.mT
cam_T = - torch.einsum('bij,bj->bi', cam_R, cam_T)

cam_R = cam_R.to('cuda')
cam_T = cam_T.to('cuda')

##### Render video for visualization #####
writer = imageio.get_writer(f'{seq_folder}/tram_output.mp4', fps=30, mode='I', 
                            format='FFMPEG', macro_block_size=1)
bin_size = 64
max_faces_per_bin = 20000
img = cv2.imread(imgfiles[0])
renderer = Renderer(img.shape[1], img.shape[0], img_focal-100, 'cuda', 
                    smpl.faces, bin_size=bin_size, max_faces_per_bin=max_faces_per_bin)
renderer.set_ground(scale, cx.item(), cz.item())

for i in tqdm(range(len(imgfiles))):
    img = cv2.imread(imgfiles[i])[:,:,::-1]
    
    verts_list = track_verts[i]
    if len(verts_list)>0:
        verts_list = torch.stack(track_verts[i])#[:,None]
        verts_list = torch.einsum('ij,bnj->bni', R, verts_list)[:,None]
        verts_list -= offset
        verts_list = verts_list.to('cuda')
        
        tid = track_tid[i]
        verts_colors = torch.stack([colors[t] for t in tid]).to('cuda')

    faces = renderer.faces.clone().squeeze(0)
    cameras, lights = renderer.create_camera_from_cv(cam_R[[i]], cam_T[[i]])
    rend = renderer.render_with_ground_multiple(verts_list, faces, verts_colors, 
                                                cameras, lights)
    
    out = np.concatenate([img, rend], axis=1)
    writer.append_data(out)

writer.close()
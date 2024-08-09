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


def visualize_tram(seq_folder, contact_frames=None, floor_scale=2, 
                   bin_size=-1, max_faces_per_bin=30000):
    img_folder = f'{seq_folder}/images'
    hps_folder = f'{seq_folder}/hps'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    hps_files = sorted(glob(f'{hps_folder}/*.npy'))

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
    pred_cam = np.load(f'{seq_folder}/camera.npy', allow_pickle=True).item()
    img_focal = pred_cam['img_focal'].item()
    pred_cam_R = torch.tensor(pred_cam['pred_cam_R'])
    pred_cam_T = torch.tensor(pred_cam['pred_cam_T'])

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

        cam_r = pred_cam_R[frame]
        cam_t = pred_cam_T[frame]

        pred_vert_w = torch.einsum('bij,bnj->bni', cam_r, pred_vert) + cam_t[:,None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', cam_r, pred_j3d) + cam_t[:,None]
        pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)
        locations.append(pred_j3d_w.mean(1))

        for j, f in enumerate(frame.tolist()):
            track_tid[f].append(i)
            track_verts[f].append(pred_vert_w[j])
            track_joints[f].append(pred_j3d_w[j])

    ##### Fit to Ground #####
    grounding_verts = []
    grounding_joints = []
    if contact_frames is None:
        contact_frames = tstamp
    for t in contact_frames:
        try:
            verts = torch.stack(track_verts[t])
            joints = torch.stack(track_joints[t])
            grounding_verts.append(verts)
            grounding_joints.append(joints)
        except Exception:
            continue
        
    grounding_verts = torch.cat(grounding_verts)
    grounding_joints = torch.cat(grounding_joints)

    R, offset = fit_to_ground_easy(grounding_verts, grounding_joints)
    offset = torch.tensor([0, offset, 0])

    locations = torch.cat(locations)
    locations = torch.einsum('ij,bj->bi', R, locations) - offset
    cx, cz = (locations.max(0)[0] + locations.min(0)[0])[[0, 2]] / 2.0
    sx, sz = (locations.max(0)[0] - locations.min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item()) * floor_scale

    ##### Viewing Camera #####
    pred_cam = np.load(f'{seq_folder}/camera.npy', allow_pickle=True).item()
    pred_cam_R = torch.tensor(pred_cam['pred_cam_R'])
    pred_cam_T = torch.tensor(pred_cam['pred_cam_T'])

    cam_R = torch.einsum('ij,bjk->bik', R, pred_cam_R)
    cam_T = torch.einsum('ij,bj->bi', R, pred_cam_T) - offset
    cam_R = cam_R.mT
    cam_T = - torch.einsum('bij,bj->bi', cam_R, cam_T)

    cam_R = cam_R.to('cuda')
    cam_T = cam_T.to('cuda')

    ##### Render video for visualization #####
    writer = imageio.get_writer(f'{seq_folder}/tram_output.mp4', fps=30, mode='I', 
                                format='FFMPEG', macro_block_size=1)
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
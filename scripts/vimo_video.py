import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from glob import glob

from lib.core.config import update_cfg
from lib.models.hmr_vimo import HMR_VIMO
from lib.pipeline.tools import parse_chunks


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default='configs/config_vimo.yaml')
parser.add_argument("--ckpt",  type=str, default='data/pretrain/vimo_checkpoint.pth.tar')
parser.add_argument("--video", type=str, default='./example_video.mov')
parser.add_argument("--max_track", type=int, default=10)
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()


# VIMO
cfg = update_cfg(args.cfg)
cfg.DEVICE = args.device

model = HMR_VIMO(cfg)
ckpt = torch.load(args.ckpt, map_location='cpu')
mess = model.load_state_dict(ckpt['model'], strict=False)
model = model.to('cuda')
_ = model.eval()
print('Loaded checkpoint:', args.ckpt)

# Folders
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'{root}/{seq}'
img_folder = f'{seq_folder}/images'
hps_folder = f'{seq_folder}/hps'
os.makedirs(hps_folder, exist_ok=True)

# Previous steps
imgfiles = np.array(sorted(glob(f'{img_folder}/*.jpg')))
tracks = np.load(f'{seq_folder}/tracks.npy', allow_pickle=True).item()
slam = dict(np.load(f'{seq_folder}/masked_droid_slam.npz'))
img_focal = slam['img_focal'].tolist()
img_center = slam['img_center'].tolist()

# --- Tracks: sort by length  ---
tid = np.array([tr for tr in tracks])
tlen = np.array([len(tracks[tr]) for tr in tracks])
sort = np.argsort(tlen)[::-1]
tid = tid[sort]

print(f'Running on {file} ...')

# --- Run VIMO on each track ---
track_count = 0
for k, idx in enumerate(tid):
    trk = tracks[idx]
    valid = np.array([t['det'] for t in trk])
    boxes = np.concatenate([t['det_box'] for t in trk])[valid]
    frame = np.array([t['frame'] for t in trk])[valid]
    frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=16)

    if len(frame_chunks) == 0:
        continue

    pred_cam = []
    pred_pose = []
    pred_shape = []
    pred_rotmat = []
    pred_trans = []
    frame = []

    for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
        img_ck = imgfiles[frame_ck]
        results = model.inference(img_ck, boxes_ck, img_focal=img_focal, img_center=img_center)

        pred_cam.append(results['pred_cam'])
        pred_pose.append(results['pred_pose'])
        pred_shape.append(results['pred_shape'])
        pred_rotmat.append(results['pred_rotmat'])
        pred_trans.append(results['pred_trans'])
        frame.append(torch.from_numpy(frame_ck))
    
    results = {'pred_cam': torch.cat(pred_cam),
            'pred_pose': torch.cat(pred_pose),
            'pred_shape': torch.cat(pred_shape),
            'pred_rotmat': torch.cat(pred_rotmat),
            'pred_trans': torch.cat(pred_trans),
            'frame': torch.cat(frame)}

    np.save(f'{hps_folder}/vimo_track_{track_count}.npy', results)
    
    track_count += 1
    if track_count >= args.max_track:
        break
    





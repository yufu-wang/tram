import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
from pycocotools import mask as masktool
from lib.pipeline.masked_droid_slam import *
from lib.pipeline.est_scale import *


parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./example_video.mov')
parser.add_argument("--img_focal", type=int, default=None)
parser.add_argument('--img_center', nargs='+', default=None)
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# File and folders
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'{root}/{seq}'
img_folder = f'{seq_folder}/images'
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
print(f'Running on {file} ...')

##### Run Masked DROID-SLAM #####
# Use Masking
masks = np.load(f'{seq_folder}/masks.npy', allow_pickle=True)
masks = np.array([masktool.decode(m) for m in masks])
masks = torch.from_numpy(masks)

# Camera calibration (intrinsics) for SLAM
focal = args.img_focal
center = args.img_center
calib = np.array(est_calib(img_folder))

if focal is None:
    print('No focal length provided ...')
    print('Search for a good focal length for SLAM ...')
    focal = search_focal_length(img_folder, masks)
if center is None:
    center = calib[2:]
    
calib[:2] = focal
calib[2:] = center

# Droid-slam with masking
droid, traj = run_slam(img_folder, masks=masks, calib=calib)
n = droid.video.counter.value
tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
disps = droid.video.disps_up.cpu().numpy()[:n]
print('DBA errors:', droid.backend.errors)

# Save results
np.savez(f'{seq_folder}/masked_droid_slam.npz', 
            tstamp=tstamp, disps=disps, traj=traj, 
            img_focal=focal, img_center=calib[-2:])

del droid
torch.cuda.empty_cache()


##### Estimate Metric Depth #####
repo = "isl-org/ZoeDepth"
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
_ = model_zoe_n.eval()
model_zoe_n = model_zoe_n.to('cuda')

print('Predicting Metric Depth ...')
pred_depths = []
H, W = get_dimention(img_folder)
for t in tqdm(tstamp):
    img = cv2.imread(imgfiles[t])[:,:,::-1]
    img = cv2.resize(img, (W, H))
    
    img_pil = Image.fromarray(img)
    pred_depth = model_zoe_n.infer_pil(img_pil)
    pred_depths.append(pred_depth)
np.save(f'{seq_folder}/zoe_depth.npy', pred_depths)

##### Estimate Metric Scale #####
print('Estimating Metric Scale ...')
scales_ = []
n = len(tstamp)   # for each keyframe
for i in tqdm(range(n)):
    t = tstamp[i]
    disp = disps[i]
    pred_depth = pred_depths[i]
    slam_depth = 1/disp
    
    # Estimate scene scale
    msk = masks[t].numpy()
    scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, far_thresh=10)
    scales_.append(scale)

median_s = np.median(scales_)

# Save results
np.savez(f'{seq_folder}/masked_droid_slam.npz', 
            tstamp=tstamp, disps=disps, traj=traj, 
            img_focal=focal, img_center=calib[-2:],
            scale=median_s)




import sys
sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
sys.path.insert(0, 'thirdparty/DROID-SLAM')

from tqdm import tqdm
import numpy as np
import torch
import os
import argparse
from PIL import Image
import cv2
from glob import glob

from droid import Droid
from torch.multiprocessing import Process

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from pycocotools import mask as masktool
from torchvision.transforms import Resize

# Some default settings for DROID-SLAM
parser = argparse.ArgumentParser()
parser.add_argument("--imagedir", type=str, help="path to image directory")
parser.add_argument("--calib", type=str, help="path to calibration file")
parser.add_argument("--t0", default=0, type=int, help="starting frame")
parser.add_argument("--stride", default=1, type=int, help="frame stride")

parser.add_argument("--weights", default="data/pretrain/droid.pth")
parser.add_argument("--buffer", type=int, default=512)
parser.add_argument("--image_size", default=[240, 320])
parser.add_argument("--disable_vis", action="store_true")

parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

parser.add_argument("--backend_thresh", type=float, default=22.0)
parser.add_argument("--backend_radius", type=int, default=2)
parser.add_argument("--backend_nms", type=int, default=3)
parser.add_argument("--upsample", action="store_true")
parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
args = parser.parse_args([])
args.stereo = False
args.upsample = True
args.disable_vis = True
torch.multiprocessing.set_start_method('spawn')


def est_calib(imagedir):
    """ Roughly estimate intrinsics by image dimensions """
    imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
    image = cv2.imread(imgfiles[0])

    h0, w0, _ = image.shape
    focal = np.max([h0, w0])
    cx, cy = w0/2., h0/2.
    calib = [focal, focal, cx, cy]
    return calib


def get_dimention(imagedir):
    """ Get proper image dimension for DROID """
    imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
    image = cv2.imread(imgfiles[0])

    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1%8, :w1-w1%8]
    H, W, _ = image.shape
    return H, W


def image_stream(imagedir, calib, stride, max_frame=None):
    """ Image generator for DROID """
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(glob(f'{imagedir}/*.jpg'))
    image_list = image_list[::stride]
    if max_frame is not None:
        image_list = image_list[:max_frame]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(imfile)
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def run_slam(imagedir, masks, calib=None, depth=None, stride=1,  
             filter_thresh=2.4, disable_vis=True):
    """ Maksed DROID-SLAM """
    droid = None
    depth = None
    args.filter_thresh = filter_thresh
    args.disable_vis = disable_vis
    masks = masks[::stride]

    img_msks, conf_msks = preprocess_masks(imagedir, masks)
    if calib is None:
        calib = est_calib(imagedir)

    for (t, image, intrinsics) in tqdm(image_stream(imagedir, calib, stride)):

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        img_msk = img_msks[t]
        conf_msk = conf_msks[t]
        image = image * (img_msk < 0.5)

        droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)  

    traj = droid.terminate(image_stream(imagedir, calib, stride))

    return droid, traj


def eval_slam(traj_est, cam_t, cam_q, return_traj=True, correct_scale=False, align=True, align_origin=False):
    """ Evaluation for SLAM """
    tstamps = np.array([i for i in range(len(traj_est))], dtype=np.float32)

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3], 
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=tstamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=cam_t.copy(),
        orientations_quat_wxyz=cam_q.copy(),
        timestamps=tstamps)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=align, align_origin=align_origin,
        correct_scale=correct_scale)
    
    stats = result.stats

    if return_traj:
        return stats, traj_ref, traj_est
    
    return stats


def test_slam(imagedir, img_msks, conf_msks, calib, stride=10, max_frame=50):
    """ Shorter SLAM step to test reprojection error """
    args = parser.parse_args([])
    args.stereo = False
    args.upsample = False
    args.disable_vis = True
    args.frontend_window = 10
    args.frontend_thresh = 10
    droid = None

    for (t, image, intrinsics) in image_stream(imagedir, calib, stride, max_frame):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        img_msk = img_msks[t]
        conf_msk = conf_msks[t]
        image = image * (img_msk < 0.5)
        droid.track(t, image, intrinsics=intrinsics, mask=conf_msk)  

    reprojection_error = droid.compute_error()
    del droid

    return reprojection_error


def search_focal_length(img_folder, masks, stride=10, max_frame=50,
                        low=500, high=1500, step=100):
    """ Search for a good focal length by SLAM reprojection error """
    masks = masks[::stride]
    masks = masks[:max_frame]
    img_msks, conf_msks = preprocess_masks(img_folder, masks)

    # default estimate
    calib = np.array(est_calib(img_folder))
    best_focal = calib[0]
    best_err = test_slam(img_folder, img_msks, conf_msks, 
                         stride=stride, calib=calib, max_frame=max_frame)
    
    # search based on slam reprojection error
    for focal in range(low, high, step):
        calib[:2] = focal
        err = test_slam(img_folder, img_msks, conf_msks, 
                        stride=stride, calib=calib, max_frame=max_frame)

        if err < best_err:
            best_err = err
            best_focal = focal
            
    print('Best focal length:', best_focal)

    return best_focal


def preprocess_masks(img_folder, masks):
    """ Resize masks for masked droid """
    H, W = get_dimention(img_folder)
    resize_1 = Resize((H, W), antialias=True)
    resize_2 = Resize((H//8, W//8), antialias=True)
    
    img_msks = []
    for i in range(0, len(masks), 500):
        m = resize_1(masks[i:i+500])
        img_msks.append(m)
    img_msks = torch.cat(img_msks)

    conf_msks = []
    for i in range(0, len(masks), 500):
        m = resize_2(masks[i:i+500])
        conf_msks.append(m)
    conf_msks = torch.cat(conf_msks)

    return img_msks, conf_msks






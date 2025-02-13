import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import cv2
import torch
import argparse
import numpy as np
import pickle as pkl
from glob import glob
from tqdm import tqdm

from lib.camera import run_metric_slam, align_cam_to_world
from lib.pipeline.tools import arrange_boxes
from lib.utils.utils_detectron2 import DefaultPredictor_Lazy

from torch.amp import autocast
from segment_anything import SamPredictor, sam_model_registry
from detectron2.config import LazyConfig

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--output_dir', type=str, default='results/emdb/camera')
args = parser.parse_args()


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


# Save folder
savefolder = args.ourtput_dir
os.makedirs(savefolder, exist_ok=True)

# ViTDet
device = 'cuda'
cfg_path = 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)

# SAM
sam = sam_model_registry["vit_h"](checkpoint="data/pretrain/sam_vit_h_4b8939.pth")
_ = sam.to(device)
predictor = SamPredictor(sam)


# Estimate camera motion on EMDB (subset: spl)
for root in emdb:
    print(f'Running on {root}...')

    seq = root.split('/')[-1]
    img_folder = f'{root}/images'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    masks_ = []
    for t, imgpath in enumerate(tqdm(imgfiles)):
        img_cv2 = cv2.imread(imgpath)

        ### --- Detection ---
        with torch.no_grad():
            with autocast('cuda'):
                det_out = detector(img_cv2)
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                confs = det_instances.scores[valid_idx].cpu().numpy()

                boxes = np.hstack([boxes, confs[:, None]])
                boxes = arrange_boxes(boxes, mode='size', min_size=100)


        ### --- SAM --- 
        if len(boxes)>0:
            with autocast('cuda'):
                predictor.set_image(img_cv2, image_format='BGR')

                # multiple boxes
                bb = torch.tensor(boxes[:, :4]).cuda()
                bb = predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])  
                masks, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=bb,
                    multimask_output=False
                )
                scores = scores.cpu()
                masks = masks.cpu().squeeze(1)
                mask = masks.sum(dim=0)
        else:
            mask = torch.zeros_like(mask)
        
        masks_.append(mask.byte())

    masks = torch.stack(masks_)
  

    ### --- Camera Motion ---
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))
    intr = ann['camera']['intrinsics']

    cam_int = [intr[0,0], intr[1,1], intr[0,2], intr[1,2]]
    cam_R, cam_T = run_metric_slam(img_folder, masks=masks, calib=cam_int)
    wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)

    camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(), 
            'world_cam_R': wd_cam_R.numpy(), 'world_cam_T': wd_cam_T.numpy(),
            'img_focal': cam_int[0], 'img_center': cam_int[2:], 'spec_focal': spec_f}


    ### --- Save results ---
    savefile = f'{savefolder}/{seq}.npz'
    np.savez(savefile, **camera)

    
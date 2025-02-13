import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import torch
import argparse
import numpy as np
import pickle as pkl
from glob import glob
from tqdm import tqdm

from torch.utils.data import default_collate
from lib.models import get_hmr_vimo
from lib.datasets.image_dataset import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--output_dir', type=str, default='results/emdb/smpl')
parser.add_argument('--efficient', action='store_true', help='efficient option, but increase ACC error.')
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
savefolder = args.output_dir
os.makedirs(savefolder, exist_ok=True)

# HPS model
device = 'cuda'
model = get_hmr_vimo(checkpoint='data/pretrain/vimo_checkpoint.pth.tar').to(device)


# Predict SMPL on EMDB (subset: spl)
for i, root in enumerate(emdb):
    print('Running HPS on', root)

    seq = root.split('/')[-1]
    imgfiles = sorted(glob(f'{root}/images/*.jpg'))
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))
    
    ext = ann['camera']['extrinsics']
    intr = ann['camera']['intrinsics']
    ann_boxes = ann['bboxes']['bboxes']
    img_focal = (intr[0,0] +  intr[1,1]) / 2.
    img_center = intr[:2, 2]
    
    db = ImageDataset(imgfiles, ann_boxes, img_focal=img_focal, 
                      img_center=img_center, normalization=True)
    dataloader = torch.utils.data.DataLoader(db, batch_size=64, shuffle=False, num_workers=12)

    # Results
    pred_cam = []
    pred_pose = []
    pred_shape = []
    pred_rotmat = []
    pred_trans = []

    ### Efficient option: none-overlap sliding window (higher acc error from 4.5 to 7)
    if args.efficient:
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items() if type(v)==torch.Tensor}

            # Last batch
            n = len(batch['img'])
            if n < 64:
                for k in batch:
                    batch[k] = torch.cat([previous_batch[k][n-64:],
                                        batch[k]], dim=0)
            
            with torch.no_grad():
                out, _ = model(batch)

            # Last batch
            if n < 64:
                for k in out:
                    out[k] = out[k][64-n:]
            
            pred_cam.append(out['pred_cam'].cpu())
            pred_pose.append(out['pred_pose'].cpu())
            pred_shape.append(out['pred_shape'].cpu())
            pred_rotmat.append(out['pred_rotmat'].cpu())
            pred_trans.append(out['trans_full'].cpu())
            previous_batch = batch

    ### Maximum overlapping sliding window 
    else:
        items = []
        for i in tqdm(range(len(db))):
            item = db[i]
            items.append(item)

            if len(items) < 16:
                continue
            elif len(items) == 16:
                batch = default_collate(items)
            else:
                items.pop(0)
                batch = default_collate(items)

            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items() if type(v)==torch.Tensor}
                out, _ = model.forward(batch)

            if i == 15:
                out = {k:v[:9] for k,v in out.items()}
            elif i == len(db) - 1:
                out = {k:v[8:] for k,v in out.items()}
            else:
                out = {k:v[[8]] for k,v in out.items()}
                
            pred_cam.append(out['pred_cam'].cpu())
            pred_pose.append(out['pred_pose'].cpu())
            pred_shape.append(out['pred_shape'].cpu())
            pred_rotmat.append(out['pred_rotmat'].cpu())
            pred_trans.append(out['trans_full'].cpu())


    results = {'pred_cam': torch.cat(pred_cam),
            'pred_pose': torch.cat(pred_pose),
            'pred_shape': torch.cat(pred_shape),
            'pred_rotmat': torch.cat(pred_rotmat),
            'pred_trans': torch.cat(pred_trans),
            'img_focal': img_focal,
            'img_center': img_center}

    np.savez(f'{savefolder}/{seq}.npz', **results)
    

    
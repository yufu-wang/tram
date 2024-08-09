import numpy as np
import einops
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import default_collate

from lib.utils.geometry import perspective_projection
from lib.utils.geometry import rot6d_to_rotmat_hmr2 as rot6d_to_rotmat
from lib.datasets.track_dataset import TrackDataset

from .vit import vit_huge
from .modules import *
from .smpl import SMPL
from ..pipeline.tools import parse_chunks

autocast = torch.cuda.amp.autocast


class HMR_VIMO(nn.Module):
    def __init__(self, cfg=None, device='cpu', **kwargs):

        super(HMR_VIMO, self).__init__()
        self.device = device
        self.cfg = cfg
        self.crop_size = cfg.IMG_RES
        self.seq_len = 16

        # SMPL
        self.smpl = SMPL()      

        # Backbone
        self.backbone = vit_huge()

        # Space-time memory
        if cfg.MODEL.ST_MODULE: 
            hdim = cfg.MODEL.ST_HDIM
            nlayer = cfg.MODEL.ST_NLAYER
            self.st_module = temporal_attention(in_dim=1280+3, 
                                                out_dim=1280,
                                                hdim=hdim,
                                                nlayer=nlayer,
                                                residual=True)
        else:
            self.st_module = None

        # Motion memory
        if cfg.MODEL.MOTION_MODULE:
            hdim = cfg.MODEL.MOTION_HDIM
            nlayer = cfg.MODEL.MOTION_NLAYER
            self.motion_module = temporal_attention(in_dim=144+3, 
                                                    out_dim=144,
                                                    hdim=hdim,
                                                    nlayer=nlayer,
                                                    residual=False)
        else:
            self.motion_module = None

        # SMPL Head
        self.smpl_head = SMPLTransformerDecoderHead()

        self.register_buffer('initialized', torch.tensor(False))


    def forward(self, batch, **kwargs):
        image  = batch['img']
        center = batch['center']
        scale  = batch['scale']
        img_focal = batch['img_focal']
        img_center = batch['img_center']
        bn = len(image)

        # estimate focal length, and bbox
        bbox_info = self.bbox_est(center, scale, img_focal, img_center)

        # backbone
        with autocast():
            feature = self.backbone(image[:,:,:,32:-32])
            feature = feature.float()

        # space-time module
        if self.st_module is not None:
            bb = einops.repeat(bbox_info, 'b c -> b c h w', h=16, w=12)
            feature = torch.cat([feature, bb], dim=1)

            feature = einops.rearrange(feature, '(b t) c h w -> (b h w) t c', t=16)
            feature = self.st_module(feature)
            feature = einops.rearrange(feature, '(b h w) t c -> (b t) c h w', h=16, w=12)

        # smpl_head: transformer + smpl
        pred_pose, pred_shape, pred_cam = self.smpl_head(feature)
        pred_rotmat_0 = rot6d_to_rotmat(pred_pose).reshape(-1, 24, 3, 3)

        # smpl motion module
        if self.motion_module is not None:
            bb = einops.rearrange(bbox_info, '(b t) c -> b t c', t=16)
            pred_pose = einops.rearrange(pred_pose, '(b t) c -> b t c', t=16)
            pred_pose = torch.cat([pred_pose, bb], dim=2)

            pred_pose = self.motion_module(pred_pose)
            pred_pose = einops.rearrange(pred_pose, 'b t c -> (b t) c')

        # Predictions
        rotmat_preds  = [] 
        shape_preds = []
        cam_preds   = []
        j3d_preds = []
        j2d_preds = []

        out = {}
        out['pred_cam'] = pred_cam
        out['pred_pose'] = pred_pose
        out['pred_shape'] = pred_shape
        out['pred_rotmat'] = rot6d_to_rotmat(out['pred_pose']).reshape(-1, 24, 3, 3)
        out['pred_rotmat_0'] = pred_rotmat_0
        
        s_out = self.smpl.query(out)
        j3d = s_out.joints
        j2d = self.project(j3d, out['pred_cam'], center, scale, img_focal, img_center)

        rotmat_preds.append(out['pred_rotmat'].clone())
        shape_preds.append(out['pred_shape'].clone())
        cam_preds.append(out['pred_cam'].clone())
        j3d_preds.append(j3d.clone())
        j2d_preds.append(j2d.clone())
        iter_preds = [rotmat_preds, shape_preds, cam_preds, j3d_preds, j2d_preds]

        trans_full = self.get_trans(out['pred_cam'], center, scale, img_focal, img_center)
        out['trans_full'] = trans_full
        
        return out, iter_preds
    

    def inference(self, imgfiles, boxes, img_focal=None, img_center=None, valid=None, frame=None, device='cuda'):
        nfile = len(imgfiles)
        if valid is None:
            valid = np.ones(nfile, dtype=bool)
        if frame is None:
            frame = np.arange(nfile)
        
        if isinstance(imgfiles, list):
            imgfiles = np.array(imgfiles)

        frame = frame[valid]
        boxes = boxes[valid]
        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=16)

        if len(frame_chunks) == 0:
            return

        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []
        frame = []

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            img_ck = imgfiles[frame_ck]
            results = self.inference_chunk(img_ck, boxes_ck, img_focal=img_focal, img_center=img_center)

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
        
        return results


    def inference_chunk(self, imgfiles, boxes, img_focal, img_center, device='cuda'):
        db = TrackDataset(imgfiles, boxes, img_focal=img_focal, 
                        img_center=img_center, normalization=True, dilate=1.2)

        # Results
        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []

        # To-do: efficient implementation with batch
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
                out, _ = self.forward(batch)

            if len(db) == 16:
                out = {k:v for k,v in out.items()}
            elif i == 15:
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
        
        return results


    def project(self, points, pred_cam, center, scale, img_focal, img_center, return_full=False):

        trans_full = self.get_trans(pred_cam, center, scale, img_focal, img_center)

        # Projection in full frame image coordinate
        points = points + trans_full
        points2d_full = perspective_projection(points, rotation=None, translation=None,
                        focal_length=img_focal, camera_center=img_center)

        # Adjust projected points to crop image coordinate
        # (s.t. 1. we can calculate loss in crop image easily
        #       2. we can query its pixel in the crop
        #  )
        b = scale * 200
        points2d = points2d_full - (center - b[:,None]/2)[:,None,:]
        points2d = points2d * (self.crop_size / b)[:,None,None]

        if return_full:
            return points2d_full, points2d
        else:
            return points2d


    def get_trans(self, pred_cam, center, scale, img_focal, img_center):
        b      = scale * 200
        cx, cy = center[:,0], center[:,1]            # center of crop
        s, tx, ty = pred_cam.unbind(-1)

        img_cx, img_cy = img_center[:,0], img_center[:,1]  # center of original image
        
        bs = b*s
        tx_full = tx + 2*(cx-img_cx)/bs
        ty_full = ty + 2*(cy-img_cy)/bs
        tz_full = 2*img_focal/bs

        trans_full = torch.stack([tx_full, ty_full, tz_full], dim=-1)
        trans_full = trans_full.unsqueeze(1)

        return trans_full


    def bbox_est(self, center, scale, img_focal, img_center):
        # Original image center
        img_cx, img_cy = img_center[:,0], img_center[:,1]

        # Implement CLIFF (Li et al.) bbox feature
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_cx, cy - img_cy, b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / img_focal.unsqueeze(-1) * 2.8 
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * img_focal) / (0.06 * img_focal)  

        return bbox_info


    def set_smpl_mean(self, ):
        SMPL_MEAN_PARAMS = 'data/smpl/smpl_mean_params.npz'

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def freeze_modules(self):
        frozen_modules = self.frozen_modules

        if frozen_modules is None:
            return

        for module in frozen_modules:
            if type(module) == torch.nn.parameter.Parameter:
                module.requires_grad = False
            else:
                module.eval()
                for p in module.parameters(): p.requires_grad=False

        return


    def unfreeze_modules(self, ):
        frozen_modules = self.frozen_modules

        if frozen_modules is None:
            return

        for module in frozen_modules:
            if type(module) == torch.nn.parameter.Parameter:
                module.requires_grad = True
            else:
                module.train()
                for p in module.parameters(): p.requires_grad=True

        self.frozen_modules = None

        return


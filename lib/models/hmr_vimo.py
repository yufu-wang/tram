import numpy as np
import einops
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import default_collate

from lib.utils.geometry import perspective_projection
from lib.utils.geometry import rot6d_to_rotmat_hmr2 as rot6d_to_rotmat
from lib.datasets.track_dataset import TrackDataset

from .vit import vit_base, vit_huge
from .components.pose_transformer import TransformerDecoder
from .smpl import SMPL

if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
    print('Using autocast')
else:
    # dummy GradScaler for PyTorch < 1.6 OR no cuda
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class SMPLTransformerDecoderHead(nn.Module):
    """ HMR2 Cross-attention based SMPL Transformer decoder
    """
    def __init__(self, ):
        super().__init__()
        transformer_args = dict(
            depth = 6,  # originally 6
            heads = 8,
            mlp_dim = 1024,
            dim_head = 64,
            dropout = 0.0,
            emb_dropout = 0.0,
            norm = "layer",
            context_dim = 1280,
            num_tokens = 1,
            token_dim = 1,
            dim = 1024
            )
        self.transformer = TransformerDecoder(**transformer_args)

        dim = 1024
        npose = 24*6
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load('data/smpl/smpl_mean_params.npz')
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

        
    def forward(self, x, **kwargs):

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # Pass through transformer
        token = torch.zeros(batch_size, 1, 1).to(x.device)
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1) # (B, C)

        # Readout from token_out
        pred_pose = self.decpose(token_out)  + init_body_pose
        pred_shape = self.decshape(token_out)  + init_betas
        pred_cam = self.deccam(token_out)  + init_cam

        return pred_pose, pred_shape, pred_cam


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
            print(f'Using Temporal Attention space-time: {nlayer} layers {hdim} dim.')
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
            print(f'Using Temporal Attention motion layer: {nlayer} layers {hdim} dim.')
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
    

    def inference(self, imgfiles, boxes, img_focal, img_center, device='cuda'):
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



class temporal_attention(nn.Module):
    def __init__(self, in_dim=1280, out_dim=1280, hdim=512, nlayer=6, nhead=4, residual=False):
        super(temporal_attention, self).__init__()
        self.hdim = hdim
        self.out_dim = out_dim
        self.residual = residual
        self.l1 = nn.Linear(in_dim, hdim)
        self.l2 = nn.Linear(hdim, out_dim)

        self.pos_embedding = PositionalEncoding(hdim, dropout=0.1)
        TranLayer = nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead, dim_feedforward=1024,
                                               dropout=0.1, activation='gelu')
        self.trans = nn.TransformerEncoder(TranLayer, num_layers=nlayer)
        
        nn.init.xavier_uniform_(self.l1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l2.weight, gain=0.01)

    def forward(self, x):
        x = x.permute(1,0,2)  # (b,t,c) -> (t,b,c)

        h = self.l1(x)
        h = self.pos_embedding(h)
        h = self.trans(h)
        h = self.l2(h)

        if self.residual:
            x = x[..., :self.out_dim] + h
        else:
            x = h
        x = x.permute(1,0,2)

        return x

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

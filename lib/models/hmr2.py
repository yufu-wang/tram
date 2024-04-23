import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from lib.utils import rotation_conversions as geo
from lib.utils.geometry import perspective_projection
from lib.utils.geometry import rot6d_to_rotmat_hmr2 as rot6d_to_rotmat

from .vit import vit_huge
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
    """ Cross-attention based SMPL Transformer decoder
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


class HMR2(nn.Module):
    def __init__(self, device='cpu', **kwargs):

        super(HMR2, self).__init__()
        self.device = device
        self.crop_size = 256

        # SMPL
        self.smpl = SMPL()      

        # Backbone
        self.backbone = vit_huge()

        # SMPL Head
        self.smpl_head = SMPLTransformerDecoderHead()

        self.register_buffer('initialized', torch.tensor(False))
        print('Using HMR2.0 ...')


    def forward(self, batch, **kwargs):
        image  = batch['img']
        center = batch['center']
        scale  = batch['scale']
        img_focal = batch['img_focal']
        img_center = batch['img_center']

        # backbone
        with autocast():
            feature = self.backbone(image[:,:,:,32:-32])
            feature = feature.float()

        # smpl_head: transformer + smpl
        pred_pose, pred_shape, pred_cam = self.smpl_head(feature)
      
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



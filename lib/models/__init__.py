import os
import torch
from yacs.config import CfgNode as CN
from .hmr_vimo import HMR_VIMO


def get_default_config():
    cfg_file = os.path.join(
        os.path.dirname(__file__),
        'configs/config_vimo.yaml'
        )

    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    return cfg


def get_hmr_vimo(checkpoint=None, device='cuda'):
    cfg = get_default_config()
    cfg.device = device
    model = HMR_VIMO(cfg)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location='cpu')
        _ = model.load_state_dict(ckpt['model'], strict=False)

    model = model.to(device)
    _ = model.eval()

    return model


import os
import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter


from lib.core.config import parse_args
from lib.core.losses import compile_criterion
from lib.utils.utils import prepare_output_dir, create_logger
from lib.trainer import Trainer

from lib.get_videoloader import get_dataloaders
from lib.models.hmr_vimo import HMR_VIMO


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    # create logger
    logger = create_logger(cfg.LOGDIR, phase='train')
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')
    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # Dataloaders
    data_loaders = get_dataloaders(cfg)

    # Compile Loss 
    criterion = compile_criterion(cfg)

    # Networks and optimizers
    model = HMR_VIMO(cfg=cfg)
    checkpoint = cfg.MODEL.CHECKPOINT
    state_dict = torch.load(checkpoint, map_location=cfg.DEVICE, weights_only=True)
    _ = model.load_state_dict(state_dict['state_dict'], strict=False)

    model = model.to(cfg.DEVICE)
    model.frozen_modules = [model.backbone]
    model.freeze_modules()

    logger.info(f'Loaded pretrained checkpoint {checkpoint}')
    logger.info(f'Freeze pretrained backbone')

    if cfg.TRAIN.MULTI_LR:
        params = [{'params': [p for p in model.smpl_head.parameters() if p.requires_grad]}]

        if cfg.MODEL.MOTION_MODULE:
            params.append({'params': [p for p in model.motion_module.parameters() if p.requires_grad],
                           'lr':cfg.TRAIN.LR2})
            
        if cfg.MODEL.ST_MODULE:
            params.append({'params': [p for p in model.st_module.parameters() if p.requires_grad], 
                           'lr':cfg.TRAIN.LR2})
        
        optimizer = torch.optim.AdamW(params, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)
        
        logger.info(f'Using multiple learning rates:[{cfg.TRAIN.LR}, {cfg.TRAIN.LR2}] and WD: {cfg.TRAIN.WD}')
    else:
        optimizer = torch.optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], 
                                    lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)


    # ========= Start Training ========= #
    Trainer(
        cfg=cfg,
        data_loaders=data_loaders,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        writer=writer,
        lr_scheduler=None,
    ).train()



if __name__ == '__main__':
    cfg = parse_args()
    cfg = prepare_output_dir(cfg)

    main(cfg)

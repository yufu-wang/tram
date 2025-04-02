import torch
import shutil
import logging
from tqdm import tqdm
import numpy as np
import os.path as osp

logger = logging.getLogger(__name__)


class BaseTrainer():
    def __init__(
            self, cfg, data_loaders,
            model, criterion, optimizer,
            lr_scheduler=None, writer=None
        ):

        # Base trainer
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.train_loader, self.test_loader = data_loaders

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer

        self.lr_scheduler = self.get_scheduler(cfg)

        # Model specific trainer
        self._init_fn()

        # Training parameters
        self.start_epoch = cfg.TRAIN.START_EPOCH
        self.end_epoch = cfg.TRAIN.END_EPOCH
        self.logdir = cfg.LOGDIR

        self.epoch = 0
        self.global_step = 0
        self.best_performance = None
        self.loss_meter = AverageMeter()

        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        # Load Checkpoint if provided or latest available (by cfg)
        self.load_checkpoint()


    def train_one_epoch(self):
        raise NotImplementedError('You need to provide a train_one_epoch method')

    def validate(self):
        self.performance_type = None
        return None

    def _init_fn(self):
        return

    def train(self):
        _ = self.validate()

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train_one_epoch()

            self.train_loader.re_init()

            if self.should_break():
                break


    def save_checkpoint(self, batch, performance=None, index=''):
        epoch = self.epoch

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'batch': batch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'dataperm': self.train_loader.sampler.dataset_perm,
            'performance': performance,
            'best_performance': self.best_performance
        }

        
        latest = self.logdir + f'/checkpoint.pth.tar'
        torch.save(checkpoint, latest)

        if index == '_best':
            filename = self.logdir + f'/checkpoint{index}.pth.tar'
            torch.save(checkpoint, filename)


    def load_checkpoint(self):
        if self.cfg.TRAIN.RESUME is not None:
            check_path = self.cfg.TRAIN.RESUME
            is_finetune = self.cfg.TRAIN.IS_FINETUNE
            self.resume(check_path, is_finetune)

        elif self.cfg.TRAIN.LOAD_LATEST:
            check_path = self.logdir + '/checkpoint.pth.tar'
            self.resume(check_path)

        else:
            logger.info('Starting from scratch.')


    def resume(self, check_path, is_finetune=False):
        if osp.isfile(check_path):
            checkpoint = torch.load(check_path)
            self.model.load_state_dict(checkpoint['model'], strict=False)

            logger.info(f"=> Loaded checkpoint '{check_path}'. ")

            if not is_finetune:
                self.epoch = checkpoint['epoch']
                self.start_epoch = checkpoint['epoch']
                self.global_step = checkpoint['global_step']
                self.best_performance = checkpoint['best_performance']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.train_loader.load_checkpoint(checkpoint['batch'], checkpoint['dataperm'])

                performance = checkpoint['performance']
                logger.info(f"=> Loaded previous optimizer/dataset schedule.")
                logger.info(f"=> (epoch {self.start_epoch}, performance {performance})")

        else:
            logger.info(f"=> No checkpoint found at '{check_path}'. Starting from scratch.")


    def check_performance(self, performance, batch, save_best=True):
        ptype = self.performance_type

        # Check if best
        if ptype is None or performance is None:
            is_best = False
        elif ptype == 'min':
            if self.best_performance is None:
                self.best_performance = 1e12
            is_best = performance < self.best_performance
        else:
            if self.best_performance is None:
                self.best_performance = -1e12
            is_best = performance > self.best_performance

        # Log if best
        if is_best:
            logger.info('Best performance achived, logging it!')
            self.best_performance = performance
            
            with open(self.logdir + '/best.txt', 'w') as f:
                f.write(str(float(performance)))


        if is_best and save_best:
            logger.info('Saving best model.')
            self.save_checkpoint(batch=batch, 
                                performance=performance,
                                index='_best')


        return is_best


    def upload_losses(self, step):
        losses = self.loss_meter.report(clear=True)
        for k, v in losses.items():
            self.writer.add_scalar(f"Loss/{k}", v, step)


    def upload_additional(self, step):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("Lr", lr, step)
        return


    def get_scheduler(self, cfg):

        if cfg.TRAIN.SCHEDULER == 'onecycle':
            max_steps = cfg.TRAIN.MAX_STEP
            lr_scheduler = get_lr_cycle_scheduler(optimizer = self.optimizer,
                                                  total_steps = max_steps)

        else:
            warmup_steps = cfg.TRAIN.WARMUP_STEPS
            schedule = cfg.TRAIN.LR_SCHEDULE
            decay = cfg.TRAIN.LR_DECAY
            lr_scheduler = get_lr_scheduler(optimizer = self.optimizer,
                                            warmup_steps = warmup_steps,
                                            schedule = schedule,
                                            decay = decay)
        return lr_scheduler


    def clip_gradient_norm(self, model, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


    def should_break(self):
        if self.global_step >= self.cfg.TRAIN.MAX_STEP:
            return True
        else:
            return False


class AverageMeter():
    def __init__(self):
        self.dict = None
        self.count = 0

    def update(self, info):
        if self.dict is None:
            self.dict = info
            self.count += 1
            return
        else:
            for k, v in info.items():
                self.dict[k] += v
            self.count += 1

    def report(self, clear=False):
        report = {k: v/self.count for k, v in self.dict.items()}
        if clear:
            self.dict = None
            self.count = 0
        return report


def get_lr_scheduler(optimizer, warmup_steps=1, schedule=[], decay=1):
    
    warmup_factor = 1./warmup_steps

    def f(x):
        if x < warmup_steps:
            alpha = float(x) / warmup_steps
            return warmup_factor * (1 - alpha) + alpha
        else:
            milestone = sum([x>=i for i in schedule])
            return decay ** milestone

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def get_lr_cycle_scheduler(optimizer, total_steps=50000):
    max_lr = optimizer.param_groups[0]['lr']

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps+10,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return scheduler




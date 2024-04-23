import os
import torch
import shutil
import logging


def prepare_output_dir(cfg):
    logfolder = cfg.EXP_NAME
    logdir = os.path.join(cfg.OUTPUT_DIR, logfolder)
    cfg.LOGDIR = logdir
    os.makedirs(logdir, exist_ok=True)

    shutil.copy(src=cfg.cfg_file, dst=f'{cfg.LOGDIR}/config.yaml')
    return cfg


def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = os.path.join(logdir, f'{phase}_log.txt')
    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    
    return logger


def move_dict_to_device(dict, device, tensor2float=False):
    for k,v in dict.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dict[k] = v.float().to(device)
            else:
                dict[k] = v.to(device)


def concatenate_dicts(dict_list, dim=0):
    rdict = dict.fromkeys(dict_list[0].keys())
    for k in rdict.keys():
        rdict[k] = torch.cat([d[k] for d in dict_list], dim=dim)
    return rdict


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from lib.core import constants


class CheckpointSampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.update_perm()

    def update_perm(self):
        if self.shuffle:
            self.dataset_perm = torch.randperm(len(self.data_source)).tolist()
            self.perm = self.dataset_perm
        else:
            self.dataset_perm = list(range(len(self.data_source)))
            self.perm = self.dataset_perm

    def __iter__(self):
        return iter(self.perm)
    
    def __len__(self):
        return len(self.perm)


class CheckpointDataLoader(DataLoader):
    
    def __init__(self, dataset, batch_size=1, shuffle=True, 
                 num_workers=4, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None):

        sampler = CheckpointSampler(dataset, shuffle)
        super(CheckpointDataLoader, self).__init__(dataset, sampler=sampler, shuffle=False, batch_size=batch_size,
                                                   pin_memory=pin_memory, timeout=timeout, worker_init_fn=None, 
                                                   collate_fn=collate_fn, num_workers=num_workers)

    def load_checkpoint(self, batch_idx, dataset_perm):

        perm = dataset_perm[self.batch_size * batch_idx:]
        self.sampler.dataset_perm = dataset_perm
        self.sampler.perm = perm

    def re_init(self):
        self.sampler.update_perm()



import torch
import numpy as np
from .video_dataset import VideoDataset
from .image_dataset import ImageDataset
    

class MixedVidDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_list, partition, **kwargs):

        self.dataset_list = dataset_list
        self.nds = len(self.dataset_list)
    
        self.datasets = [VideoDataset(ds, **kwargs) for ds in self.dataset_list]
        self.length = max([len(ds) for ds in self.datasets])
        
        self.partition = partition
        self.partition = np.array(self.partition).cumsum()


    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(self.nds):
            if p <= self.partition[i]:
                item = self.datasets[i][index % len(self.datasets[i])]

                return item

    def __len__(self):
        return self.length
    
class MixedImgDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_list, partition, **kwargs):

        self.dataset_list = dataset_list
        self.nds = len(self.dataset_list)
    
        self.datasets = [ImageDataset(ds, **kwargs) for ds in self.dataset_list]
        self.length = max([len(ds) for ds in self.datasets])
        
        self.partition = partition
        self.partition = np.array(self.partition).cumsum()


    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(self.nds):
            if p <= self.partition[i]:
                item = self.datasets[i][index % len(self.datasets[i])]

                return item

    def __len__(self):
        return self.length

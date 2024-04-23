import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose
import numpy as np
import cv2

from lib.core import constants
from lib.utils.imutils import crop, boxes_2_cs


class TrackDataset(Dataset):
    """
    Track Dataset Class - Load images/crops of the tracked boxes.
    """
    def __init__(self, imgfiles, boxes, crop_size=256, dilate=1.0,
                img_focal=None, img_center=None, normalization=True):
        super(TrackDataset, self).__init__()

        self.imgfiles = imgfiles
        self.crop_size = crop_size
        self.normalization = normalization
        self.normalize_img = Compose([
                            ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
                        ])

        self.boxes = boxes
        self.box_dilate = dilate
        self.centers, self.scales = boxes_2_cs(boxes)

        self.img_focal = img_focal
        self.img_center = img_center


    def __len__(self):
        return len(self.imgfiles)
    
    
    def __getitem__(self, index):
        item = {}
        imgfile = self.imgfiles[index]
        scale = self.scales[index] * self.box_dilate
        center = self.centers[index]
        img_focal = self.img_focal
        img_center = self.img_center

        img = cv2.imread(imgfile)[:,:,::-1]
        img_crop = crop(img, center, scale, 
                        [self.crop_size, self.crop_size], 
                        rot=0).astype('uint8')
    
        if self.normalization:
            img_crop = self.normalize_img(img_crop)
        else:
            img_crop = torch.from_numpy(img_crop)

        if self.img_focal is None:
            orig_shape = img.shape[:2]
            img_focal = self.est_focal(orig_shape)

        if self.img_center is None:
            orig_shape = img.shape[:2]
            img_center = self.est_center(orig_shape)

        item['img'] = img_crop
        item['img_idx'] = torch.tensor(index).long()
        item['scale'] = torch.tensor(scale).float()
        item['center'] = torch.tensor(center).float()
        item['img_focal'] = torch.tensor(img_focal).float()
        item['img_center'] = torch.tensor(img_center).float()

        return item


    def est_focal(self, orig_shape):
        h, w = orig_shape
        focal = np.sqrt(h**2 + w**2)
        return focal

    def est_center(self, orig_shape):
        h, w = orig_shape
        center = np.array([w/2., h/2.])
        return center



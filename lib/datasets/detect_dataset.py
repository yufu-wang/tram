import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose
import numpy as np
import cv2

from lib.core import constants
from lib.utils.imutils import crop, boxes_2_cs


class DetectDataset(Dataset):
    """
    Detection Dataset Class - Handles data loading from detections.
    """
    def __init__(self, img, boxes, crop_size=256, dilate=1.2,
                img_focal=None, img_center=None, normalize=True):
        super(DetectDataset, self).__init__()

        self.img = img
        self.crop_size = crop_size
        self.orig_shape = img.shape[:2]
        self.normalize = normalize
        self.normalize_img = Compose([
                            ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
                        ])

        self.boxes = boxes
        self.box_dilate = dilate
        self.centers, self.scales = boxes_2_cs(boxes)

        if img_focal is None:
            self.img_focal = self.est_focal(self.orig_shape)
        else:
            self.img_focal = img_focal

        if img_center is None:
            self.img_center = self.est_center(self.orig_shape)
        else:
            self.img_center = img_center


    def __getitem__(self, index):
        item = {}
        scale = self.scales[index] * self.box_dilate
        center = self.centers[index]
        img_focal = self.img_focal
        img_center = self.img_center

        img = crop(self.img, center, scale, 
                  [self.crop_size, self.crop_size], rot=0).astype('uint8')
        origin_crop = img.copy()
        if self.normalize:
            img = self.normalize_img(img)


        item['img'] = img
        item['origin_crop'] = origin_crop
        item['scale'] = torch.tensor(scale).float()
        item['center'] = torch.tensor(center).float()
        item['img_focal'] = torch.tensor(img_focal).float()
        item['img_center'] = torch.tensor(img_center).float()
        item['orig_shape'] = torch.tensor(self.orig_shape).float()

        return item


    def __len__(self):
        return len(self.boxes)


    def est_focal(self, orig_shape):
        h, w = orig_shape
        focal = np.sqrt(h**2 + w**2)
        return focal

    def est_center(self, orig_shape):
        h, w = orig_shape
        center = np.array([w/2., h/2.])
        return center



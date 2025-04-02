import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from os.path import join
import logging

from lib.core import constants, config
from lib.utils.imutils import crop, transform, rot_aa

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, dataset, is_train=True, crop_size=256):
        super(BaseDataset, self).__init__()
        
        self.is_train = is_train

        self.dataset = dataset
        self.data = np.load(config.DATASET_FILES[is_train][dataset])

        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.imgname = self.data['imgname'].astype(np.string_)
        self.crop_size = crop_size

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.sc = 1.0

        # Get camera intrinsic, if available
        try:    
            self.img_focal = self.data['img_focal']
            self.img_center = self.data['img_center']
            self.has_camcalib = True
            print(dataset, 'has camera intrinsics')
        except KeyError:
            self.has_camcalib = False

        self.length = self.scale.shape[0]

        if 'mpi3d_vid' in dataset:
            self.invalid = self.data['invalid']
        elif '3dpw' in dataset:
            self.invalid = self.detect_invalid_section()
        elif 'bedlam' in dataset:
            self.invalid = self.data['invalid']
        else:
            self.invalid = np.zeros(len(self))
        

    def rgb_processing(self, rgb_img, center, scale):
        """Process rgb image and do augmentation."""
        
        rgb_img = crop(rgb_img, center, scale, 
                    [self.crop_size, self.crop_size], rot=0)
        
        return rgb_img.astype('uint8')


    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Load image
        imgname = str(self.imgname[index], encoding='utf-8')
        imgname = join(self.img_dir, imgname)
        
        try:
            img = cv2.imread(imgname).copy().astype(np.float32)
        except TypeError:
            logger.info(f"cv2 loading error image={imgname}")

        # bug
        if self.dataset=='bedlam' and self.img_center[index][0] == 360: # supposely a bedlam tall image 
            if img.shape[1] != 720:  # but the given image is not ...
                img = np.transpose(img, [1,0,2])[:,::-1,:].copy()

        # Process image
        invalid = self.invalid[index]
        if invalid:
            img = np.zeros([256, 256, 3]).astype('uint8')
        else:
            try:
                img = self.rgb_processing(img, center, scale)
            except:
                img = np.zeros([256, 256, 3]).astype('uint8')


        # Store unnormalize image
        item['img'] = img
        item['imgname'] = imgname

        return item


    def __len__(self):
        return len(self.imgname)
    

    def detect_invalid_section(self,):
        center = self.center
        size = self.scale * 200
        shape = self.data['img_shape']

        xy1 = center - size[:,None]/2
        xy2 = center + size[:,None]/2
        bbox = np.concatenate([xy1, xy2], axis=1)
        invalid = (bbox[:,2]<0) + (bbox[:,3]<0) + (bbox[:,0]>shape[:,0]) + (bbox[:,1]>shape[:,1])
        invalid = invalid + (self.data['valid']!=1)
        return invalid


import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import cv2
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data_config import ROOT
from lib.datasets.base_dataset import BaseDataset


SEED_VALUE = 0
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

# Datasets
# ds_list = ['3dpw_vid', 'h36m_vid', 'bedlam_vid']
# ds_list = ['3dpw_vid_test', 'emdb_1']
ds_list = ['bedlam_vid']

save_dir = {'h36m_vid': ROOT + '/h36m/crops',
            '3dpw_vid': ROOT + '/3dpw/crops',
            'bedlam_vid': ROOT + '/bedlam_30fps/crops',
            '3dpw_vid_test': ROOT + '/3dpw/crops_test',
            'emdb_1': ROOT + '/emdb/crops_1'}

for ds in ds_list:
    print(f'Processing (crop) {ds} ...')

    # DATASET
    db = BaseDataset(ds, is_train=True, crop_size=256)
    loader = DataLoader(db, batch_size=64, num_workers=15, shuffle=False)

    imgdir = save_dir[ds]
    os.makedirs(imgdir, exist_ok=True)

    c = 0
    for i, batch in enumerate(tqdm(loader)):
        images = batch['img'].numpy()
        for img in images:
            cv2.imwrite(f'{imgdir}/{c:08d}.jpg', img)
            c += 1

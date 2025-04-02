"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

# You should have directores like this
# - datasets
# ---- dataset_ann
# ---- 3dpw
# ---- h36m
# ---- bedlam_30fps
# ---- emdb

# Please change these two lines for your directories
ROOT = './datasets'
DATASET_NPZ_PATH = './datasets/dataset_ann'

H36M_ROOT         = join(ROOT, 'h36m')
PW3D_ROOT         = join(ROOT, '3dpw')
BEDLAM_ROOT       = join(ROOT, 'bedlam_30fps')
EMDB_ROOT         = join(ROOT, 'emdb')

# Path to test/train npz files
DATASET_FILES = [ {
                   'emdb_1': join(DATASET_NPZ_PATH , 'emdb_1.npz'),
                   '3dpw_vid_test': join(DATASET_NPZ_PATH , '3dpw_vid_test.npz'),
                  },

                  {
                   '3dpw_vid': join(DATASET_NPZ_PATH , '3dpw_vid_train.npz'),
                   'h36m_vid': join(DATASET_NPZ_PATH , 'h36m_train.npz'),
                   'bedlam_vid': join(DATASET_NPZ_PATH , 'bedlam_vid.npz'),
                  }
                ]

DATASET_FOLDERS = {'h36m_vid': H36M_ROOT,
                   '3dpw_vid': PW3D_ROOT,
                   'bedlam_vid': BEDLAM_ROOT,
                   'emdb_1': EMDB_ROOT,
                   '3dpw_vid_test': PW3D_ROOT,
                }


PASCAL_OCCLUDERS = 'data/pascal_occluders.pkl'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/smpl/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/smpl/J_regressor_h36m.npy'
SMPL_MEAN_PARAMS = 'data/smpl/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
from glob import glob
from lib.pipeline.tools import video2frames, detect_segment_track


parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./example_video.mov', help='input video')
parser.add_argument("--visualization", action='store_true', help='save deva vos for visualization')
args = parser.parse_args()

# File and folders
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'{root}/{seq}'
img_folder = f'{seq_folder}/images'
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)
print(f'Running on {file} ...')

##### Extract Frames #####
print('Extracting frames ...')
nframes = video2frames(file, img_folder)

##### Detection + SAM + DEVA-Track-Anything #####
print('Detect, Segment, and Track ...')
save_vos = args.visualization
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
boxes_, masks_, tracks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25, 
                                               min_size=100, save_vos=save_vos)
np.save(f'{seq_folder}/boxes.npy', boxes_)
np.save(f'{seq_folder}/masks.npy', masks_)
np.save(f'{seq_folder}/tracks.npy', tracks_)


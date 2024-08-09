import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
from glob import glob
from lib.pipeline import visualize_tram

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='./example_video.mov', help='input video')
parser.add_argument('--bin_size', type=int, default=-1, help='rasterization bin_size; set to [64,128,...] to increase speed')
parser.add_argument('--floor_scale', type=int, default=3, help='size of the floor')
parser.add_argument('--ground_estimate', type=int, default=[10, -10], nargs='+', 
                    help='Frist/last N frames to estiamte ground. Set to 0 to use all frames.')

args = parser.parse_args()

# File and folders
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'results/{seq}'
img_folder = f'{seq_folder}/images'
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

##### Combine camera & human motion #####
# Set the frame to estimate the ground plane
if args.ground_estimate[0] == 0 :
    contact_frames = np.arange(len(imgfiles))
else:
    idx = np.concatenate([np.arange(args.ground_estimate[0]), 
                          np.arange(args.ground_estimate[1],0)])  
    contact_frames = np.arange(len(imgfiles))[idx]

# Render video
print('Visualize results ...')
visualize_tram(seq_folder, contact_frames=contact_frames, floor_scale=args.floor_scale, bin_size=args.bin_size)

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import ffmpeg

from data_config import ROOT

def mp4_to_jpg(v, frame_folder):
    cap = cv2.VideoCapture(v)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite(f'{frame_folder}/{count:04d}.jpg', frame)
            count += 1
        else:
            break
    cap.release()

def mp4_to_jpg_ffmpeg(v, frame_folder):
    (
        ffmpeg.input(v)
        .output(f'{frame_folder}/%04d.jpg', 
                vf='fps=30', 
                start_number=0,
                qscale=1)
        .run(quiet=True)
    )

root = f'{ROOT}/bedlam_30fps'
mp4_scene = sorted(glob(f'{root}/mp4/*'))

for scene in mp4_scene:
    mp4_files = sorted(glob(f'{scene}/mp4/*.mp4'))
    for file in tqdm(mp4_files):
        s = file.split('/')[-3]
        seq = file.split('/')[-1][:-4]
        
        img_outdir = f'{root}/bedlam_data/images/{s}/jpg/{seq}'
        os.makedirs(img_outdir, exist_ok=True)
        
        # mp4_to_jpg(file, img_outdir)
        mp4_to_jpg_ffmpeg(file, img_outdir)


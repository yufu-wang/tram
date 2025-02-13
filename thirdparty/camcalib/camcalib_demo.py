# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import sys
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from skimage.io import imsave
import matplotlib.pyplot as plt

from scipy.special import softmax

sys.path.append('')
from camcalib.vis_utils import show_horizon_line
from camcalib.model import CameraRegressorNetwork
from camcalib.pano_dataset import CameraRegressorDataset, ImageFolder
from camcalib.cam_utils import bins2vfov, bins2pitch, bins2roll, convert_preds_to_angles

from pare.utils.image_utils import denormalize_images
from pare.utils.train_utils import load_pretrained_model

CKPT = 'data/camcalib/checkpoints/camcalib_sa_biased_l2.ckpt'
# CKPT = '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/logs/cam_reg/pano_scalenet_v3_softargmax_l2_lw10/22-02-2021_18-32-07_pano_scalenet_v3_softargmax_l2_lw10_train/tb_logs/0/checkpoints/epoch=26-step=337742.ckpt'

@torch.no_grad()
def main(args):
    img_folder = args.img_folder
    out_folder = args.out_folder
    loss_type = args.loss

    if img_folder == '-':
        img_folder = None

    if args.ckpt == '':
        args.ckpt = CKPT

    if img_folder is not None:
        image_list = sorted([os.path.join(img_folder, x) for x in os.listdir(img_folder)
                             if (x.endswith('jpg') or x.endswith('jpeg') or x.endswith('png')) and not x.startswith('.')])

        val_dataset = ImageFolder(image_list)
    elif args.dataset is not None:
        from pare.core.config import DATASET_FILES, DATASET_FOLDERS
        image_list = [os.path.join(DATASET_FOLDERS[args.dataset], x)
                      for x in np.load(DATASET_FILES[0][args.dataset])['imgname']]

        val_dataset = ImageFolder(image_list)
    else:
        val_dataset = CameraRegressorDataset(
            dataset='pano_scalenet',
            is_train=False,
            loss_type=loss_type,
        )

    device = 'cuda'

    model = CameraRegressorNetwork(
        backbone='resnet50',
        num_fc_layers=1,
        num_fc_channels=1024,
    ).to(device)

    ckpt = torch.load(args.ckpt)
    model = load_pretrained_model(model, ckpt['state_dict'], remove_lightning=True, strict=True)

    logger.info('Loaded pretrained model')

    model.eval()

    output_path = out_folder

    os.makedirs(output_path, exist_ok=True)

    focal_length = []

    logger.info('Running CamCalib')

    for idx, batch in enumerate(tqdm(val_dataset)):

        img_fname = batch['imgname']
        results_file = os.path.join(output_path, img_fname.split('/')[-1] + '.pkl')

        images = batch['img'].unsqueeze(0).to(device).float()

        preds = model(images)

        pred_distributions = preds

        batch_img = images
        batch_img = denormalize_images(batch_img) * 255
        batch_img = np.transpose(batch_img.cpu().numpy(), (0, 2, 3, 1))

        extract = lambda x: x.detach().cpu().numpy().squeeze()
        img = batch_img[0].copy()

        if loss_type in ('kl', 'ce'):
            pred_vfov, pred_pitch, pred_roll = map(extract, preds)
            pred_vfov, pred_pitch, pred_roll = convert_preds_to_angles(
                pred_vfov, pred_pitch, pred_roll, loss_type=loss_type,
                return_type='np',
            )
        else:
            preds = convert_preds_to_angles(
                *preds, loss_type=loss_type,
            )
            pred_vfov = extract(preds[0])
            pred_pitch = extract(preds[1])
            pred_roll = extract(preds[2])

        orig_img_w, orig_img_h = batch['orig_shape']

        pred_f_pix = orig_img_h / 2. / np.tan(pred_vfov / 2.)

        pitch = np.degrees(pred_pitch)
        roll = np.degrees(pred_roll)
        vfov = np.degrees(pred_vfov)

        results = {
            'vfov': pred_vfov,
            'f_pix': pred_f_pix,
            'pitch': pred_pitch,
            'roll': pred_roll,
        }

        focal_length.append(pred_f_pix)

        if img_folder is None and args.dataset is None:
            gt_res = {
                'gt_vfov': gt_vfov,
                'gt_f_pix': gt_f_pix,
                'gt_pitch': gt_pitch,
                'gt_roll': gt_roll,
            }

            results = results.update(gt_res)

        img, _ = show_horizon_line(img.copy(), pred_vfov, pred_pitch, pred_roll, focal_length=pred_f_pix,
                                   debug=True, color=(255, 0, 0), width=3, GT=False)

        if img_folder is not None or args.dataset is not None:
            pass
        else:
            gt_vfov = batch['vfov']
            gt_pitch = batch['pitch']
            gt_roll = batch['roll']

            gt_f_pix = img.shape[0] / 2. / np.tan(gt_vfov / 2.)

            extract = lambda x: x.detach().cpu().numpy()

            gt_vfov, gt_pitch, gt_roll = map(extract, (gt_vfov, gt_pitch, gt_roll))

            img, _ = show_horizon_line(img.copy(), gt_vfov, gt_pitch, gt_roll,
                                       debug=True, color=(0, 0, 255), width=3, GT=True)


        joblib.dump(results, results_file)

        if args.show:
            # plt.title

            from camcalib.cam_utils import roll_new_bins_centers as roll_bins_centers
            from camcalib.cam_utils import pitch_bins_centers, vfov_bins_centers

            fig = plt.figure(figsize=(18, 7), constrained_layout=True)
            gs = fig.add_gridspec(3, 2, width_ratios=[20, 10], height_ratios=[1, 1, 1])

            ax_inp_image = fig.add_subplot(gs[:, 0])
            ax_inp_image.imshow(img)
            ax_inp_image.set_title(f'fov: {vfov:.1f}, pitch: {pitch:.1f}, roll: {roll:.1f}, fpx: {pred_f_pix:.1f}')

            to_numpy = lambda x: softmax(x[0].detach().cpu().numpy())

            vfov_dist, pitch_dist, roll_dist = map(to_numpy, pred_distributions)

            ax_vfov = fig.add_subplot(gs[0, 1])
            # ax_vfov.plot(np.degrees(vfov_bins_centers), vfov_dist)
            ax_vfov.bar(np.degrees(vfov_bins_centers), vfov_dist)
            ax_vfov.axvline(x=np.degrees(pred_vfov), color='r')
            ax_vfov.set_title('VFOV distribution')

            ax_pitch = fig.add_subplot(gs[1, 1])
            # ax_pitch.plot(np.degrees(pitch_bins_centers), pitch_dist)
            ax_pitch.bar(np.degrees(pitch_bins_centers), pitch_dist)
            ax_pitch.axvline(x=np.degrees(pred_pitch), color='r')
            ax_pitch.set_title('PITCH distribution')

            ax_roll = fig.add_subplot(gs[2, 1])
            # ax_roll.plot(np.degrees(roll_bins_centers), roll_dist)
            ax_roll.bar(np.degrees(roll_bins_centers), roll_dist)
            ax_roll.axvline(x=np.degrees(pred_roll), color='r')
            ax_roll.set_title('ROLL distribution')

            plt.savefig(os.path.join(output_path, img_fname.split('/')[-1]+'_fig.png'))
            # plt.show()
            # plt.show(block=False)
            # plt.pause(0.03)
            plt.close('all')

        if not args.no_save:
            imsave(os.path.join(output_path, img_fname.split('/')[-1]), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_folder', help='input image folder', type=str)
    parser.add_argument('--out_folder', help='output folder', type=str)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--loss', default='softargmax_l2')
    parser.add_argument('--ckpt', default=CKPT)
    parser.add_argument('--show', help='visualize raw network predictions', action='store_true')
    parser.add_argument('--no_save', help='do not save output images', action='store_true')

    args = parser.parse_args()

    main(args)
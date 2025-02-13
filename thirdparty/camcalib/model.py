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

import torch
import torch.nn as nn
import math
from collections import OrderedDict
import torchvision.transforms as transforms
from .resnet import resnet50
from .cam_utils import convert_preds_to_angles
# from pare.models.backbone import *
# from pare.models.backbone.utils import get_backbone_info


def get_backbone_info(backbone):
    info = {
        'resnet18': {'n_output_channels': 512, 'downsample_rate': 4},
        'resnet34': {'n_output_channels': 512, 'downsample_rate': 4},
        'resnet50': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet50_adf_dropout': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet50_dropout': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet101': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet152': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnext50_32x4d': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnext101_32x8d': {'n_output_channels': 2048, 'downsample_rate': 4},
        'wide_resnet50_2': {'n_output_channels': 2048, 'downsample_rate': 4},
        'wide_resnet101_2': {'n_output_channels': 2048, 'downsample_rate': 4},
        'mobilenet_v2': {'n_output_channels': 1280, 'downsample_rate': 4},
        'hrnet_w32': {'n_output_channels': 480, 'downsample_rate': 4},
        'hrnet_w48': {'n_output_channels': 720, 'downsample_rate': 4},
        # 'hrnet_w64': {'n_output_channels': 2048, 'downsample_rate': 4},
        'dla34': {'n_output_channels': 512, 'downsample_rate': 4},
    }
    return info[backbone]


def load_pretrained_model(model, state_dict, strict=False, overwrite_shape_mismatch=True, remove_lightning=False):
    if remove_lightning:
        # logger.warning(f'Removing "model." keyword from state_dict keys..')
        pretrained_keys = state_dict.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = state_dict[pk]
            else:
                new_state_dict[pk] = state_dict[pk]

        model.load_state_dict(new_state_dict, strict=strict)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = state_dict.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
                        # logger.warning(f'size mismatch for \"{pk}\": copying a param with shape {state_dict[pk].shape} '
                                    #    f'from checkpoint, the shape in current model is {model_state_dict[pk].shape}')

                        if pk == 'model.head.fc1.weight':
                            updated_pretrained_state_dict[pk] = torch.cat(
                                [state_dict[pk], state_dict[pk][:,-7:]], dim=-1
                            )
                            # logger.warning(f'Updated \"{pk}\" param to {updated_pretrained_state_dict[pk].shape} ')
                            continue
                        else:
                            del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError('there are shape inconsistencies between pretrained ckpt and current ckpt')
    return model


def rotation_about_x(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]])


def rotation_about_y(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[cos, 0, sin, 0], [0, 1, 0, 0], [-sin, 0, cos, 0], [0, 0, 0, 1]])


def rotation_about_z(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


class CameraRegressorNetwork(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            num_fc_layers=1,
            num_fc_channels=1024,
            num_out_channels=256,
    ):
        super(CameraRegressorNetwork, self).__init__()
        self.backbone = eval(backbone)(pretrained=False)

        self.num_out_channels = num_out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        out_channels = get_backbone_info(backbone)['n_output_channels']

        assert num_fc_layers > 0, 'Number of FC layers should be more than 0'
        if num_fc_layers == 1:
            self.fc_vfov = nn.Linear(out_channels, num_out_channels)
            self.fc_pitch = nn.Linear(out_channels, num_out_channels)
            self.fc_roll = nn.Linear(out_channels, num_out_channels)

            nn.init.normal_(self.fc_vfov.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_vfov.bias, 0)

            nn.init.normal_(self.fc_pitch.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_pitch.bias, 0)

            nn.init.normal_(self.fc_roll.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_roll.bias, 0)

        else:
            self.fc_vfov = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)
            self.fc_pitch = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)
            self.fc_roll = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)

        min_size = 600
        max_size = 1000
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(min_size, max_size=max_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def _get_fc_layers(self, num_layers, num_channels, inp_channels):
        modules = []

        for i in range(num_layers):
            if i == 0:
                modules.append(nn.Linear(inp_channels, num_channels))
            elif i == num_layers - 1:
                modules.append(nn.Linear(num_channels, self.num_out_channels))
            else:
                modules.append(nn.Linear(num_channels, num_channels))

        return nn.Sequential(*modules)


    def forward(self, images, return_angles=True, transform_data=True):
        if transform_data:
            if isinstance(images, torch.Tensor):
                images = images.numpy()
            images = self.data_transform(images)

        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(self._get_device())
        x = self.backbone(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        vfov = self.fc_vfov(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)

        if return_angles:
            preds = convert_preds_to_angles(vfov, pitch, roll, loss_type='softargmax_l2')

            extract = lambda x: x.detach().cpu().numpy().squeeze()
            vfov = extract(preds[0])
            pitch = extract(preds[1])
            roll = extract(preds[2])

        return [vfov, pitch, roll]
    

    def to_gravity_cam(self, pitch, roll):
        Rpitch = rotation_about_x(-pitch)[:3, :3]
        Rroll = rotation_about_y(roll)[:3, :3]
        R_g = Rpitch @ Rroll
        return R_g


    def load_ckpt(self, ckpt):
        if isinstance(ckpt, str):
            ckpt = torch.load(ckpt)
            
        state_dict = ckpt['state_dict']
        model = load_pretrained_model(self, state_dict, remove_lightning=True, strict=True)
        _ = model.eval()

        return model


    def _get_device(self,):
        return self.backbone.conv1.weight.device


def test_model():
    backbones = ['resnet50', 'resnet34']
    num_fc_layers = [1, 2, 3]
    num_fc_channels = [256, 512, 1024]
    img_size = [(224, 224), (480,640), (500, 450)]
    from itertools import product

    # print(list(product(backbones, num_fc_layers, num_fc_channels)))
    inp = torch.rand(1, 3, 128, 128)

    for (b, nl, nc, im_size) in list(product(backbones, num_fc_layers, num_fc_channels, img_size)):
        print('backbone', b, 'n_f_layer', nl, 'n_ch', nc, 'im_size', im_size)
        inp = torch.rand(1, 3, *im_size)
        model = CameraRegressorNetwork(backbone=b, num_fc_layers=nl, num_fc_channels=nc)
        out = model(inp)

        breakpoint()
        print('vfov', out[0].shape, 'pitch', out[1].shape, 'roll', out[2].shape)


if __name__ == '__main__':
    test_model()

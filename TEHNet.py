import logging
import torch
import torch.nn as nn

from manopth.manolayer import ManoLayer
from pytorch3d.transforms import quaternion_to_axis_angle
from resnet50 import ResNet
from unet import UNet


# TwoEventHandsNet = UNet + ResNet50 + ManoLayer
class TEHNet(nn.Module):
    def __init__(self, use_unet=True):
        super(TEHNet, self).__init__()
        self.use_unet = use_unet

        if use_unet:
            self.unet = UNet()

        self.resnet = ResNet(use_unet=use_unet)
        # self.layer_mano_right = ManoLayer(flat_hand_mean=False, side='right', mano_root='mano', use_pca=False,
        #                                   root_rot_mode='axisang', joint_rot_mode='axisang')
        # self.layer_mano_left = ManoLayer(flat_hand_mean=False, side='left', mano_root='mano', use_pca=False,
        #                                  root_rot_mode='axisang', joint_rot_mode='axisang')

    def load_resnet(self, path, device):
        self.resnet.load_state_dict(
            torch.load(path, map_location=device)
        )
        logging.info(f'ResNet model loaded from {path}')

        # fix weights if desired
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

    def load_unet(self, path, device):
        self.unet.load_state_dict(
            torch.load(path, map_location=device)
        )
        logging.info(f'UNet model loaded from {path}')

        # fix weights if desired
        # for param in self.unet.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        mask = None

        if self.use_unet:
            mask = self.unet(x)

            # LNES and mask
            x_masked = torch.cat([x, mask], dim=1)

            mano_params_resnet = self.resnet(x_masked)
        else:
            mano_params_resnet = self.resnet(x)

        # # quaternion to axis-angle
        # mano_rots_quat_right = mano_params_resnet[:, 0:64].reshape(-1, 16, 4)
        # mano_rots_quat_left = mano_params_resnet[:, 67:131].reshape(-1, 16, 4)
        # mano_trans_right = mano_params_resnet[:, 64:67]
        # mano_trans_left = mano_params_resnet[:, 131:134]
        # mano_rots_axisangle_right = quaternion_to_axis_angle(mano_rots_quat_right).reshape(-1, 48)
        # mano_rots_axisangle_left = quaternion_to_axis_angle(mano_rots_quat_left).reshape(-1, 48)
        # mano_params_manolayer = torch.cat([mano_rots_axisangle_right, mano_trans_right,
        #                                    mano_rots_axisangle_left, mano_trans_left], dim=1)
        #
        # vertices_right, joints_right = self.layer_mano_right(mano_params_manolayer[:, :48],
        #                                                      th_betas=torch.zeros((mano_params_manolayer.shape[0], 10))
        #                                                      .cuda(),
        #                                                      th_trans=mano_params_manolayer[:, 48:51])
        # vertices_left, joints_left = self.layer_mano_left(mano_params_manolayer[:, 51:99],
        #                                                   th_betas=torch.zeros((mano_params_manolayer.shape[0], 10))
        #                                                   .cuda(),
        #                                                   th_trans=mano_params_manolayer[:, 99:102])

        if self.use_unet:
            return mask, mano_params_resnet
        else:
            return mano_params_resnet

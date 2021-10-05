from manopth.manolayer import ManoLayer
import math
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms import quaternion_to_axis_angle
from pytorch3d.transforms import rotation_6d_to_matrix
from resnet50 import ResNet
import torch
import torch.nn as nn
from unet import UNet


# camera parameters
res = (240, 180)
fov = 45.0
fovy = np.radians(fov)
focal = 0.5 * res[1] / math.tan(fovy / 2.0)
mat_cam = torch.from_numpy(np.array([[focal, 0.0, -res[0] / 2.0],
                                     [0.0, -focal, -res[1] / 2.0],
                                     [0.0, 0.0, -1.0]])).float().cuda()


# TwoEventHandsNet = UNet + ResNet50 + ManoLayer
class TEHNet(nn.Module):
    def __init__(self):
        super(TEHNet, self).__init__()

        self.resnet = ResNet()
        self.unet = UNet()
        self.layer_mano_right = ManoLayer(flat_hand_mean=True, side='right', mano_root='mano', use_pca=False,
                                          root_rot_mode='axisang', joint_rot_mode='axisang')
        self.layer_mano_left = ManoLayer(flat_hand_mean=True, side='left', mano_root='mano', use_pca=False,
                                         root_rot_mode='axisang', joint_rot_mode='axisang')

    def forward(self, x):
        masks = self.unet(x)
        x_masked = torch.cat((x, masks), 1)
        mano_params = self.resnet(x_masked)

        # quaternion to axis-angle
        mano_rots_6d_right = mano_params[:, 0:96].reshape(-1, 16, 6)
        mano_rots_6d_left = mano_params[:, 99:195].reshape(-1, 16, 6)
        mano_trans_right = mano_params[:, 96:99]
        mano_trans_left = mano_params[:, 195:198]
        mano_rots_axisangle_right = quaternion_to_axis_angle(matrix_to_quaternion(rotation_6d_to_matrix(
            mano_rots_6d_right))).reshape(-1, 48)
        mano_rots_axisangle_left = quaternion_to_axis_angle(matrix_to_quaternion(rotation_6d_to_matrix(
            mano_rots_6d_left))).reshape(-1, 48)
        shape = torch.zeros((x.shape[0], 10)).cuda()

        vertices_right, joints_right = self.layer_mano_right(mano_rots_axisangle_right, th_betas=shape,
                                                             th_trans=mano_trans_right)
        vertices_left, joints_left = self.layer_mano_left(mano_rots_axisangle_left, th_betas=shape,
                                                          th_trans=mano_trans_left)

        joints_3d = torch.cat((joints_right, joints_left), 1)
        joints_intrinsic = torch.matmul(joints_3d, torch.transpose(mat_cam, 0, 1))
        joints_2d = joints_intrinsic[..., 0:2] / joints_intrinsic[..., [2]]

        return mano_params, masks, joints_3d, joints_2d

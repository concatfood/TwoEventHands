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
focal = 0.5 * res[1] / math.tan(fovy / 2.0)     # constant as of now
mat_cam = torch.from_numpy(np.array([[focal, 0.0, -res[0] / 2.0],
                                     [0.0, -focal, -res[1] / 2.0],
                                     [0.0, 0.0, -1.0]])).float().cuda()
mat_cam_inv = torch.linalg.inv(mat_cam)


# TwoEventHandsNet
class TEHNet(nn.Module):
    def __init__(self):
        super(TEHNet, self).__init__()

        self.layer_mano_right = ManoLayer(flat_hand_mean=True, side='right', mano_root='mano', use_pca=False,
                                          root_rot_mode='axisang', joint_rot_mode='axisang').cuda()
        self.layer_mano_left = ManoLayer(flat_hand_mean=True, side='left', mano_root='mano', use_pca=False,
                                         root_rot_mode='axisang', joint_rot_mode='axisang').cuda()
        self.resnet = ResNet()
        self.unet = UNet()

        # assume constant roots because of constant shapes
        pose_zero = torch.zeros((1, 48)).cuda()
        shape_zero = torch.zeros((1, 10)).cuda()
        trans_zero = torch.zeros((1, 3)).cuda()

        vertices_right, joints_right = self.layer_mano_right(pose_zero, th_betas=shape_zero, th_trans=trans_zero)
        vertices_left, joints_left = self.layer_mano_left(pose_zero, th_betas=shape_zero, th_trans=trans_zero)
        self.roots = [joints_right[0, 0, ...], joints_left[0, 0, ...]]

    # def forward(self, x):
    def forward(self, x):
        masks = self.unet(x)
        x_masked = torch.cat((x, masks), 1)
        hand_params = self.resnet(x_masked)
        # hand_params = self.resnet(x)

        # quaternion to axis-angle
        hand_rots_6d_right = hand_params[:, 0:96].reshape(-1, 16, 6)
        hand_rots_6d_left = hand_params[:, 99:195].reshape(-1, 16, 6)

        hand_trans_right = torch.matmul(torch.cat([hand_params[:, 96:98], torch.ones(hand_params.shape[0], 1).cuda()],
                                                  1) * focal, torch.transpose(mat_cam_inv, 0, 1))\
                           * hand_params[:, [98]] - torch.unsqueeze(self.roots[0], 0)
        hand_trans_left = torch.matmul(torch.cat([hand_params[:, 195:197], torch.ones(hand_params.shape[0], 1).cuda()],
                                                 1) * focal, torch.transpose(mat_cam_inv, 0, 1))\
                          * hand_params[:, [197]] - torch.unsqueeze(self.roots[1], 0)

        hand_rots_axisangle_right = quaternion_to_axis_angle(matrix_to_quaternion(rotation_6d_to_matrix(
            hand_rots_6d_right))).reshape(-1, 48)
        hand_rots_axisangle_left = quaternion_to_axis_angle(matrix_to_quaternion(rotation_6d_to_matrix(
            hand_rots_6d_left))).reshape(-1, 48)
        shape = torch.zeros((x.shape[0], 10)).cuda()

        vertices_right, joints_right = self.layer_mano_right(hand_rots_axisangle_right, th_betas=shape,
                                                             th_trans=hand_trans_right)
        vertices_left, joints_left = self.layer_mano_left(hand_rots_axisangle_left, th_betas=shape,
                                                          th_trans=hand_trans_left)

        vertices = torch.cat((vertices_right, vertices_left), 1)
        joints_3d = torch.cat((joints_right, joints_left), 1)
        joints_intrinsic = torch.matmul(joints_3d, torch.transpose(mat_cam, 0, 1))
        joints_2d = joints_intrinsic[..., 0:2] / joints_intrinsic[..., [2]]

        return hand_params, vertices, masks, joints_3d, joints_2d
        # return hand_params, vertices, joints_3d, joints_2d

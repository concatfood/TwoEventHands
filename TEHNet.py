from manopth.manolayer import ManoLayer
import math
import numpy as np
from pytorch3d.transforms import quaternion_to_axis_angle
from resnet50 import ResNet
import torch
import torch.nn as nn


# camera parameters
res = (240, 180)
far = 1.0
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
        self.layer_mano_right = ManoLayer(flat_hand_mean=True, side='right', mano_root='mano', use_pca=False,
                                          root_rot_mode='axisang', joint_rot_mode='axisang')
        self.layer_mano_left = ManoLayer(flat_hand_mean=True, side='left', mano_root='mano', use_pca=False,
                                         root_rot_mode='axisang', joint_rot_mode='axisang')

    def forward(self, x):
        mano_params = self.resnet(x)

        # quaternion to axis-angle
        mano_rots_quat_right = mano_params[:, 0:64].reshape(-1, 16, 4)
        mano_rots_quat_left = mano_params[:, 67:131].reshape(-1, 16, 4)
        mano_trans_right = mano_params[:, 64:67]
        mano_trans_left = mano_params[:, 131:134]
        mano_rots_axisangle_right = quaternion_to_axis_angle(mano_rots_quat_right).reshape(-1, 48)
        mano_rots_axisangle_left = quaternion_to_axis_angle(mano_rots_quat_left).reshape(-1, 48)
        mano_params_manolayer = torch.cat([mano_rots_axisangle_right, mano_trans_right,
                                           mano_rots_axisangle_left, mano_trans_left], dim=1)

        vertices_right, joints_right = self.layer_mano_right(mano_params_manolayer[:, :48],
                                                             th_betas=torch.zeros((mano_params_manolayer.shape[0], 10))
                                                             .cuda(),
                                                             th_trans=mano_params_manolayer[:, 48:51])
        vertices_left, joints_left = self.layer_mano_left(mano_params_manolayer[:, 51:99],
                                                          th_betas=torch.zeros((mano_params_manolayer.shape[0], 10))
                                                          .cuda(),
                                                          th_trans=mano_params_manolayer[:, 99:102])

        joints_3d = torch.cat((joints_right, joints_left), 1)
        joints_intrinsic = torch.matmul(joints_3d, torch.transpose(mat_cam, 0, 1))
        joints_2d = joints_intrinsic[..., 0:2] / joints_intrinsic[..., [2]]

        return mano_params, joints_3d, joints_2d

import torch
import torch.nn as nn

from resnet50 import ResNet
from unet import UNet


# TwoEventHandsNet = UNet + ResNet50
class TEHNet(nn.Module):
    def __init__(self):
        super(TEHNet, self).__init__()
        self.unet = UNet()
        self.resnet = ResNet()

    def forward(self, x):
        x1 = self.unet(x)

        # LNES scaled by each mask prediction
        x_masked = torch.cat([x * x1[:, [0], :, :], x * x1[:, [1], :, :], x * x1[:, [2], :, :]], dim=1)

        mano_params = self.resnet(x_masked)

        return x1, mano_params

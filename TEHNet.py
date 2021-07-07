import logging
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
        mask = self.unet(x)

        # LNES and mask
        x_masked = torch.cat([x, mask], dim=1)

        mano_params = self.resnet(x_masked)

        return mask, mano_params

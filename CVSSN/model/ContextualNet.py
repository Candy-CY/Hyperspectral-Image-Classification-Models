# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2022-Nov
# @Address  : Time Lab @ SDU
# @FileName : ContextualNet.py
# @Project  : CVSSN (HSIC), IEEE TCSVT

# source:
# https://github.com/suvojit-0x55aa/A2S2K-ResNet/blob/master/ContextualNet/ContextualNet.py
# https://github.com/eecn/Hyperspectral-Classification/blob/master/models.py

# make revision from CONTEXTUAL according to
# Going Deeper With Contextual CNN for Hyperspectral Image Classification, TIP 2017


import torch
from torch import nn
import torch.nn.functional as F


def conv128(in_planes, out_planes, kernel):
    return nn.Conv2d(in_planes, out_planes, kernel)


class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
    #         init.kaiming_uniform_(m.weight)
    #         init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes, channels_1=128, channels_2=384):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)

        self.conv_5x5 = nn.Conv3d(
            1, 128, (5, 5, in_channels), stride=(1, 1, 3), padding=(2, 2, 0))

        self.conv_3x3 = nn.Conv3d(
            1, 128, (3, 3, in_channels), stride=(1, 1, 2), padding=(1, 1, 0))
        self.conv_1x1 = nn.Conv3d(
            1, 128, (1, 1, in_channels), stride=(1, 1, 1), padding=0)

        self.name = 'LeeEtAl'

        # We use two modules from the residual learning approach
        # Residual block 1
        # self.conv1 = nn.Conv2d(in_channels=channels_2, out_channels=channels_1, kernel_size=1)
        self.conv1 = nn.Conv2d(channels_2, channels_1, kernel_size=1)
        # self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv2 = nn.Conv2d(channels_1, channels_1, kernel_size=1)
        # self.conv3 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(channels_1, channels_1, kernel_size=1)

        # Residual block 2
        # self.conv4 = nn.Conv2d(128, 128, (1, 1))
        # self.conv5 = nn.Conv2d(128, 128, (1, 1))
        self.conv4 = nn.Conv2d(channels_1, channels_1, kernel_size=1)
        self.conv5 = nn.Conv2d(channels_1, channels_1, kernel_size=1)

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        # self.conv6 = nn.Conv2d(128, 128, (1, 1))
        # self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv6 = nn.Conv2d(channels_1, channels_1, kernel_size=1)
        self.conv7 = nn.Conv2d(channels_1, channels_1, kernel_size=1)
        self.conv8 = nn.Conv2d(128, n_classes, (9, 9))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

        # self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_5x5 = self.conv_5x5(x)
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_5x5, x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x, dim=-1)

        # Local Response Normalization
        # x = F.relu(self.lrn1(x))
        x = self.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        # x = F.relu(self.lrn2(x))
        x = self.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        x = x.squeeze(2).squeeze(2)
        return x

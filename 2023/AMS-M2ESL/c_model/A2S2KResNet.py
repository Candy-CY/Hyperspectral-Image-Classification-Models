# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : A2S2KResNet.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

# https://github.com/suvojit-0x55aa/A2S2K-ResNet/blob/master/A2S2KResNet/A2S2KResNet.py
# Attention-Based Adaptive Spectral-Spatial Kernel ResNet for Hyperspectral Image Classification, TGRS 2020

import math
import torch
from torch import nn
import torch.nn.functional as F

PARAM_KERNEL_SIZE = 24


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(
            self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(
            input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor,
                                  squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(
            self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(
            in_channels=num_channels,
            out_channels=num_channels_reduced,
            kernel_size=1,
            stride=1)
        self.conv_cT = nn.Conv3d(
            in_channels=num_channels_reduced,
            out_channels=num_channels,
            kernel_size=1,
            stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([
            squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
            squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
            squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)
        ])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(
            self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w, t = x.size()

        # feature descriptor on the global spatial information
        # 24, 1, 1, 1
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(
            -1, -3).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Residual(nn.Module):  # pytorch
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            use_1x1conv=False,
            stride=1,
            start_block=False,
            end_block=False,
    ):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride), nn.ReLU())
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if start_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        if end_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        # ECA Attention Layer
        self.ecalayer = eca_layer(out_channels)

        # start and end block initialization
        self.start_block = start_block
        self.end_block = end_block

    def forward(self, X):
        identity = X

        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)

        out = self.ecalayer(out)  # EFR

        out += identity

        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out)

        return out


class S3KAIResNet(nn.Module):
    def __init__(self, band, classes, reduction):
        super(S3KAIResNet, self).__init__()
        self.name = 'SSRN'
        self.conv1x1 = nn.Conv3d(
            in_channels=1,
            out_channels=PARAM_KERNEL_SIZE,
            kernel_size=(1, 1, 7),
            stride=(1, 1, 2),
            padding=0)

        self.conv3x3 = nn.Conv3d(
            in_channels=1,
            out_channels=PARAM_KERNEL_SIZE,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 2),
            padding=(1, 1, 0))

        self.batch_norm1x1 = nn.Sequential(
            nn.BatchNorm3d(
                PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))
        self.batch_norm3x3 = nn.Sequential(
            nn.BatchNorm3d(
                PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(
            nn.Conv3d(
                PARAM_KERNEL_SIZE, band // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True))
        self.conv_ex = nn.Conv3d(
            band // reduction, PARAM_KERNEL_SIZE, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.res_net1 = Residual(
            PARAM_KERNEL_SIZE,
            PARAM_KERNEL_SIZE, (1, 1, 7), (0, 0, 3),
            start_block=True)
        self.res_net2 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE,
                                 (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE,
                                 (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(
            PARAM_KERNEL_SIZE,
            PARAM_KERNEL_SIZE, (3, 3, 1), (1, 1, 0),
            end_block=True)

        kernel_3d = math.ceil((band - 6) / 2)  # 97
        # print(kernel_3d)

        self.conv2 = nn.Conv3d(
            in_channels=PARAM_KERNEL_SIZE,
            out_channels=128,
            kernel_size=(1, 1, kernel_3d),
            stride=(1, 1, 1),
            padding=(0, 0, 0))

        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),  # 0.1
            nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(
            in_channels=1,
            out_channels=PARAM_KERNEL_SIZE,
            kernel_size=(3, 3, 128),
            stride=(1, 1, 1),
            padding=(0, 0, 0)
        )
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(
                PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))  # kernel_size=stride
        self.full_connection = nn.Sequential(
            nn.Linear(PARAM_KERNEL_SIZE, classes)
            # nn.Softmax()
        )

    def forward(self, X):
        # X = X.permute(0,1,4,3,2)
        x_1x1 = self.conv1x1(X)  # 16X1X9X9X200-->[16,24,9,9,97]  # Dimensionality Reduction
        x_1x1 = self.batch_norm1x1(x_1x1).unsqueeze(dim=1)  # [16,1,24,9,9,97]
        x_3x3 = self.conv3x3(X)  # [16,24,9,9,97]
        x_3x3 = self.batch_norm3x3(x_3x3).unsqueeze(dim=1)  # [16,1,24,9,9,97]

        x1 = torch.cat([x_3x3, x_1x1], dim=1)  # [16,2,24,9,9,97]
        U = torch.sum(x1, dim=1)  # [2,24,9,9,97] # Dimensionality 1 will be reduced
        S = self.pool(U)  # [16,24,1,1,1]
        Z = self.conv_se(S)  # [16,100,1,1,1]
        attention_vector = torch.cat(
            [
                self.conv_ex(Z).unsqueeze(dim=1),
                self.conv_ex(Z).unsqueeze(dim=1)
            ],
            dim=1)  # [16,2,24,1,1,1]
        attention_vector = self.softmax(attention_vector)  # [16,2,24,1,1,1]
        V = (x1 * attention_vector).sum(dim=1)  # [16,24,9,9,97]

        ########################################
        x2 = self.res_net1(V)  # [16,24,9,9,97]
        x2 = self.res_net2(x2)  # [16,24,9,9,97]
        x2 = self.batch_norm2(self.conv2(x2))  # [16,128,9,9,1]
        x2 = x2.permute(0, 4, 2, 3, 1)  # [16,1,9,9,128]
        x2 = self.batch_norm3(self.conv3(x2))  # [16,24,7,7,1]

        x3 = self.res_net3(x2)  # [16,24,7,7,1]
        x3 = self.res_net4(x3)  # [16,24,7,7,1]
        x4 = self.avg_pooling(x3)  # [16,24,1,1,1]
        x4 = x4.view(x4.size(0), -1)  # [16,24]
        return self.full_connection(x4)

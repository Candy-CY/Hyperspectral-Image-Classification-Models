# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2022-Nov
# @Address  : Time Lab @ SDU
# @FileName : SSAN.py
# @Project  : CVSSN (HSIC), IEEE TCSVT

# code reproduced from tensorflow version
# source: https://github.com/XiangtaoZheng/SSAN
# Spectral–Spatial Attention Network for Hyperspectral Image Classification, TGRS 2021
# the adopted non-local block is from https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/models/non_local.py

import torch
import torch.nn as nn
from torch.nn import functional as F


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class SSAN(nn.Module):
    def __init__(self, band_nums, patch_size, class_count):
        super(SSAN, self).__init__()
        self.band_nums = band_nums

        self.class_count = class_count
        self.in_channels = 1
        self.mid_channels = 32 * self.band_nums
        self.fin_channels = 64 * patch_size * patch_size

        self.relu = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv3d(self.in_channels, 32, kernel_size=(1, 1, 7), stride=(1, 1, 1),
                                  padding=(0, 0, 3))  # keep spatial window ubchange
        self.conv3d_2 = nn.Conv3d(32, 32, kernel_size=(1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3))

        # self.nonlocal_atten_1=NLBlockND(in_channels=self.mid_channels,inter_channels=64,dimension=2,bn_layer=True)
        self.nonlocal_atten_1 = NLBlockND(in_channels=self.mid_channels, dimension=2, bn_layer=True)
        self.nonlocal_atten_2 = NLBlockND(64, dimension=2, bn_layer=True)

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(self.mid_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # self.bn = nn.BatchNorm2d(self.channels)

        self.flatten = nn.Flatten(1)

        self.dropout = nn.Dropout(p=0.5)
        self.fc_1 = nn.Sequential(
            nn.Linear(self.fin_channels, 256),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.fc_0 = nn.Sequential(
            nn.Linear(256, self.class_count),
            nn.BatchNorm1d(self.class_count)
        )

    def forward(self, x):
        batch_size, c, h, w, band = x.size()

        # 3D
        x_3d = self.conv3d_1(x)
        x_3d = self.conv3d_2(x_3d)

        x_2d = x_3d.permute(0, 1, 4, 2, 3).reshape(batch_size, -1, h, w)

        x_2d = self.nonlocal_atten_1(x_2d)
        x_2d = self.conv2d_1(x_2d)

        x_2d = self.nonlocal_atten_2(x_2d)
        x_2d = self.conv2d_2(x_2d)

        x_2d = self.nonlocal_atten_2(x_2d)
        x_f = self.flatten(x_2d)

        x = self.dropout(self.fc_1(x_f))
        out = self.fc_0(x)
        return out

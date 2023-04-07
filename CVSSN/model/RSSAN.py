# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2022-Nov
# @Address  : Time Lab @ SDU
# @FileName : RSSAN.py
# @Project  : CVSSN (HSIC), IEEE TCSVT


# source: https://github.com/lierererniu/RSSAN-Hyperspectral-Image/blob/main/model/network.py
# Residual Spectral–Spatial Attention Network for Hyperspectral Image Classification, TGRS 2020

import torch
from torch import nn


class Spectral_attention(nn.Module):
    #  batchsize 16 25 200
    def __init__(self, in_features, hidden_features, out_features):
        super(Spectral_attention, self).__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.MaxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.SharedMLP = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()  #

    def forward(self, X):
        y1 = self.AvgPool(X)
        y2 = self.MaxPool(X)
        y1 = y1.view(y1.size(0), -1)
        y2 = y2.view(y2.size(0), -1)
        # print(y1.shape, y2.shape)
        y1 = self.SharedMLP(y1)
        y2 = self.SharedMLP(y2)
        y = y1 + y2
        y = torch.reshape(y, (y.shape[0], y.shape[1], 1, 1))
        return self.sigmoid(y)  #


class Spatial_attention(nn.Module):
    # 2, 1, 3, 1, 1
    def __init__(self, in_chanels, kernel_size, out_chanel, stride, padding):
        super(Spatial_attention, self).__init__()
        # self.AvgPool = nn.AdaptiveAvgPool2d((17, 17))  # (N, 200, 17, 17)
        # self.MaxPool = nn.AdaptiveMaxPool2d((17, 17))
        self.conv1 = nn.Conv2d(in_chanels, out_chanel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.Sigmoid()

    def forward(self, X):
        # y1 = self.AvgPool(X)
        # y2 = self.MaxPool(X)
        avg_out = torch.mean(X, dim=1, keepdim=True)
        max_out, _ = torch.max(X, dim=1, keepdim=True)
        y = torch.cat((avg_out, max_out), 1)
        y = self.conv1(y)
        return self.act(y)


class RSSAN(nn.Module):
    def __init__(self, feature_class, in_chanels, kernel_size, out_chanel, stride, padding, windows):
        # 16, 200, 3, 32, 1, 1
        super(RSSAN, self).__init__()
        self.attention1 = Spectral_attention(in_chanels, int(in_chanels // 8), in_chanels)
        self.attention2 = Spatial_attention(2, 3, 1, 1, 1)
        self.conv1 = nn.Conv2d(in_chanels, out_chanel, kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(out_chanel, out_channels=out_chanel, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.bn2 = nn.Sequential(
            nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_chanel, out_chanel, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True)
        self.attention3 = Spectral_attention(out_chanel, out_chanel // 8, out_chanel)
        self.attention4 = Spatial_attention(2, 3, 1, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv4 = nn.Conv2d(out_chanel, out_chanel, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn4 = nn.Sequential(
            nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Conv2d(out_chanel, out_chanel, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn5 = nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True)
        self.attention5 = Spectral_attention(out_chanel, out_chanel // 8, out_chanel)
        self.attention6 = Spatial_attention(2, 3, 1, 1, 1)
        self.relu2 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(1)
        # 1*1
        self.conv6 = nn.Conv2d(out_chanel, out_chanel, kernel_size=(1, 1),
                               stride=stride, padding=0)
        self.full_connection = nn.Sequential(
            nn.Linear(out_chanel * windows * windows, feature_class),
            # nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, X):
        x = X.permute(0, 3, 1, 2)
        x1 = self.attention1(x)
        x3 = x1 * x
        # print(x3.shape)
        x4 = self.attention2(x3) * x3
        x5 = self.conv1(x4)
        x6 = self.bn1(x5)  # #
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.bn3(self.conv3(x8))  # #
        se = self.attention3(x9) * x9
        sa = self.attention4(se) * se
        x10 = self.relu1(sa * x9 + x6)  # #
        x11 = self.conv4(x10)
        x12 = self.bn4(x11)
        x13 = self.bn5(self.conv5(x12))  # #
        se1 = self.attention5(x13) * x13
        sa1 = self.attention6(se1) * se1
        x14 = self.relu2(sa1 * x13 + x10)
        # print(x14.size())
        x15 = self.conv6(self.avgpool(x14))
        x16 = x15.contiguous().view(x15.size(0), -1)
        # print(x16.size())
        y = self.full_connection(x16)
        return y


def RSSAN_net(in_shape, num_classes):
    model = RSSAN(num_classes, in_shape[0], 3, 32, 1, 1, 9)
    return model

# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : MCM_CNN.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS


# unofficial implementation based offical Matlab version
# https://github.com/henanjun/demo_MCMs
# Feature Extraction With Multiscale Covariance Maps for Hyperspectral Image Classification, TGRS, 2018


import torch.nn as nn


class MCM_CNN(nn.Module):
    def __init__(self, scales, class_count, ds):
        super(MCM_CNN, self).__init__()
        self.channels = scales
        self.class_count = class_count
        if ds == 'IP':
            self.channels_fc_1 = 576
            self.channels_fc_2 = 128
        elif ds == 'UP':
            self.channels_fc_1 = 576
            self.channels_fc_2 = 512
        elif ds == 'UH_tif':
            self.channels_fc_1 = 576
            self.channels_fc_2 = 128

        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.AvgPool2d((2, 2), stride=2)

        self.conv_1 = nn.Conv2d(self.channels, 128, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten(1)
        self.fc2 = nn.Linear(self.channels_fc_1, self.channels_fc_2)
        self.fc1 = nn.Linear(self.channels_fc_2, self.channels_fc_2)
        self.fc0 = nn.Linear(self.channels_fc_2, self.class_count)

    def forward(self, x):

        x = self.conv_1(x)
        x = self.relu(self.pooling(x))
        x = self.conv_2(x)
        x = self.relu(self.pooling(x))

        x = self.flatten(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc1(x))
        out = self.fc0(x)
        return out

    def _init_weight(self):
        for m in self.modules():
            nn.init.xavier_normal(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def MCM_CNN_(channels, num_classes, ds):
    model = MCM_CNN(channels, num_classes, ds)
    return model

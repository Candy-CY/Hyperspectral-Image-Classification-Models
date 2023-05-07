# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : SSSAN.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

# unofficial implementation based on part of offical Keras version
# dense net backbone from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/densenet.py
# Spectral–Spatial Self-Attention Networks for Hyperspectral Image Classification, TGRS 2021

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import cosine_similarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class spa_self_atten(nn.Module):
    def __init__(self, in_channels):
        super(spa_self_atten, self).__init__()
        self.to_ab = nn.Conv2d(in_channels, in_channels * 2, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, c, h, w = x.size()

        a, b = self.to_ab(x).chunk(2, dim=1)
        a = a.view(batch_size, -1, h * w).permute(0, 2, 1)
        b = b.view(batch_size, -1, h * w).permute(0, 2, 1)
        cent_spec_vector = a[:, int((h * w - 1) / 2)]
        cent_spec_vector = torch.unsqueeze(cent_spec_vector, 1)

        sim_cosine = cosine_similarity(cent_spec_vector, b, dim=2)  # cos
        sim_cosine_2 = torch.pow(sim_cosine, 2)  # cos^2

        atten_s = self.softmax(sim_cosine_2)
        atten_s = torch.unsqueeze(atten_s, 2)

        out = torch.mul(atten_s, b).contiguous().view(batch_size, -1, h, w) + x

        return out


class spe_self_atten(nn.Module):
    def __init__(self):
        super(spe_self_atten, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, c, l = x.size()
        x = x.to(device)

        sim_cosine_mat = torch.zeros(batch_size, c, c).to(device)
        for i in range(c):
            target_vector = x[:, i]
            target_vector = torch.unsqueeze(target_vector, 1)
            sim_cosine_mat[:, i] = cosine_similarity(target_vector, x, dim=2)
        atten_s = self.softmax(sim_cosine_mat)

        out = torch.bmm(atten_s, x).contiguous() + x

        return out


class transition_2D(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(transition_2D, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=self.conv_init_a, mode=self.conv_init_mode,
                                        nonlinearity='leaky_relu')


class transition_1D(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(transition_1D, self).__init__()
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv1 = nn.Conv1d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool1d(out, 2)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=self.conv_init_a, mode=self.conv_init_mode,
                                        nonlinearity='leaky_relu')


class spe_dense_bottleneck_1D(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(spe_dense_bottleneck_1D, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

        self.spe_atten = spe_self_atten()

    def forward(self, x):
        x = x.to(device)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = self.spe_atten(out)
        out = torch.cat((x, out), 1)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=self.conv_init_a, mode=self.conv_init_mode,
                                        nonlinearity='leaky_relu')


class spa_dense_bottleneck_2D(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(spa_dense_bottleneck_2D, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

        self.spa_artten = spa_self_atten(growthRate)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = self.spa_artten(out)

        out = torch.cat((x, out), 1)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=self.conv_init_a, mode=self.conv_init_mode,
                                        nonlinearity='leaky_relu')


class SpaNet(nn.Module):
    def __init__(self, in_channels, depth=16, dropRate=0, growthRate=22, compressionRate=2):
        super(SpaNet, self).__init__()

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.in_channels = in_channels
        self.inplanes = growthRate * 2

        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=3, padding=1,
                               bias=False)

        self.spa_dense_1 = self._make_denseblock(spa_dense_bottleneck_2D, n)
        self.trans_2d_1 = self._make_transition(compressionRate)
        self.spa_dense_2 = self._make_denseblock(spa_dense_bottleneck_2D, n)
        self.trans_2d_2 = self._make_transition(compressionRate)
        self.spa_dense_3 = self._make_denseblock(spa_dense_bottleneck_2D, n)
        self.trans_2d_3 = self._make_transition(compressionRate)

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(self.inplanes, 32)

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return transition_2D(inplanes, outplanes)

    def forward(self, x_2d):
        x_2d = x_2d.permute(0, 3, 1, 2)

        x = self.conv1(x_2d)

        x = self.trans_2d_1(self.spa_dense_1(x))
        x = self.trans_2d_2(self.spa_dense_2(x))
        x = self.trans_2d_3(self.spa_dense_3(x))

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class SpeNet(nn.Module):
    def __init__(self, depth=16, dropRate=0, growthRate=22, compressionRate=2):
        super(SpeNet, self).__init__()
        self.growthRate = growthRate
        self.dropRate = dropRate

        self.in_channels = 1
        self.inplanes = growthRate * 2

        n = (depth - 4) // 6
        self.conv1 = nn.Conv1d(self.in_channels, self.inplanes, kernel_size=3, padding=1,
                               bias=False)

        self.spe_dense_1 = self._make_denseblock(spe_dense_bottleneck_1D, n)
        self.trans_1d_1 = self._make_transition(compressionRate)
        self.spe_dense_2 = self._make_denseblock(spe_dense_bottleneck_1D, n)
        self.trans_1d_2 = self._make_transition(compressionRate)
        self.spe_dense_3 = self._make_denseblock(spe_dense_bottleneck_1D, n)
        self.trans_1d_3 = self._make_transition(compressionRate)

        self.bn = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(self.inplanes, 32)

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return transition_1D(inplanes, outplanes)

    def forward(self, x_spe):
        x_spe = torch.unsqueeze(x_spe, 1)

        x = self.conv1(x_spe)

        x = self.trans_1d_1(self.spe_dense_1(x))
        x = self.trans_1d_2(self.spe_dense_2(x))
        x = self.trans_1d_3(self.spe_dense_3(x))

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class SSSAN(nn.Module):
    def __init__(self, in_channels, dr_channels, class_count):
        super(SSSAN, self).__init__()
        self.in_channels = in_channels
        self.dr_channels = dr_channels
        self.class_count = class_count

        self.spa_net = SpaNet(self.dr_channels)
        self.spe_net = SpeNet()
        self.lamuda = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)
        self.fc = nn.Linear(32, self.class_count)

    def forward(self, X_spa, X_spe):
        # X_spa=X_spa.type(torch.float)
        # X_spe=X_spe.type(torch.float)
        out_spa = self.spa_net(X_spa)
        out_spe = self.spe_net(X_spe)

        lmd = torch.sigmoid(self.lamuda)
        out = lmd * out_spa + (1 - lmd) * out_spe
        out = self.fc(out)

        return out

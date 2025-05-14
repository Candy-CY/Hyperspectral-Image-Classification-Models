# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2022-Nov
# @Address  : Time Lab @ SDU
# @FileName : CVSSN.py
# @Project  : CVSSN (HSIC), IEEE TCSVT


import torch
import torch.nn as nn
import numpy as np

from torch import cosine_similarity

# import model.self_sim as self_sim

# import random

# import model.self_atten as self_atten
# import model.sim_atten as sim_atten
# import model.Attention as atten


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_ch, out_ch, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv_kxk(in_ch, out_ch, kz, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=kz, stride=stride,
                     padding=kz // 2, groups=groups, bias=False, dilation=dilation)


class spa_ED_Cos_SVSS(nn.Module):
    def __init__(self):
        super(spa_ED_Cos_SVSS, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.lamuda = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)

    def forward(self, x):
        batch_size, c, h, w = x.size()
        q, k = x.view(batch_size, -1, h * w).permute(0, 2, 1), x.view(batch_size, -1, h * w)

        cent_spec_vector = q[:, int((h * w - 1) / 2)]
        cent_spec_vector = torch.unsqueeze(cent_spec_vector, 1)
        csv_expand = cent_spec_vector.expand(batch_size, h * w, c)

        # ED_sim
        E_dist = torch.norm(csv_expand - q, dim=2, p=2)
        sim_E_dist = 1 / (1 + E_dist)

        # Cos_sim
        sim_cos = cosine_similarity(cent_spec_vector, k.permute(0, 2, 1), dim=2)  # include negative

        lmd = torch.sigmoid(self.lamuda)
        atten_ED = self.softmax(sim_E_dist)
        atten_cos = self.softmax(sim_cos)

        # adaptive weight addition
        atten_s = lmd * atten_ED + (1 - lmd) * atten_cos
        atten_s = torch.unsqueeze(atten_s, 2)

        q_attened = torch.mul(atten_s, q)
        out = q_attened.contiguous().view(batch_size, -1, h, w) + x

        return out


class spa_ED_FVSS(nn.Module):
    def __init__(self, in_channels):
        super(spa_ED_FVSS, self).__init__()

        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fmap):
        batch_size, c, h, w = fmap.size()

        q, k, v = self.to_qkv(fmap).view(batch_size, -1, h * w).permute(0, 2, 1).chunk(3, dim=-1)

        cent_spec_vector = q[:, int((h * w - 1) / 2)]
        cent_spec_vector = torch.unsqueeze(cent_spec_vector, 1)
        csv_expand = cent_spec_vector.expand(batch_size, h * w, c)

        # ED_sim
        E_dist = torch.norm(csv_expand - k, dim=2, p=2)
        sim_E_dist = 1 / (1 + E_dist)

        atten_ED = self.softmax(sim_E_dist)
        atten_sim = torch.unsqueeze(atten_ED, 2)
        # view()函数只能用在 contiguous 的variable上, 即占用连续整块内存的变量。
        # 如果在view() 之前用了 transpose, permute 等，需要用 contiguous() 来返回一个 contiguous copy.
        v_attened = torch.mul(atten_sim, v)
        out = v_attened.contiguous().view(batch_size, -1, h, w) + fmap

        return out


class CSS_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CSS_Conv, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_channels,
            # bias=False
        )
        self.leaky = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_channels)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.leaky(out)
        out = self.depth_conv(out)
        out = self.relu(out)

        return out


class SIC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SIC_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.leaky = nn.LeakyReLU(inplace=True)

        self.branch_conv1x1 = nn.Sequential(
            conv1x1(self.in_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.branch_conv3x3 = nn.Sequential(
            conv3x3(self.in_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch_conv1x1(x)
        out2 = self.branch_conv3x3(x)

        out = self.bn(out1 + out2)
        out = self.relu(out)

        return out


class CVSSN_body(nn.Module):
    def __init__(self, in_channels, in_channels_fused, class_count):
        super(CVSSN_body, self).__init__()
        self.class_count = class_count

        self.in_channels = in_channels
        self.in_channels_fused = in_channels_fused
        self.out_channels = 128

        self.relu = nn.ReLU(inplace=True)

        self.AWA_SVSS = spa_ED_Cos_SVSS()
        self.ED_FVSS = spa_ED_FVSS(self.out_channels)

        self.CSS_C_1 = CSS_Conv(self.in_channels_fused, self.out_channels, 1)
        self.CSS_C_2 = CSS_Conv(self.out_channels, self.out_channels, 3)

        self.SIC_C = SIC_Conv(self.out_channels, self.out_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.bn_2d = nn.BatchNorm2d(self.out_channels)

        self.bn_1d = nn.BatchNorm1d(128)
        self.fc = nn.Linear(self.out_channels, self.class_count)

    def forward(self, x_fused):
        x = self.AWA_SVSS(x_fused)

        x = self.CSS_C_1(x)
        x = self.CSS_C_2(x)

        x = self.ED_FVSS(x)

        x = self.bn_2d(x)
        x = self.SIC_C(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.bn_1d(x)
        out = self.fc(x)

        return out


class CVSSN(nn.Module):
    def __init__(self, in_channels, h, w, class_count):
        super(CVSSN, self).__init__()
        self.class_count = class_count

        self.in_channels = in_channels
        self.height_p = h
        self.width_p = w
        self.win_spa_size = self.height_p * self.width_p
        self.in_channels_fused = self.in_channels + int(np.ceil(self.in_channels / self.win_spa_size))

        if self.in_channels % self.win_spa_size != 0:
            pad_len = self.win_spa_size - self.in_channels % self.win_spa_size
        else:
            pad_len = 0
        self.pad_len = pad_len

        self.CVSSN_body = CVSSN_body(self.in_channels, self.in_channels_fused, self.class_count)

    def forward(self, x_spa, x_spe):

        x_spa = x_spa.permute(0, 3, 1, 2)
        batch_size, c, h, w = x_spa.size()

        # SSIF
        x_spe = torch.unsqueeze(torch.unsqueeze(x_spe, 1), 1)#squeeze()函数的功能是维度压缩，unsqueeze()是升维操作.
        pad = torch.nn.ReflectionPad2d(padding=(0, self.pad_len, 0, 0))
        #镜像操作（左右上下的规则）
        x_spe_paded = pad(x_spe)

        x_spe_3D = x_spe_paded.view(batch_size, int(x_spe_paded.shape[-1] / self.win_spa_size), self.height_p,
                                    self.width_p)

        x_fused = torch.cat([x_spa, x_spe_3D], dim=1)

        # CVSSN_body
        out = self.CVSSN_body(x_fused)

        return out


def CVSSN_(in_channels, h, w, num_classes):
    model = CVSSN(in_channels, h, w, num_classes)
    return model

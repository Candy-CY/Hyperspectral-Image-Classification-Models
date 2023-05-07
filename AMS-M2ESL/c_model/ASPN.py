# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : ASPN.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

# unofficial implementation based on offical Keras version
# https://github.com/mengxue-rs/a-spn
# Attention-Based Second-Order Pooling Network for Hyperspectral Image Classification, TGRS 2021


import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ASOP(nn.Module):
    def __init__(self, bs, hw):
        super(ASOP, self).__init__()
        self.bs = bs
        self.hw = hw

        self.kernel = nn.Parameter(torch.ones(self.bs, self.hw, 1, device=device), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.bs, self.hw, device=device), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, c, hw = x.size()

        Xmm = self._second_order_pooling(x)

        norm = torch.norm(Xmm, p=2, dim=-1, keepdim=True)
        out = Xmm.div(norm)

        central_vector = out[:, int((hw - 1) / 2)]
        central_vector = torch.unsqueeze(central_vector, dim=1)

        cos = torch.mul(central_vector, self.kernel)

        out = torch.bmm(out, cos) + torch.unsqueeze(self.bias, dim=-1)
        att = self.softmax(out)
        out = torch.bmm(x, att)

        return out

    def _second_order_pooling(self, x):
        x1 = x.permute(0, 2, 1)
        out = torch.bmm(x1, x)

        return out


class A_SPN(nn.Module):
    def __init__(self, bs, height, weight, in_channels, class_count):
        super(A_SPN, self).__init__()
        self.bs = bs
        self.h = height
        self.w = weight
        self.in_channels = in_channels
        self.class_count = class_count

        self.bn = nn.BatchNorm2d(self.in_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.ASOP = ASOP(self.bs, self.h * self.w)
        self.flatten = nn.Flatten(1)

        self.fc = nn.Linear(self.in_channels * self.in_channels, class_count)

    def forward(self, x):
        b, h, w, c = x.size()
        x = x.permute(0, 3, 1, 2)

        x = self.bn(x)
        x = x.view(b, -1, h * w)

        out = self.dropout(x)

        norm = torch.norm(out, p=2, dim=-1, keepdim=True)
        out = out.div(norm)

        out = self.ASOP(out)
        out = self._second_order_pooling(out)

        norm = torch.norm(out, p=2, dim=-1, keepdim=True)
        out = out.div(norm)

        norm = torch.norm(out, p='fro', dim=-1, keepdim=True)
        out = out.div(norm)

        out = self.flatten(out)

        out = self.fc(out)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0, std=1e-4)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _second_order_pooling(self, x):
        x1 = x.permute(0, 2, 1)
        out = torch.bmm(x, x1)

        return out


def ASPN_(bs, height, weight, in_channels, num_classes):
    model = A_SPN(bs, height, weight, in_channels, num_classes)
    return model

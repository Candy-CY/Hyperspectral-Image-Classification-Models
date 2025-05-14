#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 03:22:40 2023

@author: Rojan Basnet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# Mapping Layer
class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


# 3D Convolution
def create_3d_conv_layer(in_channel, out_channel, groups):
    # Sequential convolutional layers with batch normalization
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False, groups=groups),
        nn.BatchNorm3d(out_channel)
    )
    return layer


# Attention Module
class AttentionModule3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AttentionModule3D, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        out = self.squeeze(x).view(b, c)
        out = self.excitation(out).view(b, c, 1, 1, 1)
        return x * out

# Residual Block with Attention
class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, groups):
        super(residual_block, self).__init__()
        self.conv1 = create_3d_conv_layer(in_channel, out_channel, groups)
        self.conv2 = create_3d_conv_layer(out_channel, out_channel, groups)
        self.conv3 = create_3d_conv_layer(out_channel, out_channel, groups)
        self.attention = AttentionModule3D(out_channel)  # Add attention mechanism here.
        self.shortcut = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False, groups=1) if in_channel != out_channel else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)
        x3 = self.attention(x3)  # Apply attention mechanism here.
        out = F.relu(shortcut + x3, inplace=True)
        return out

class ACDFSL(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, groups1, groups2):
        super(ACDFSL, self).__init__()

        self.block1 = residual_block(in_channel, out_channel1, groups1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_block(out_channel1, out_channel2, groups2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.block3 = residual_block(out_channel2, out_channel2, groups2)  # Add the third residual block
        self.block4 = residual_block(out_channel2, out_channel2, groups2)  # Add the fourth residual block
        self.conv = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False, groups=2)

    def forward(self, x):  # x:(400,100,9,9)
        x = x.unsqueeze(1)  # (400,1,100,9,9)
        x = self.block1(x)  # (1,8,100,9,9)
        x = self.maxpool1(x)  # (1,8,25,5,5)
        x = self.block2(x)  # (1,16,25,5,5)
        x = self.maxpool2(x)  # (1,16,7,3,3)
        x = self.block3(x)  # (1,16,7,3,3)
        x = self.block4(x)  # (1,16,7,3,3)
        x = self.conv(x)  # (1,32,5,1,1)
        x = x.view(x.shape[0], -1)  # (1,160)
        return x
    
    
class Network(nn.Module):
    def __init__(self, FEATURE_DIM, CLASS_NUM, TAR_INPUT_DIMENSION, SRC_INPUT_DIMENSION, N_DIMENSION, groups1, groups2):
        super(Network, self).__init__()
        self.feature_encoder = ACDFSL(1, 8, 16, groups1, groups2)
        self.final_feat_dim = FEATURE_DIM  # 64+32
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x, domain='source'):  # x
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)
        feature = self.feature_encoder(x)  # (45, 64)
        output = self.classifier(feature)
        return feature, output
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()

def pairwise_euclidean_distance(a, b):
    """
    Computes the Euclidean distance between two tensors.

    Args:
    - a: Tensor of shape (n, d) representing the first set of embeddings.
    - b: Tensor of shape (m, d) representing the second set of embeddings.

    Returns:
    - distances: Tensor of shape (n, m) representing the pairwise Euclidean distances.
    """
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    distances = -((a - b)**2).sum(dim=2)
    return distances


def gradient_reversal_coefficient(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    """
    Calculates the coefficient for gradient reversal layer.

    Args:
    - iter_num: Current iteration number.
    - high: Upper bound of the coefficient.
    - low: Lower bound of the coefficient.
    - alpha: Scaling factor for the iteration number.
    - max_iter: Maximum number of iterations.

    Returns:
    - coeff: Coefficient for gradient reversal.
    """
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def reverse_gradients(coeff):
    """
    Hook for the gradient reversal layer.

    Args:
    - coeff: Coefficient for gradient reversal.

    Returns:
    - fun1: Function for reversing gradients.
    """
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.domain = nn.Linear(1024, 1)

    def forward(self, x, iter_num):
        coeff = gradient_reversal_coefficient(iter_num, 1.0, 0.0, 10, 10000.0)
        x.register_hook(reverse_gradients(coeff))
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
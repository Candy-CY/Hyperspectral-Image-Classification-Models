import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import math
import scipy as sp
import numpy as np
import scipy.stats
import random
from collections import Counter
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib
matplotlib.use('AGG')
import spectral
import heapq
from conv import DOConv2d, DOConv2d_multi
from torch.utils.data.sampler import WeightedRandomSampler


N_DIMENSION = 100
FEATURE_DIM = 160
CLASS_NUM = 16
SRC_INPUT_DIMENSION = 128
TAR_INPUT_DIMENSION = 200

# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): 
        x1 = F.relu(self.conv1(x), inplace=True) 
        x2 = F.relu(self.conv2(x1), inplace=True) 
        x3 = self.conv3(x2) 
        out = F.relu(x1+x3, inplace=True)
        return out


class multi_scale(nn.Module):
    def __init__(self,in_channel, out_channel, middle_channel):
        super(multi_scale, self).__init__()
        
        self.conv1x1 = DOConv2d(in_channels = in_channel, out_channels=out_channel, kernel_size=1, bias=False)
        self.conv1x3 = DOConv2d_multi(in_channels = out_channel, out_channels=middle_channel, kernel_size=(1,3), padding=(0,1), bias=False)
        self.conv3x1 = DOConv2d_multi(in_channels = out_channel, out_channels=middle_channel, kernel_size=(3,1), padding=(1,0), bias=False)
        self.conv3x3 = DOConv2d(in_channels = out_channel, out_channels=out_channel, kernel_size=3, bias=False)
        self.conv3_1x3 = DOConv2d_multi(in_channels = out_channel, out_channels=middle_channel, kernel_size=(1,3), padding=(0,1), bias=False)
        self.conv3_3x1 = DOConv2d_multi(in_channels = out_channel, out_channels=middle_channel, kernel_size=(3,1), padding=(1,0), bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = DOConv2d(in_channels = out_channel, out_channels=out_channel, kernel_size=1, bias=False)
        
    def forward(self,X):
        X_1 = self.conv1x1(X)
        X_1_out = self.pool(X_1)
        X_2_1 = self.conv1x3(X_1)
        X_2_2 = self.conv3x1(X_1)
        X_2 = torch.cat((X_2_1,X_2_2),1)
        X_2_out = self.pool(X_2)
        X_3_1 = self.conv3x3(X_1)        
        X_3_1_1 = self.conv3_1x3(X_3_1)
        X_3_1_2 = self.conv3_3x1(X_3_1)
        X_3 = torch.cat((X_3_1_1,X_3_1_2),1)
        X_3_out = self.pool(X_3)
        X_4 = self.pool_conv(X_1)
        X_4_out = self.pool(X_4)
        
        out = torch.cat((X_1_out,X_2_out,X_3_out,X_4_out),1)
        
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()
        
        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,1,1),padding=(0,0,0),stride=(4,1,1))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=(3,1,1), bias=False)
        self.final_feat_dim = 160

    def forward(self, x): 
        x_0 = x.unsqueeze(1)
        x_1 = self.block1(x_0)
        x_2 = self.maxpool1(x_1)
        x_4 = self.block2(x_2)
        x_5 = self.maxpool2(x_4)
        x_6 = self.conv(x_5)
        return x_6

class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio = 4):
        super(ChannelSELayer,self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,input):
        N, C, H, W = input.shape
        x_1 = self.avg_pool(input)
        x = x_1.view(-1,C)
        fc_out_1 = self.relu(self.fc1(x))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        output = torch.mul(input,fc_out_2.view(N,C,1,1))
        return output

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.multi = multi_scale(64, 32, 16)
        
        self.feature_encoder = D_Res_3d_CNN(1,8,16)  
             
        self.target_mapping = Mapping(N_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.source_channel = ChannelSELayer(N_DIMENSION)
        self.target_channel = ChannelSELayer(N_DIMENSION)
        
        self.spe_conv1 = nn.Conv2d(160, 128, 3, padding=1, bias=False)
        self.spe_bn1 = nn.BatchNorm2d(128)
        self.spe_conv2 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.spe_bn2 = nn.BatchNorm2d(64)

        self.classifier = nn.Linear(in_features=128, out_features=CLASS_NUM)
        
    def forward(self, x, domain='source'): 
        if domain == 'target':
            x = self.target_mapping(x) 
            x = self.target_channel(x)
        elif domain == 'source':
            x = self.source_mapping(x)
            x = self.source_channel(x)  
        
        feature = self.feature_encoder(x)
        feature_fusion = feature.reshape(feature.shape[0], feature.shape[1] * feature.shape[2], feature.shape[3], feature.shape[4])

        feature = self.spe_conv1(feature_fusion)
        feature = self.spe_bn1(feature)
        feature = F.relu(feature)
        feature = self.spe_conv2(feature)
        feature = self.spe_bn2(feature)
        feature = F.relu(feature)
        feature = self.multi(feature)
        
        feature = feature.view(-1,feature.shape[1])
        output = self.classifier(feature)
        
        return feature_fusion, feature, output
 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
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
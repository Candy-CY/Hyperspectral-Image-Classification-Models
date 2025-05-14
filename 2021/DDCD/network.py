import torch
from torch import nn
from Linear_Attention_Mechanism import PositionLinearAttention,ChannelLinearAttention
from attontion import PAM_Module,CAM_Module
import math
import numpy as np
import torch.nn.functional as F

class DDCD_LAM(nn.Module):
    def __init__(self, band, classes):
        super(DDCD_LAM, self).__init__()
        # spectral branch
        self.name = 'DDCD_LAM'
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                               kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )
        self.conv2a = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=48, padding=0,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=12, padding=1,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), groups=3),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True)
        )
        self.conv2c = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=48, padding=0,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )
        self.conv2d = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=12, padding=1,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), groups=3),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True)
        )
        self.conv2e = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=1,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), groups=3),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.ceil((band - 6) / 2)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=72, out_channels=60, padding=(0, 0, 0),
                      kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1), groups=3),
            nn.BatchNorm3d(60),
            nn.ReLU(inplace=True)
        )
        self.attention_spectral = ChannelLinearAttention()
        self.attention_spatial = PositionLinearAttention(24)
        self.global_pooling_spectral = nn.AdaptiveAvgPool2d(1)
        self.global_pooling_spatial = nn.AdaptiveAvgPool2d(1)
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(60, classes)
        )
    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.batch_norm1(x1)
        # Linear_Attention_Mechanism
        x2p = self.attention_spatial(x2)
        x2c = self.attention_spectral(x2)
        x2a = torch.cat((x2p, x2c), dim=1)
        # bottom channel
        x2l = self.conv2a(x2a)
        x2l = self.conv2b(x2l)
        # top channel
        x2r = self.conv2c(x2a)
        x2r = self.conv2d(x2r)
        x2r = self.conv2e(x2r)
        x10 = torch.cat((x2a, x2l, x2r), dim=1)
        x10 = self.conv3(x10)
        x10 = x10.squeeze(-1)
        x10 = self.global_pooling_spatial(x10)
        x10 = x10.view(x10.size(0), -1)
        output = self.full_connection(x10)
        return output


if __name__ == "__main__":
    x = torch.rand((10, 1, 9, 9, 200), dtype=torch.float)
    DBDA_LAM = DBDA_network_MISH_LAM(200,7)

    print(DBDA_LAM(x).shape)

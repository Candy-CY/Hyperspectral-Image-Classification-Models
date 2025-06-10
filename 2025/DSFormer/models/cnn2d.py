import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init


class CNN2D(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels,256,3)
        self.conv_2 = nn.Conv2d(256,512,3)
        self.fc_1 = nn.Linear(512,128)
        self.fc_2 = nn.Linear(128,n_classes)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.conv_1(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.mp(x)
        x = x.view(-1,x.shape[1])
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        return x

def cnn2d(dataset):
    model = None
    if dataset == 'sa':
        model = CNN2D(in_channels=204,n_classes=16)
    elif dataset == 'ip':
        model = CNN2D(in_channels=200, n_classes=16)
    elif dataset == 'whuhh':
        model = CNN2D(in_channels=270, n_classes=22)
    elif dataset == 'pu':
        model = CNN2D(in_channels=103, n_classes=9)
    elif dataset == 'houston13':
        model = CNN2D(in_channels=144, n_classes=15)
    return model

from thop import profile

if __name__ == '__main__':
    t = torch.randn(size=(1, 1, 103, 13, 13))
    print("input shape:", t.shape)
    net = cnn2d(dataset='pu')
    print("output shape:", net(t).shape)
    flops, params = profile(net, inputs=(t,))
    print('params', params)
    print('flops', flops)  ## 打印计算量
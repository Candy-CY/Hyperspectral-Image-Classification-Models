import torch
import torch.nn as nn
from torch.nn import init

def Init_Weights(net, init_type, gain):
    print('Init Network Weights')
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class MUNet(nn.Module):
    def __init__(self, band, num_classes, ldr_dim, reduction):
        super(MUNet, self).__init__()
        self.fc_hsi = nn.Sequential(
            nn.Conv2d(band, band//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//2),
            nn.ReLU(),
            nn.Conv2d(band//2, band//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//4),
            nn.ReLU(),
            nn.Conv2d(band//4, num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.softmax = nn.Softmax(dim=1)
        self.spectral_fe = nn.Sequential(# SE Attention 模块不应该是 Linear 函数？？？
            nn.Conv2d(ldr_dim, num_classes//reduction, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes//reduction),
            nn.ReLU(),
            nn.Conv2d(num_classes//reduction, num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )
        self.spectral_se = nn.Sequential(
            nn.Conv2d(num_classes, num_classes//reduction, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(num_classes//reduction, num_classes, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_classes, band, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
    def forward(self, x, y):
        encode = self.fc_hsi(x)
        ## spectral attention
        y_fe = self.spectral_fe(y)

        attention = self.spectral_se(y_fe)
        abu = self.softmax(torch.mul(encode, attention))

        output = self.decoder(abu)

        return abu, output

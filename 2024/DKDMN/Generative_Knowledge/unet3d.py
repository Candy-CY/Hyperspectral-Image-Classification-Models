import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
import math
import numpy as np
from thop import profile
from thop import clever_format

class Block3d(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, kernal = (4, 3, 3), stride = (2, 1, 1), padding = (1, 1, 1), up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv3d(2*in_ch, out_ch, 3, padding = 1)
            self.transform = nn.ConvTranspose3d(out_ch, out_ch, kernal, stride, padding)
        else:
            self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding = 1)
            self.transform = nn.Conv3d(out_ch, out_ch, kernal, stride, padding)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding = 1)
        self.bnorm1 = nn.BatchNorm3d(out_ch)
        self.bnorm2 = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU()
        
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 3]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)
        

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device = device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, _image_channels):
        super().__init__()
        image_channels = _image_channels 
        down_channels = (16, 32, 64, 128)
        down_params = [
            [(4, 5, 5), (2, 1, 1), (1, 2, 2)],
            [(4, 5, 5), (2, 1, 1), (1, 2, 2)],
            [(4, 5, 5), (2, 1, 1), (1, 2, 2)],
        ]
        up_channels = (128, 64, 32, 16)
        up_params = [
            [(4, 5, 5), (2, 1, 1), (1, 2, 2)],
            [(4, 5, 5), (2, 1, 1), (1, 2, 2)],
            [(4, 5, 5), (2, 1, 1), (1, 2, 2)],
        ]
        out_dim = 1 
        time_emb_dim = 32
        self.features = []

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv3d(image_channels, down_channels[0], 3, padding = 1)

        # Downsample
        self.downs = nn.ModuleList([Block3d(down_channels[i], down_channels[i + 1], time_emb_dim, \
                                     down_params[i][0], down_params[i][1], down_params[i][2]) \
                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block3d(up_channels[i], up_channels[i + 1], time_emb_dim, \
                                     up_params[i][0], up_params[i][1], up_params[i][2], up=True) \
                    for i in range(len(up_channels) - 1)])

        self.output = nn.Conv3d(up_channels[-1], image_channels, out_dim)

    def forward(self, x, timestep, feature=False):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        print(x.shape)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            # print(x.shape)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # print("down=",residual_x.shape, "up=", x.shape)
            # Add residual x as additional )
            x = torch.cat((x, residual_x), dim=1) 
            if feature:
                self.features.append(x.detach().cpu().numpy())
            x = up(x, t)
        return self.output(x)

    def return_features(self):
        temp_features = []
        temp_features = self.features[:]
        self.features = []
        return temp_features
            

if __name__ == "__main__":
    model = SimpleUnet(_image_channels = 1)
    t = torch.full((1, ), 64, dtype = torch.long)
    a = torch.randn((64, 1, 200, 16, 16))
    # print(model(a, t))
    macs, params = profile(model, inputs=(a, t))
    macs, params = clever_format([macs, params], "%.4f")
    print('FLOPs=', macs)
    print('Params=', params)

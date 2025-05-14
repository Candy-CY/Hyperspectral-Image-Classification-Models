# -*- coding: utf-8 -*-
# Torch
import math
import torch
from torch import nn
import torch.nn as nn
from CNN_ViT import trans_ctf
import torch.nn.functional as F
from ChebConv import _ResChebGC
from einops.layers.torch import Rearrange
from embeddings import PatchEmbeddings, PositionalEmbeddings


"""
AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov et al.
ICLR 2021
ViT
"""
class Pooling(nn.Module):

    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")
        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


"""
@article{ref-sprn,
title={Spectral partitioning residual network with spatial attention mechanism for hyperspectral image classification},
author={Zhang, Xiangrong and Shang, Shouwang and Tang, Xu and Feng, Jie and Jiao, Licheng},
journal={IEEE Trans. Geosci. Remote Sens.},
volume={60},
pages={1--14},
year={2021},
publisher={IEEE}
}
"""
class Res2(nn.Module):  

    def __init__(self, in_channels, inter_channels, kernel_size, padding=0):
        super(Res2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        return X


class Res(nn.Module):  
    def __init__(self, in_channels, kernel_size, padding, groups_s):
        super(Res, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s//2)
        self.bn1_1 = nn.BatchNorm2d(in_channels)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s//2)
        self.bn1_2 = nn.BatchNorm2d(in_channels)

        self.conv2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s)
        self.bn2_1 = nn.BatchNorm2d(in_channels)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s)
        self.bn2_2 = nn.BatchNorm2d(in_channels)

        self.conv3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s*2)
        self.bn3_1 = nn.BatchNorm2d(in_channels)
        self.conv3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s*2)
        self.bn3_2 = nn.BatchNorm2d(in_channels)

        self.res2 = Res2(in_channels, 32, kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        Y1 = F.relu(self.bn1_1(self.conv1_1(X)))
        Y1 = self.bn1_2(self.conv1_2(Y1))

        Y2 = F.relu(self.bn2_1(self.conv2_1(X)))
        Y2 = self.bn2_2(self.conv2_2(Y2))

        Y3 = F.relu(self.bn3_1(self.conv3_1(X)))
        Y3 = self.bn3_2(self.conv3_2(Y3))

        Z = self.res2(X)
        return F.relu(X + Y1 + Y2 + Y3 + Z)


class DiffViT_CLS(nn.Module):
    
    def __init__(self, channels, num_classes, image_size, datasetname, 
                 head_dim: int = 32, hidden_dim: int = 32, emb_dim: int = 512, 
                 patch_size: int = 1, num_layers: int = 1, num_heads: int = 4, pool: str = "mean"):
        super().__init__()

        self.datasetname = datasetname
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        patch_dim = channels * patch_size ** 2
        self.act = nn.ReLU(inplace=True)

        if datasetname == 'IndianPines':
            groups = 16
            groups_width = 32
        elif datasetname == 'PaviaU':
            groups = 5
            groups_width = 64
        elif datasetname == 'Salinas':
            groups = 5
            groups_width = 64
        elif datasetname == 'Houston':
            groups = 5
            groups_width = 64
        else:
            groups = 11
            groups_width = 37
            
        new_bands = math.ceil(channels/groups) * groups
        patch_dim = (groups*groups_width) * patch_size ** 2
        pad_size = new_bands - channels
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, pad_size))
        self.conv_1 = nn.Conv2d(new_bands, groups*groups_width, (1, 1), groups=groups)
        self.bn_1 = nn.BatchNorm2d(groups*groups_width)
        self.res0 = Res(groups*groups_width, (1, 1), (0, 0), groups_s=groups)
        if self.datasetname == 'Houston':
            self.res1 = Res(groups*groups_width, (1, 1), (0, 0), groups_s=groups)

        self.patch_embeddings = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, emb_dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=emb_dim)
        self.transformer = trans_ctf(dim=emb_dim, num_layers=num_layers, num_heads=num_heads, 
                                        head_dim=head_dim, hidden_dim=hidden_dim, num_patch=self.num_patch, patch_size=patch_size)

        self.graconv = _ResChebGC(input_dim=emb_dim, output_dim=emb_dim, hid_dim=hidden_dim, n_seq=self.num_patches, p_dropout=0)
        self.graconv1 = _ResChebGC(input_dim=hidden_dim, output_dim=emb_dim, hid_dim=emb_dim, n_seq=self.num_patches, p_dropout=0)
        self.patchify = Rearrange("b d c  -> b c d")
        self.bnc = nn.BatchNorm1d(emb_dim)
        self.convc = nn.Conv1d(emb_dim, emb_dim, kernel_size=1, bias=True)

        self.dropout = nn.Dropout(0.8)

        self.pool = Pooling(pool=pool)
        self.classifier = Classifier(dim=emb_dim, num_classes=num_classes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x).squeeze(axis=1)
        b, c, h, w = x.shape
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = self.res0(x)
        if self.datasetname == 'Houston':
            x = self.res1(x)

        x = self.patch_embeddings(x)
        x3 = self.pos_embeddings(x)
        x_t = self.transformer(x3)

        x_g = self.graconv1(self.graconv(x))
        x_c = self.patchify(self.convc(self.act(self.bnc(self.patchify(x_g))))) + x

        x = x_t + x_c

        x_fea = self.pool(self.dropout(x))
        x_cls = self.classifier(x_fea)
        return x_fea, x_cls


class MultiViewNet(nn.Module):
    # channels, num_classes, image_size
    def __init__(self, channels, num_classes, image_size, datasetname, num_views):
        super(MultiViewNet, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        if datasetname == 'IndianPines':
            head_dim = 128
            hidden_dim = 128
            emb_dim = 256
        elif datasetname == 'PaviaU':
            head_dim = 32
            hidden_dim = 32
            emb_dim = 256
        elif datasetname == 'Salinas':
            head_dim = 32
            hidden_dim = 32
            emb_dim = 256
        elif datasetname == 'Houston':
            head_dim = 16
            hidden_dim = 16
            emb_dim = 32
        else:
            head_dim = 128
            hidden_dim = 128
            emb_dim = 256
        self.view_net1 = DiffViT_CLS(channels // num_views, num_classes, image_size, datasetname, head_dim, hidden_dim, emb_dim)
        self.view_net2 = DiffViT_CLS(channels // num_views, num_classes, image_size, datasetname, head_dim, hidden_dim, emb_dim)
        # self.fc = nn.Linear(num_views, num_classes)
    
    def view_gen(self, x):
        b, s, c, h, w = x.shape
        view1_x = x[:, :, :c // self.num_views, :, :]
        view2_x = x[:, :, c // self.num_views:, :, :]
        return view1_x, view2_x

    def forward(self, x):
        view_outputs = []
        view1_x, view2_x = self.view_gen(x)
        view1_fea, view1_cls = self.view_net1(view1_x)
        view2_fea, view2_cls = self.view_net2(view2_x)
        view_fea_all = torch.cat([view1_fea, view2_fea], dim=0)
        view_cls_all = torch.cat([view1_cls, view2_cls], dim=0)
        # output = torch.mean(view_outputs, dim = 1)
        return view_fea_all, view_cls_all


# Read the Readme file for more details
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randn(64, 1, 200, 11, 11).to(device)
model = MultiViewNet(channels=200, num_classes=16, image_size=11, datasetname='IndianPines', num_views=2).to(device)
out_fea, out_cls = model(input)
print(model)
print(out_fea.shape)
print(out_cls.shape)
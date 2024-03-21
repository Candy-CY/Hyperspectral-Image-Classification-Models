import einops
import torch
import math
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from einops.layers.torch import Rearrange


class PatchEmbeddings(nn.Module):

    def __init__(self, patch_size: int, patch_dim: int, emb_dim: int):
        super().__init__()
        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            p1=patch_size, p2=patch_size)

        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(in_features=patch_dim, out_features=emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x


class PatchEmbeddingsEfficient(nn.Module):

    def __init__(self, patch_size: int, emb_dim: int, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)
        self.patchify = Rearrange("b e h w -> b (h w) e")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.patchify(x)
        return x


class CLSToken(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class PositionalEmbeddings(nn.Module):

    def __init__(self, num_pos: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_pos, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos
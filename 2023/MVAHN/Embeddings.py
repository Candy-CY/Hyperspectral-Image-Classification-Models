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


class PositionalEmbeddings(nn.Module):

    def __init__(self, num_pos: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_pos, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos
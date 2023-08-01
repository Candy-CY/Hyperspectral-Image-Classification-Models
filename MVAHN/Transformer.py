import einops
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch.fft


class Att_MVAHN(nn.Module):

    def __init__(self, dim: int, head_dim: int, num_heads: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads

        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)

        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)
        self.qkvc = nn.Conv2d(dim, self.inner_dim * 4, kernel_size=1, padding=0, groups=dim, bias=False)
        self.bnc = nn.BatchNorm2d(self.inner_dim)
        self.local = nn.Conv2d(self.inner_dim, self.inner_dim, kernel_size=3, padding=1, groups=self.inner_dim, bias=False)
        self.avgpool=nn.AdaptiveAvgPool1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        # x-->[b, num_seq=num_patch*num_patch, dim]

        x = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        # x-->[b, dim, num_patch, num_patch]

        qkvc = self.qkvc(self.act(self.bn(x)))
        # qkvc-->[b, inner_dim*4, num_patch, num_patch]

        qkvc = qkvc.contiguous().view(b, self.num_patch*self.num_patch, self.inner_dim * 4)
        # qkvc-->[b, num_patch*num_patch, inner_dim*4]

        qkvc = qkvc.chunk(4, dim=-1)
        # qkvc-->q, k, v, c-->[b, num_seq, inner_dim=head_dim * num_heads]

        q, k, v, c = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkvc)
        # q, k, v, c-->[b, num_heads, num_seq, head_dim]

        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        # Dot product of q and k

        scores = scores * self.scale
        # Scale scores [b, num_heads, num_seq, num_seq] (similarity matrix)

        attn = self.attn(scores)
        # Normalize scores to pdist [b, num_heads, num_seq, num_seq]
        
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        # Apply attention to values [b, num_heads, num_seq, num_seq]

        out = einops.rearrange(out, "b h n d -> b n (h d)")
        # Reshape to [b, num_seq, num_heads*head_dim]
        
        c = self.local(self.act(self.bnc(c.reshape(b, self.inner_dim, self.num_patch, self.num_patch)))).reshape(b, n, -1)
        # c-->[b, num_seq, inner_dim]

        out = self.avgpool(out + c)
        # out-->[b, num_seq, dim]

        return out


class ConvM_MVAHN(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(dim), nn.GELU(), 
            nn.Conv2d(dim, 64, kernel_size=1, padding=0, bias=False))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64), nn.GELU(), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False))
        
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64), nn.GELU(), 
            nn.Conv2d(64, dim, kernel_size=1, padding=0, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, hw, dim = x.shape     # [bs, num_seq, dim]
        x_reshape = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        out1 = self.conv1(x_reshape)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2) + x_reshape
        result = out3.contiguous().view(b, self.num_patch * self.num_patch, self.dim)

        return result


class Encoder_MVAHN(nn.Module):

    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), Att_MVAHN(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), ConvM_MVAHN(dim, hidden_dim, num_patch, patch_size))
            ]
            self.layers.append(nn.ModuleList(layer))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
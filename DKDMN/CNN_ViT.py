import einops
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class mish_act(nn.Module):
    def __init__(self, inplace=False):
        super(mish_act, self).__init__()
        self.inplace = inplace
    def forward(self, input):
        return input * torch.tanh(F.softplus(input))


class trans_layer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(trans_layer, self).__init__()
        # self.act = mish_act()
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(self.act(self.bn(x)))
        return out


class Convblock(nn.Module):
    def __init__(self, in_planes, mid_planes):
        super(Convblock, self).__init__()
        # self.act = mish_act()
        self.act = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        mid_planes = in_planes + 2*mid_planes

        self.bn2_1 = nn.BatchNorm2d(mid_planes)
        self.conv2_1 = nn.Conv2d(mid_planes, mid_planes, kernel_size=(3, 3), padding=(1, 1), groups=mid_planes, bias=False)
        self.bn2_2 = nn.BatchNorm2d(mid_planes)
        self.conv2_2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=(3, 3), padding=(1, 1), groups=mid_planes, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, in_planes, kernel_size=1, bias=False)

    def forward(self, x):

        output1 = self.conv1(self.act(self.bn1(x)))
        output11 = torch.cat([output1, x, output1], 1)
        output2_1 = self.conv2_1(self.act(self.bn2_1(output11)))
        output2_2 = self.conv2_2(self.act(self.bn2_2(output2_1)))
        output3 = self.conv3(self.act(self.bn3(output2_2)))
        output = output3 + x
        return output


class Attention_ctf(nn.Module):

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
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(dim)
        self.qkvc = nn.Conv2d(dim, self.inner_dim * 4, kernel_size=1, padding=0, groups=dim, bias=False)

        self.bnc = nn.BatchNorm2d(self.inner_dim//2)
        self.local = nn.Conv2d(self.inner_dim//2, self.inner_dim//2, kernel_size=3, padding=1, groups=self.inner_dim//2, bias=False)
        
        self.bnc1 = nn.BatchNorm2d(self.inner_dim//2)
        self.local1 = nn.Conv2d(self.inner_dim//2, self.inner_dim//2, kernel_size=3, padding=1, groups=self.inner_dim//2, bias=False)

        self.bn2_2 = nn.BatchNorm2d(self.inner_dim//2)
        self.conv2_2 = nn.Conv2d(self.inner_dim//2, self.inner_dim//2, kernel_size=num_patch, padding=0, groups=self.inner_dim//2)
        
        self.bn2_3 = nn.BatchNorm2d(self.inner_dim//2)
        self.avg2_3 = nn.AdaptiveAvgPool2d(1)

        self.avgpool=nn.AdaptiveAvgPool1d(dim)

        self.bncs1 = nn.BatchNorm2d(self.inner_dim//2)
        self.cs1 = nn.Conv2d(self.inner_dim//2, self.inner_dim//2, kernel_size=(3, 3), padding=(1, 1), groups=self.inner_dim//2, bias=False)
        self.bncs2 = nn.BatchNorm2d(self.inner_dim//2)
        self.cs2 = nn.Conv2d(self.inner_dim//2, self.inner_dim//2, kernel_size=(3, 3), padding=(1, 1), groups=self.inner_dim//2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.reshape(b, d, int(math.sqrt(n)), int(math.sqrt(n)))
        b, channelsss, h, w = x.shape

        qkvc = self.qkvc(self.act(self.bn(x)))
        qkvc = qkvc.contiguous().view(b, self.num_patch*self.num_patch, self.inner_dim * 4)
        qkvc = qkvc.chunk(4, dim=-1)
        q, k, v, c = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkvc)

        q_1 = q[:, 0:self.num_heads//2, :, :]
        k_1 = k[:, 0:self.num_heads//2, :, :]
        v_1 = v[:, 0:self.num_heads//2, :, :] 
        c_1 = c[:, 0:self.num_heads//2, :, :]
        q_2 = q[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, h, w)
        k_2 = k[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, h, w)
        v_2 = v[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, h, w)
        c_2 = c[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, h, w)

        q2_out = self.conv2_2(self.act(self.bn2_2(q_2)))
        k2_out = (self.avg2_3(self.act(self.bn2_3(k_2))) + q2_out).sigmoid()
        attn2 = v_2 * k2_out
        c2_out = self.cs1(self.act(self.bncs1(c_2)))
        c2_out = self.cs2(self.act(self.bncs2(c2_out)))
        attn_out = attn2 + c2_out

        scores = torch.einsum("b h i d, b h j d -> b h i j", q_1, k_1)
        scores = scores * self.scale
        attn = self.attn(scores)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v_1)
        out = einops.rearrange(out, "b h n d -> b n (h d)").reshape(b, self.inner_dim//2, h, w)
        c = self.local(self.act(self.bnc(c_1.reshape(b, self.inner_dim//2, self.num_patch, self.num_patch))))
        c = self.local1(self.act(self.bnc1(c)))

        res = torch.cat([out + c, attn_out], axis=1).reshape(b, h*w, -1)
        out = self.avgpool(res)
        return out


class FeedForward_ctf(nn.Module):

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


class trans_ctf(nn.Module):

    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), Attention_ctf(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), FeedForward_ctf(dim, hidden_dim, num_patch, patch_size))
            ]
            self.layers.append(nn.ModuleList(layer))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import numbers
from einops import rearrange
from einops.layers.torch import Reduce
from itertools import repeat
import collections.abc
# from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from thop import profile
from torchsummaryX import summary
import os
import math
from timm.models.layers import DropPath
import matplotlib.pyplot as plt

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x, (Hp, Wp)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Token Selective Attention
class Token_Selective_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, k, group_num):
        super(Token_Selective_Attention, self).__init__()
        self.num_heads = num_heads
        self.k = k
        self.group_num = group_num
        self.dim_group = dim // group_num
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(self.group_num, self.group_num * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(self.group_num * 3, self.group_num * 3, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                  groups=self.group_num * 3, bias=bias)  # 331
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b,self.group_num,c//self.group_num,h,w)
        b, t, c, h, w = x.shape  # 2,4,32,8,8

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = rearrange(q, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        k = rearrange(k, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, _, N = q.shape  # N=hw

        mask = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)

        attn = (q.transpose(-2, -1) @ k) * self.temperature  # [b, hw, hw]

        index = torch.topk(attn, k=int(N * self.k), dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # [b, c, hw]

        out = rearrange(out, 'b head c (h w t) -> b t (head c) h w', head=self.num_heads, h=h, w=w)

        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)


        return out

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class block(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)
        attn1 = self.conv0(x)  # conv_3*3
        attn2 = self.conv_spatial(attn1)  # conv_3*3 -> conv_5*5

        attn1 = self.conv1(attn1) # b, dim/2, h, w
        attn2 = self.conv2(attn2) # b, dim/2, h, w

        attn = torch.cat([attn1, attn2], dim=1)  # b,c,h,w
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # b,1,h,w
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # b,1,h,w
        agg = torch.cat([avg_attn, max_attn], dim=1) # spa b,2,h,w

        ch_attn1 = self.global_pool(attn) # b,dim,1, 1
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)

        a1,a2 =  a_b.chunk(2, dim=1)
        a1 = a1.reshape(batch_size,dim // 2,1,1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)

        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1)

        attn = attn1 * w1 + attn2 * w2
        attn = self.conv(attn).sigmoid()

        return x * attn

class FFN(nn.Module):
    def __init__(self, dim, bias,kernel_size):
        super(FFN, self).__init__()
        if kernel_size not in [3, 5, 7]:
            raise ValueError("Invalid kernel_size. Must be 3, 5, or 7.")

        self.kernel_size = kernel_size
        hidden_features = 180

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,groups=hidden_features, bias=bias) #dwconv
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_features, bias=bias)
        self.relu3 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        if self.kernel_size == 3:
            x = self.relu3(self.dwconv3x3(x))
        elif self.kernel_size == 5:
            x = self.relu3(self.dwconv5x5(x))
        elif self.kernel_size == 7:
            x = self.relu3(self.dwconv7x7(x))
        x = self.project_out(x)

        return x

class Attention_KSB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = block(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)    # conv 1×1
        x = self.activation(x) # GELU
        x = self.spatial_gating_unit(x) # LSKblock
        x = self.proj_2(x)   # conv 1×1
        x = x + shorcut
        return x

class Mlp_KSB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)     # conv 1*1
        x = self.dwconv(x)  # dwconv
        x = self.act(x)     # GELU
        x = self.drop(x)
        x = self.fc2(x)    # conv 1*1
        x = self.drop(x)
        return x

class Transformer_KSFA(nn.Module):
    def __init__(self, dim, mlp_ratio=1., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention_KSB(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_KSB(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class Transformer_TSFA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, kernel_size, k, group_num):
        super(Transformer_TSFA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_selective = Token_Selective_Attention(dim, num_heads, bias, k, group_num)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn_conv = FFN(dim, bias, kernel_size)

    def forward(self, x):
        x = x + self.attn_selective(self.norm1(x))  # selective_Attention_topk

        x = x + self.ffn_conv(self.norm2(x))

        return x


class dsformer(nn.Module):
    def __init__(self,
                 embed_dim=64,
                 img_size=128,
                 patch_size=4,
                 norm_layer=nn.LayerNorm,
                 num_heads=8,
                 depth=6,
                 drop_rate=0.,
                 num_classes=9,
                 kernel_size=3,
                 k=0.8,
                 group_num=4
                 ):
        super(dsformer, self).__init__()

        self.num_features = embed_dim
        self.conv0 = nn.Conv2d(30, embed_dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer)

        self.blocks_TSFA = nn.ModuleList([
            Transformer_TSFA(
                dim=embed_dim, num_heads=num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias',
                kernel_size=kernel_size, k=k, group_num=group_num
            )
            for i in range(depth)])

        self.block_KSFA = Transformer_KSFA(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )


    def forward(self, x):
        b,_,c,h,w = x.size()
        x = torch.squeeze(x)

        x = self.conv0(x)  # b, dim, PS ,PS

        x1,(Hp, Wp) = self.patch_embed(x)
        _,_,dim = x1.size() # b,len_seq, embed_dim
        x1 = self.pos_drop(x1)

        x1 = x1.view(b, Hp,Wp,dim)
        x1 = x1.permute(0,3,1,2)

        for i, blk in enumerate(self.blocks_TSFA):
            if i == 0 or i == 3:
                x1 = self.block_KSFA(x1)
            x1 = blk(x1)

        x = self.mlp_head(x1)

        return x


def DSFormer(dataset,kernel_size, ps, k, group_num, emb_dim):
    model = None
    if dataset == 'pu':
        model = dsformer(
            embed_dim=emb_dim,
            img_size=256,
            patch_size=ps,
            norm_layer=nn.LayerNorm,
            num_heads=8,
            depth=6,
            drop_rate=0.,
            num_classes=9,
            kernel_size=kernel_size,
            k = k,
            group_num=group_num
        )
    elif dataset == 'ip':
        model = dsformer(
            embed_dim=emb_dim,
            img_size=256,
            patch_size=ps,
            norm_layer=nn.LayerNorm,
            num_heads=8,
            depth=6,
            drop_rate=0.,
            num_classes=16,
            group_num=group_num
        )
    elif dataset == 'houston13':
        model = dsformer(
            embed_dim=emb_dim,
            img_size=256,
            patch_size=ps,
            norm_layer=nn.LayerNorm,
            num_heads=8,
            depth=6,
            drop_rate=0.,
            num_classes=15,
            group_num=group_num
        )
    elif dataset == 'whuhh':
        model = dsformer(
            embed_dim=emb_dim,
            img_size=256,
            patch_size=ps,
            norm_layer=nn.LayerNorm,
            num_heads=8,
            depth=6,
            drop_rate=0.,
            num_classes=22,
            group_num=group_num
        )
    return model



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    input = torch.rand(2,1,30,32,32).cuda()

    model = DSFormer(dataset='ip', kernel_size=3,  ps=2, k=0.8, group_num=4, emb_dim=128).cuda()
    output= model(input)
    print(output.size())

    # summary(model, torch.zeros((2, 1, 200, 8, 8)).cuda())
    flops, params = profile(model, inputs=(input,))
    print('Param:{} K' .format(params/1e3))
    print('Flops:{} M' .format(flops/1e6))  ## 打印计算量

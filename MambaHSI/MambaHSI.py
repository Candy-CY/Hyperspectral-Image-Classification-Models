import math
import torch
from torch import nn
from mamba_ssm import Mamba


class SpeMamba(nn.Module):
    def __init__(self,channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self,x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class SpaMamba(nn.Module):
    def __init__(self,channels,use_residual=True,group_num=4,use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape
        x_flat = x_re.view(1,-1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self,channels,token_num,use_residual,group_num=4,use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels,use_residual=use_residual,group_num=group_num)
        self.spe_mamba = SpeMamba(channels,token_num=token_num,use_residual=use_residual,group_num=group_num)

    def forward(self,x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class MambaHSI(nn.Module):
    def __init__(self,in_channels=128,hidden_dim=64,num_classes=10,use_residual=True,mamba_type='both',token_num=4,group_num=4,use_att=True):
        super(MambaHSI, self).__init__()
        self.mamba_type = mamba_type

        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                             nn.GroupNorm(group_num,hidden_dim),
                                             nn.SiLU())
        if mamba_type == 'spa':
            self.mamba = nn.Sequential(SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                        SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                        SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num)
                                        )

        elif mamba_type=='both':
            self.mamba = nn.Sequential(BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                       BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                       BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       )


        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
                                      nn.GroupNorm(group_num,128),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=128,out_channels=num_classes,kernel_size=1,stride=1,padding=0))

    def forward(self,x):

        x = self.patch_embedding(x)
        x = self.mamba(x)

        logits = self.cls_head(x)
        return logits



# if __name__=='__main__':
#     batch, length, dim = 2, 512*512, 256
#     x = torch.randn(batch, length, dim).to("cuda")
#     model = Mamba(
#         # This module uses roughly 3 * expand * d_model^2 parameters
#         d_model=dim,  # Model dimension d_model
#         d_state=16,  # SSM state expansion factor
#         d_conv=4,  # Local convolution width
#         expand=2,  # Block expansion factor
#     ).to("cuda")
#     y = model(x)
#     assert y.shape == x.shape
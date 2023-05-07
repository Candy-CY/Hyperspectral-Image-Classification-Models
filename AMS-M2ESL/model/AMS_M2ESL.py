# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : AMS_M2ESL.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

import torch
import torch.nn as nn
import model.module.AMIPS as AMIPS
import model.module.DCR as DCR
import model.module.manifold_learning as SPD_net
import model.module.EucProject as EP

# import model.module.MPA_Lya as MPA


class AMS_M2ESL(nn.Module):
    def __init__(self, in_channels, patch_size, class_count, ds_name):
        super(AMS_M2ESL, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.class_count = class_count
        self.ds = ds_name

        self.channels_1 = 2
        self.inter_num = 2

        # for AMIPS
        self.am_ip_sampling = AMIPS.AM_IPS(self.ds)

        # for DC-DCR
        self.spe_spa_corr_mine = DCR.Spectral_corr_mining(self.in_channels)
        self.dw_deconv_5 = nn.ConvTranspose2d(self.in_channels, self.in_channels, kernel_size=5, stride=1,
                                              padding=5 // 2, groups=self.in_channels)
        self.dw_conv_5 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=5, stride=1, padding=5 // 2,
                                   groups=self.in_channels)

        # for SPD mainifold subspace learning
        '''
        the implementation of deep mainifold learning is mainly based on the source code of RBN,
        i.e., A Riemannian Network for SPD Matrix Learning, NeurIPS 2019
        '''
        self.bit_map = SPD_net.BiMap(self.channels_1, self.channels_1, self.in_channels, self.in_channels)
        self.re_eig = SPD_net.ReEig()

        # BN test
        self.bn_spd = SPD_net.BatchNormSPD(self.in_channels)

        # for Euclidean projection
        self.app_mat_sqrt = EP.ASQRT_autograd_mc(norm_type='Frob_n', num_iter=2)

        # ASQRT test
        # self.log_eig=SPD_net.LogEig()
        # self.sqrt_eig=SPD_net.SqmEig()
        # self.sqrt_MPA_Lya = MPA.MPA_Lya.apply

        # for Euclidean subspace learning
        self.bn = nn.BatchNorm2d(self.channels_1)
        self.flatten = nn.Flatten(1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

        if self.ds == 'IP':
            self.fc_0 = nn.Linear(7200, 512)
        elif self.ds == 'UP':
            self.fc_0 = nn.Linear(1922, 512)
        elif self.ds == 'UH_tif':
            self.fc_0 = nn.Linear(3872, 512)

        self.fc_1 = nn.Linear(512, 128)
        self.fc_2 = nn.Linear(128, class_count)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        # AMIPS
        x_sampled = self.am_ip_sampling(x)

        # DC-DCR
        x_channel_1 = self.spe_spa_corr_mine(x_sampled)

        x_deC = self.dw_deconv_5(x_sampled)
        x_deC_C = self.dw_conv_5(x_deC)
        x_channel_2 = self.spe_spa_corr_mine(x_deC_C)

        x_channel_1 = torch.unsqueeze(x_channel_1, dim=1)
        x_channel_2 = torch.unsqueeze(x_channel_2, dim=1)
        a_0 = torch.cat((x_channel_1, x_channel_2), dim=1)

        # M2ESL
        a_1 = self.bit_map(a_0)
        # a_1=self.bn_spd(a_1)
        a_2 = self.re_eig(a_1.cpu())

        a_2_proj = self.app_mat_sqrt(a_2.cuda())

        # a_2_proj=self.log_eig(a_2.cpu())
        # a_2_proj=self.sqrt_eig(a_2.cpu())
        # a_2_proj = self._sqrt_mpa_c2(a_2.cuda())

        a_2_2 = self.bn(a_2_proj)
        a_3 = self.flatten(a_2_2)

        a_3_2 = self.fc_0(a_3)
        a_4 = self.fc_1(a_3_2)
        a_4_2 = self.sigmoid(a_4)
        a_4_3 = self.dropout(a_4_2)

        out = self.fc_2(a_4_3)
        return out

    def _sqrt_mpa_c2(self, x):
        x_channel_0, x_channel_1 = x[:, 0], x[:, 1]
        x_channel_0_sqrt, x_channel_1_sqrt = self.sqrt_MPA_Lya(x_channel_0), self.sqrt_MPA_Lya(x_channel_1)
        x_channel_0_sqrt, x_channel_1_sqrt = torch.unsqueeze(x_channel_0_sqrt, dim=1), torch.unsqueeze(x_channel_1_sqrt,
                                                                                                       dim=1)
        out = torch.cat((x_channel_0_sqrt, x_channel_1_sqrt), dim=1)
        return out


def AMS_M2ESL_(in_channels, patch_size, num_classes, ds):
    model = AMS_M2ESL(in_channels, patch_size, num_classes, ds)
    return model

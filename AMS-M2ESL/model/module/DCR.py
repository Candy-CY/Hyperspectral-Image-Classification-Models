# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : DCR.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
for the two implementations of distance covariance represntation (DCR),
the implementation1 (_DCR_1) is based on DeepBDC, i.e.,
Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification, CVPR 2022 
https://github.com/Fei-Long121/DeepBDC/blob/main/methods/bdc_module.py,
and _DCR_2 is our original implementation based on Brownian distance covariance, https://doi.org/10.1214/09-AOAS312.
the two implementations achieve similar performance in our AMS-M2ESL framework, we randomly chose _DCR_1 as the final version.
'''


class Spectral_corr_mining(nn.Module):
    def __init__(self, in_channels):
        super(Spectral_corr_mining, self).__init__()
        self.temperature = nn.Parameter(
            torch.log((3.2 / (in_channels * in_channels)) * torch.ones(1, 1, device=device)), requires_grad=True)

    def forward(self, x):
        x_corr = self._DCR_1(x, self.temperature)
        # x_corr=self._DCR_2(x)

        # for abla of DCR
        # x_corr=self._CR(x)

        return x_corr

    def _DCR_1(self, x, t):
        len_x = len(x.size())

        if len_x == 3:
            # spatial
            batchSize, c, h_w = x.size()
            x = x.permute(0, 2, 1)
            c = h_w
        elif len_x == 4:
            # spectral channel
            batchSize, c, h, w = x.size()
            h_w = h * w
            x = x.reshape(batchSize, c, h_w)

        I = torch.eye(c, c, device=x.device).view(1, c, c).repeat(batchSize, 1, 1).type(x.dtype)
        I_M = torch.ones(batchSize, c, c, device=x.device).type(x.dtype)
        x_pow2 = x.bmm(x.transpose(1, 2))
        dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2

        dcov = torch.clamp(dcov, min=0.0)
        dcov = torch.exp(t) * dcov
        dcov = torch.sqrt(dcov + 1e-5)

        out = dcov - 1. / c * dcov.bmm(I_M) - 1. / c * I_M.bmm(dcov) + 1. / (c * c) * I_M.bmm(dcov).bmm(I_M)

        return out * (-1)

    def _DCR_2(self, x):
        batch_size, c, h, w = x.size()

        x = x.view(batch_size, -1, h * w).permute(0, 2, 1)

        x = x.permute(0, 2, 1)
        x1, x2 = x[:, :, None], x[:, None]
        x3 = x1 - x2
        band_l2_mat = torch.norm(x3, dim=3, p=2)

        bem_mean_row, becm_mean_col = torch.mean(band_l2_mat, dim=1, keepdim=True), torch.mean(band_l2_mat, dim=2,
                                                                                               keepdim=True)
        bem_mean_row_expand, becm_mean_col_expand = bem_mean_row.expand(band_l2_mat.shape), becm_mean_col.expand(
            band_l2_mat.shape)
        bem_mean_plus_row_col = bem_mean_row_expand + becm_mean_col_expand
        bem_mean_all = torch.mean(bem_mean_row, dim=2)
        becm = band_l2_mat - bem_mean_plus_row_col + torch.unsqueeze(bem_mean_all, dim=-1)

        return becm * (-1)

    def _CR(self, x):
        batch_size, c, h, w = x.size()

        x = x.view(batch_size, -1, h * w).permute(0, 2, 1)
        mean_pixel = torch.mean(x, dim=1, keepdim=True)
        mean_pixel_expand = mean_pixel.expand(x.shape)

        x_cr = x - mean_pixel_expand
        CR = torch.bmm(x_cr.permute(0, 2, 1), x_cr)
        CR = torch.div(CR, h * w - 1)

        return CR

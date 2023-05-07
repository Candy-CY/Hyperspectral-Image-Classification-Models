# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : AMIPS.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

import torch
import torch.nn as nn
from torch import cosine_similarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AM_IPS(nn.Module):
    def __init__(self, ds):
        super(AM_IPS, self).__init__()
        self.ds = ds

    def forward(self, raw_patch):
        bt, c, h, w = raw_patch.size()

        patch_2d = raw_patch.view(bt, -1, h * w).permute(0, 2, 1)

        # central spectral vector sampling
        cent_spec_vec = patch_2d[:, int((h * w - 1) / 2)]
        cent_spec_vec = torch.unsqueeze(cent_spec_vec, dim=1)

        # central spectral vector oriented similarity
        sim_mat = self._sim_euc(cent_spec_vec, patch_2d)
        # sim_mat = self._sim_mat_mul(cent_spec_vec, patch_2d)
        # sim_mat = self._sim_cos(cent_spec_vec, patch_2d)

        if self.ds == 'UH_tif':
            threshold_sampling = torch.mean(sim_mat, dim=1) - 0.25 * torch.std(sim_mat, dim=1)
        else:
            threshold_sampling = torch.mean(sim_mat, dim=1) - 0.2 * torch.std(sim_mat, dim=1)

        # sampling
        threshold_mat = torch.unsqueeze(threshold_sampling, dim=1) * torch.ones_like(sim_mat)
        threshold_mask = sim_mat - threshold_mat

        index_mask = torch.where(threshold_mask >= 0, 1, 0)
        index_mask = torch.unsqueeze(index_mask, -1)
        x_sampling = index_mask * patch_2d

        return x_sampling.contiguous().view(bt, h, w, c).permute(0, 3, 1, 2)

    def _sim_mat_mul(self, central_vector, x_2d):
        sim_M = torch.bmm(x_2d, central_vector.permute(0, 2, 1))
        return torch.squeeze(sim_M, dim=-1)

    def _sim_euc(self, central_vector, x_2d):
        bt, h_w, c = x_2d.size()
        cen_vec_mat = central_vector.expand(bt, h_w, c)
        euc_dist = torch.norm(cen_vec_mat - x_2d, dim=2, p=2)
        sim_M = 1 / (1 + euc_dist)
        return sim_M

    def _sim_cos(self, central_vector, x_2d):
        sim_M = cosine_similarity(central_vector, x_2d, dim=2)
        return sim_M

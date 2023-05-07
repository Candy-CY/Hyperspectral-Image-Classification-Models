# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : EucProject.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

'''
the ASQRT layer is implemented based on the source code of COSONet, i.e.,
COSONet: Compact Second-Order Network for Video Face Recognition, ACCV 2018,
https://github.com/YirongMao/COSONet/blob/master/layer_utils.py

the earliest version is surly based on excellent work, iSQRT-Conv, i.e.,
Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization, CVPR 2018,
https://github.com/jiangtaoxie/fast-MPN-COV/blob/master/src/representation/MPNCOV.py
'''

import torch
import torch.nn as nn
from torch.autograd import Variable


# ASQRT for multichannels via autograd
class ASQRT_autograd_mc(nn.Module):

    def __init__(self, norm_type, num_iter):
        super(ASQRT_autograd_mc, self).__init__()
        self.norm_type = norm_type
        self.num_iter = num_iter

    def forward(self, A):
        b_s, c, n_c, n_c = A.size()
        A = A.view(b_s * c, n_c, n_c)
        b_s_c = A.shape[0]

        dtype = A.dtype
        device = A.device
        # pre normalization
        if self.norm_type == 'Frob_n':
            normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        elif self.norm_type == 'Trace_n':
            I_bs_mat = torch.eye(n_c, n_c, device=A.device).view(1, n_c, n_c).expand_as(A).type(dtype)
            normA = A.mul(I_bs_mat).sum(dim=1).sum(dim=1)
        else:
            raise NameError('invalid normalize type {}'.format(self.norm_type))

        Y = A.div(normA.view(b_s_c, 1, 1).expand_as(A))
        # Iteration
        I = Variable(torch.eye(n_c, n_c).view(1, n_c, n_c).
                     repeat(b_s_c, 1, 1).type(dtype).to(device), requires_grad=False)
        Z = Variable(torch.eye(n_c, n_c).view(1, n_c, n_c).
                     repeat(b_s_c, 1, 1).type(dtype).to(device), requires_grad=False)

        for i in range(self.num_iter):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)

        # post compensation
        sA = Y * torch.sqrt(normA).view(b_s_c, 1, 1).expand_as(A)

        sA = sA.view(b_s, c, n_c, n_c)
        del I, Z
        return sA

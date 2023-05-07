# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : manifold_learning.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

# based on the source code of RBN, i.e., A Riemannian Network for SPD Matrix Learning, NeurIPS 2019
# https://proceedings.neurips.cc/paper/2019/hash/6e69ebbfad976d4637bb4b39de261bf7-Abstract.html

import torch
import torch.nn as nn
import model.module.manifold_learning_fun as m_fun

dtype = torch.float64
device = torch.device('cuda')


class BiMap(nn.Module):
    """
    Input X: (batch_size,hi) SPD matrices of size (ni,ni)
    Output P: (batch_size,ho) of bilinearly mapped matrices of size (no,no)
    Stiefel parameter of size (ho,hi,ni,no)
    """

    def __init__(self, ho, hi, ni, no):
        super(BiMap, self).__init__()
        self._W = m_fun.StiefelParameter(
            torch.empty(ho, hi, ni, no, dtype=dtype, device=device))
        self._ho = ho
        self._hi = hi
        self._ni = ni
        self._no = no
        m_fun.init_bimap_parameter(self._W)
        # self._no

    def forward(self, X):
        return m_fun.bimap_channels(X, self._W)


class ReEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return m_fun.ReEig.apply(P.cpu())
        # return m_fun.ReEig.apply(P)


class LogEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return m_fun.LogEig.apply(P)


class SqmEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of sqrt eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return m_fun.SqmEig.apply(P)


class BatchNormSPD(nn.Module):
    """
    Input X: (N,h) SPD matrices of size (n,n) with h channels and batch size N
    Output P: (N,h) batch-normalized matrices
    SPD parameter of size (n,n)
    """

    def __init__(self, n):
        super(__class__, self).__init__()
        self.momentum = 0.1
        self.running_mean = torch.eye(
            n, dtype=dtype, device=device)  ################################
        # self.running_mean=nn.Parameter(th.eye(n,dtype=dtype),requires_grad=False)
        self.weight = m_fun.SPDParameter(torch.eye(n, dtype=dtype, device=device))

    def forward(self, X):
        N, h, n, n = X.shape
        X_batched = X.permute(2, 3, 0,
                              1).contiguous().view(n, n, N * h,
                                                   1).permute(2, 3, 0,
                                                              1).contiguous()
        if (self.training):
            mean = m_fun.BaryGeom(X_batched)
            with torch.no_grad():
                self.running_mean.data = m_fun.geodesic(
                    self.running_mean, mean, self.momentum)
            X_centered = m_fun.CongrG(X_batched, mean, 'neg')
        else:
            X_centered = m_fun.CongrG(X_batched, self.running_mean,
                                      'neg')  # subtract mean
        X_normalized = m_fun.CongrG(X_centered, self.weight,
                                    'pos')  # add bias
        return X_normalized.permute(2, 3, 0,
                                    1).contiguous().view(n, n, N, h).permute(
            2, 3, 0, 1).contiguous()

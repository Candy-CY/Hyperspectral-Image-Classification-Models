# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : manifold_learning_fun.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

# based on the source code of RBN, i.e., A Riemannian Network for SPD Matrix Learning, NeurIPS 2019

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function as F


class StiefelParameter(nn.Parameter):
    """ Parameter constrained to the Stiefel manifold (for BiMap layers) """
    pass


class SPDParameter(nn.Parameter):
    """ Parameter constrained to the SPD manifold (for ParNorm) """
    pass


def init_bimap_parameter(W):
    """ initializes a (ho,hi,ni,no) 4D-StiefelParameter"""
    # C* = C +λI , where I is the identity matrix, λ could be set to \alpha× trace(C ), and \alpha is a very small value like 10^(-6).
    ho, hi, ni, no = W.shape
    for i in range(ho):  # can vectorize
        for j in range(hi):  # can vectorize
            v = torch.empty(ni, ni, dtype=W.dtype,
                            device=W.device).uniform_(0., 1.)
            inp_svd = v.matmul(v.t())
            alpha = 1e-5
            inp_svd = add_id_matrix(inp_svd, alpha)
            # vv = torch.svd(inp_svd)[0][:, :no]
            vv = torch.svd(inp_svd.cpu())[0][:, :no]
            W.data[i, j] = vv.cuda()


def add_id_matrix(P, alpha):
    '''
    Input P of shape (batch_size,1,n,n)
    Add Id
    '''
    P = P + alpha * P.trace() * torch.eye(
        P.shape[-1], dtype=P.dtype, device=P.device)
    return P


def bimap(X, W):
    '''
    Bilinear mapping function
    :param X: Input matrix of shape (batch_size,n_in,n_in)
    :param W: Stiefel parameter of shape (n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    # print(W.dtype)
    # print(X.dtype)
    # print(X.shape)
    # print(W.shape)
    return W.t().float().matmul(X.float()).matmul(W.float())


def bimap_channels(X, W):
    '''
    Bilinear mapping function over multiple input and output channels
    :param X: Input matrix of shape (batch_size,channels_in,n_in,n_in)
    :param W: Stiefel parameter of shape (channels_out,channels_in,n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,channels_out,n_out,n_out)
    '''
    # Pi=th.zeros(X.shape[0],1,W.shape[-1],W.shape[-1],dtype=X.dtype,device=X.device)
    # for j in range(X.shape[1]):
    #     Pi=Pi+bimap(X,W[j])
    batch_size, channels_in, n_in, _ = X.shape
    channels_out, _, _, n_out = W.shape
    P = torch.zeros(batch_size,
                    channels_out,
                    n_out,
                    n_out,
                    dtype=X.dtype,
                    device=X.device)
    for co in range(channels_out):
        P[:, co, :, :] = sum([
            bimap(X[:, ci, :, :], W[co, ci, :, :]) for ci in range(channels_in)
        ])
    return P


class Re_op():
    """ Relu function and its derivative """
    _threshold = 1e-4

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).double()


class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward_re(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Re_op)


def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size, channels, n = P.shape  # batch size,channel depth,dimension
    Q = torch.zeros(batch_size, channels, n, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):  # can vectorize
        for j in range(channels):  # can vectorize
            Q[i, j] = P[i, j].diag()
    return Q


def modeig_forward_re(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape
    U, S = torch.zeros_like(P, device=P.device), torch.zeros(batch_size,
                                                             channels,
                                                             n,
                                                             dtype=P.dtype,
                                                             device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if (eig_mode == 'eig'):
                s, U[i, j] = torch.linalg.eig(P[i, j], True)
                S[i, j] = s[:, 0]
            elif (eig_mode == 'svd'):
                U[i, j], S[i, j], _ = torch.svd(add_id_matrix(P[i, j], 1e-5))
                # U[i, j], S[i, j], _ = torch.svd(add_id_matrix(P[i, j].cpu(), 1e-5))
    # U, S = U.cuda(), S.cuda()
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def modeig_forward_etc(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape
    U, S = torch.zeros_like(P, device=P.device), torch.zeros(batch_size,
                                                             channels,
                                                             n,
                                                             dtype=P.dtype,
                                                             device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if (eig_mode == 'eig'):
                # s, U[i, j] = torch.linalg.eig(P[i, j])
                s, U[i, j] = torch.eig(P[i, j].cpu(), True)
                S[i, j] = s[:, 0]
            elif (eig_mode == 'svd'):
                # U[i, j], S[i, j], _ = torch.svd(add_id_matrix(P[i, j], 1e-5))
                U[i, j], S[i, j], _ = torch.svd(add_id_matrix(P[i, j].cpu(), 1e-5))
    U, S = U.cuda(), S.cuda()
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def modeig_backward(dx, U, S, S_fn, op, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    # print("Correct back prop")

    S_fn_deriv = BatchDiag(op.fn_deriv(S, param)).float()
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SS_fn = S_fn[..., None].repeat(1, 1, 1, S_fn.shape[-1])
    L = (SS_fn - SS_fn.transpose(2, 3)) / (SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0
    L[L == np.inf] = 0
    L[torch.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp


class Sqm_op():
    """ sqrt function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return torch.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 0.5 / torch.sqrt(S)


class Sqminv_op():
    """ Inverse sqrt function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return 1 / torch.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return -0.5 / torch.sqrt(S) ** 3


class SqmEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        # X, U, S, S_fn = modeig_forward_re(P, Sqm_op)
        X, U, S, S_fn = modeig_forward_etc(P, Sqm_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqm_op)


class SqminvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward_etc(P, Sqminv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqminv_op)


def CongrG(P, G, mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    P, G = P.float(), G.float()
    if (mode == 'pos'):
        GG = SqmEig.apply(G[None, None, :, :])
    elif (mode == 'neg'):
        GG = SqminvEig.apply(G[None, None, :, :])
    PP = GG.matmul(P.float()).matmul(GG.float())
    return PP


class Log_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return torch.log(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 1 / S


class Re_op():
    """ Relu function and its derivative """
    _threshold = 1e-4

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).double()


class LogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        # X, U, S, S_fn = modeig_forward_re(P, Sqm_op)
        X, U, S, S_fn = modeig_forward_etc(P, Log_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Log_op)


def LogG(x, X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    return CongrG(LogEig.apply(CongrG(x, X, 'neg')), X, 'pos')


class Exp_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return torch.exp(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return torch.exp(S)


class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward_etc(P, Exp_op, eig_mode='eig')
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Exp_op)


class Power_op():
    """ Power function and its derivative """
    _power = 1

    @classmethod
    def fn(cls, S, param=None):
        return S ** cls._power

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (cls._power) * S ** (cls._power - 1)


class PowerEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P, power):
        Power_op._power = power
        X, U, S, S_fn = modeig_forward_etc(P, Power_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Power_op), None


def geodesic(A, B, t):
    '''
    Geodesic from A to B at step t
    :param A: SPD matrix (n,n) to start from
    :param B: SPD matrix (n,n) to end at
    :param t: scalar parameter of the geodesic (not constrained to [0,1])
    :return: SPD matrix (n,n) along the geodesic
    '''
    M = CongrG(PowerEig.apply(CongrG(B.float(), A.float(), 'neg'), t), A, 'pos')[0, 0]
    return M


def ExpG(x, X):
    """ Exponential mapping of x on the SPD manifold at X """
    return CongrG(ExpEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def karcher_step(x, G, alpha):
    '''
    One step in the Karcher flow
    '''
    x_log = LogG(x, G)
    G_tan = x_log.mean(dim=0)[None, ...]
    G = ExpG(alpha * G_tan, G)[0, 0]
    return G


def BaryGeom(x, by_channel=False):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    k = 1
    alpha = 1
    batch_size = x.shape[0]
    channels = x.shape[1]
    n = x.shape[2]
    G = []
    if by_channel == True:
        for i in range(batch_size):
            inp = x[i, :, :, :]
            inp = inp.view(channels, 1, x.shape[2], x.shape[3])
            G_sample = torch.mean(inp, dim=0)[0, :, :]
            for _ in range(k):
                G_sample = karcher_step(inp, G_sample, alpha)
                G_sample.view(1, G_sample.shape[0], G_sample.shape[1])
            G.append(G_sample)
        G = torch.cat(G, dim=0)
        G = G.view(batch_size, 1, n, n)
    else:
        # with th.no_grad():
        G = torch.mean(x, dim=0)[0, :, :]
        for _ in range(k):
            G = karcher_step(x, G, alpha)
    return G

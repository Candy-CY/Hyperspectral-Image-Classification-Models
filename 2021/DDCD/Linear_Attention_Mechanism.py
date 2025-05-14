###########################################################################
# Created by: Rui Li
# Copyright (c) 2020
###########################################################################
import torch
from torch.nn import Module, Conv3d, Parameter, Softmax
__all__ = ['PositionLinearAttention', 'ChannelLinearAttention']
def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

class PositionLinearAttention(Module):
    """Position linear attention"""
    def __init__(self, in_places, eps=1e-6):
        super(PositionLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps
        self.query_conv = Conv3d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.key_conv = Conv3d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.value_conv = Conv3d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, C, width, height, channel = x.size()
        Q = self.query_conv(x).view(batch_size, -1, width * height * channel)
        K = self.key_conv(x).view(batch_size, -1, width * height * channel)
        V = self.value_conv(x).view(batch_size, -1, width * height * channel)
        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)
        #torch.einsum函数()：Einsum 可以计算向量、矩阵、张量运算，
        # 包括计算 transposes、sum、column/row sum、
        # Matrix-Vector Multiplication、Matrix-Matrix Multiplication。
        tailor_sum = 1 / (width * height * channel + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        #unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度。
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        # expand（）函数的功能是用来扩展张量中某维数据的尺寸，它返回输入张量在某维扩展为更大尺寸后的张量。
        value_sum = value_sum.expand(-1, C, width * height * channel)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)
        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, C, height, width, channel)
        return (x + self.gamma * weight_value).contiguous()

class ChannelLinearAttention(Module):
    """Channel linear attention"""
    def __init__(self, eps=1e-6):
        super(ChannelLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.l2_norm = l2_norm
        self.eps = eps
    def forward(self, x):
        batch_size, C, width, height, channel = x.shape
        Q = x.view(batch_size, C, -1)
        K = x.view(batch_size, C, -1)
        V = x.view(batch_size, C, -1)
        Q = self.l2_norm(Q)
        #permute函数可以对任意高维矩阵进行转置
        K = self.l2_norm(K).permute(-3, -1, -2)
        # tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, t))
        tailor_sum = 1 / (width * height * channel + torch.einsum("bnc, bn->bc", K, torch.sum(Q, dim=-2) + self.eps))
        value_sum = torch.einsum("bcn->bn", V).unsqueeze(-1).permute(0, 2, 1)
        value_sum = value_sum.expand(-1, C, width * height * channel)
        matrix = torch.einsum('bcn, bnm->bcm', V, K)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, Q)
        weight_value = torch.einsum("bcn, bc->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, C, height, width, channel)
        return (x + self.gamma * weight_value).contiguous()

if __name__ == "__main__":
    x = torch.rand((10, 16, 256, 256), dtype=torch.float)
    PLA = PositionLinearAttention(16)
    CLA = ChannelLinearAttention()
    print(PLA(x).shape, CLA(x).shape)

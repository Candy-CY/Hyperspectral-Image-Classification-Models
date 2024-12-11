import torch.nn.functional as F
import torch
import torch.nn as nn
from module import *

from einops import rearrange
from layer import GraphConvolution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout = 0.1
FM=4*2
# wavelet = 'bior1.3'  # 小波基函数
# wavelet = 'haar'  # 小波基函数


class GatedUnit(nn.Module):
    def __init__(self, feature_size, gate_size, num):
        super(GatedUnit, self).__init__()
        self.feature_size = feature_size  # 特征大小
        self.gate_size = gate_size  # 门控大小
        self.num =num

        # 定义门控权重
        self.gate_weights = nn.Parameter(torch.Tensor(3 * gate_size, 3 * feature_size))
        # self.Weight_Alpha = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(3 * gate_size))  # 不再对每个b和h定义bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.gate_weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.gate_weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x1, x2, x3):
        # weight_alpha = F.softmax(self.Weight_Alpha, dim=0)
        # 合并三个输入
        combined = torch.cat((x1, x2, x3), dim=-1)  # 维度变为 [b, h, 3 * feature_size]
        # 使用点乘和广播处理门控
        gating_scores = torch.einsum('bhi,ij->bhj', combined, self.gate_weights) + self.bias
        gating_scores = gating_scores.view(-1, self.num, self.gate_size, 3)  # 重新整理形状为 [b, h, gate_size, 3]
        gates = torch.softmax(gating_scores, dim=-1)  # 对最后一个维度进行softmax
        output = gates[..., 0] * x1 + gates[..., 1] * x2 + gates[..., 2] * x3
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):  # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
        super(GCN, self).__init__()  # super()._init_()在利用父类里的对象构造函数
        self.gc1 = GraphConvolution(nfeat, nhid)  # gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(nhid, out)  # gc2输入尺寸nhid，输出尺寸ncalss
        self.dropout = dropout

    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        x = self.gc2(x, adj)
        # return torch.log_softmax(x,dim=-1)
        return x

class lidar_conv(nn.Module):
    def __init__(self, in_channels, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(lidar_conv, self).__init__()
        self.lidar_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, FM, 3, 1, 0),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.Conv2d(FM, FM * 2, 3, 1, 0),
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(),
        )

    def forward(self, X_l):
        x = self.lidar_conv(X_l)
        return x

class feature_conv(nn.Module):
    def __init__(self, in_channels, patch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(feature_conv, self).__init__()
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 7, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, X_h):
        x1_0 = self.conv1_0(X_h)
        x1_1 = self.conv1_1(X_h)
        x1_2 = self.conv1_2(x1_1)
        x1_3 = self.conv1_3(x1_2)
        x = x1_0+x1_3
        return x

class Feature_HSI_Lidar(nn.Module):
    def __init__(self, in_channels, in_channels2, num_classes, patch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Feature_HSI_Lidar, self).__init__()
        self.stride = stride
        self.patch = patch
        self.F_1 = nn.Conv2d(in_channels, in_channels, (1,3), (1,1), padding, dilation, groups, bias)
        self.F_2 = nn.Conv2d(in_channels, in_channels, (3,1), (1,1), padding, dilation, groups, bias)
        self.down_hsi = nn.AdaptiveAvgPool2d((1,1))
        self.bn_h = nn.BatchNorm2d(in_channels)
        self.F_3 = nn.Conv2d(in_channels2, in_channels2, (1, 3), (1, 1), padding, dilation, groups, bias)
        self.F_4 = nn.Conv2d(in_channels2, in_channels2, (3, 1), (1, 1), padding, dilation, groups, bias)
        self.down_lidar = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_l = nn.BatchNorm2d(in_channels2)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear((in_channels+1)*9, num_classes)
        self.gcn_h = GCN(FM*4, FM*4, FM*4, dropout)
        self.gcn_l = GCN(FM*2, FM*2, FM*2, dropout)
        self.gcn_all = GCN(FM*3, FM*3, FM*3, dropout)

        self.lidar_conv1 = nn.Sequential(
            nn.Conv2d(in_channels2, 2, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, FM, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(FM),
            nn.ReLU(inplace=True),
            nn.Conv2d(FM, FM * 2, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.lidar_conv2 = nn.Sequential(
            nn.Conv2d(in_channels2, 2, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, FM, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(FM),
            nn.ReLU(inplace=True),
            nn.Conv2d(FM, FM * 2, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(FM * 4 * (patch-6) * (patch-6), 128),
            nn.BatchNorm1d(128),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.BatchNorm1d(num_classes),

        )

        self.gcn_h1 = GCN(FM * 4, FM * 4, FM * 4, dropout)
        self.gcn_l1 = GCN(FM * 2, FM * 2, FM * 2, dropout)
        self.lidar_conv = lidar_conv(in_channels=in_channels2, stride=1, padding=0)
        self.hsi_conv = feature_conv(FM*4, patch, kernel_size=3, stride=1, padding=0)
        self.gated_unit = GatedUnit(FM*4, FM*4, (patch-6)*(patch-6))

    def forward(self, x_h, h1, h2, x_l, l1, l2):

        out1 = x_h
        out2 = x_l

        l1 = torch.mean(l1, dim=1, keepdim=True)
        l2 = torch.mean(l2, dim=1, keepdim=True)
        x_h1 = self.F_1(x_h)
        while x_h1.shape[3]!=1:
            x_h1 = self.F_1(x_h1)   #3 1
        x_h1 = torch.sqrt(self.relu(self.bn_h(torch.matmul(x_h1, l1)))) # 3*3

        x_h2 = self.F_2(x_h)  # 1 3
        while x_h2.shape[2] != 1:
            x_h2 = self.F_2(x_h2)
        x_h2 = torch.sqrt(self.relu(self.bn_h(torch.matmul(l2, x_h2))))  # 3*3
        out3 = x_h1+x_h2

        h1 = torch.mean(h1, dim=1, keepdim=True)
        h2 = torch.mean(h2, dim=1, keepdim=True)
        x_l1 = self.F_3(x_l)  # 3 1
        while x_l1.shape[3] != 1:
            x_l1 = self.F_3(x_l1)  # 3 1
        x_l1 = torch.sqrt(self.relu(self.bn_l(torch.matmul(x_l1, h1))))  # 3*3

        x_l2 = self.F_4(x_l)  # 1 3
        while x_l2.shape[2] != 1:
            x_l2 = self.F_4(x_l2)
        x_l2 = torch.sqrt(self.relu(self.bn_l(torch.matmul(h2, x_l2))))  # 3*3
        out4 = x_l1+x_l2

        # CNN提取特征
        out3 = self.hsi_conv2(out3)
        out4 = self.lidar_conv2(out4)

        out1 = self.hsi_conv1(out1)
        out2 = self.lidar_conv1(out2)

        out1 = rearrange(out1, 'n c h w -> n (h w) c')
        out2 = rearrange(out2, 'n c h w -> n (h w) c')

        out1_mask = dist_mask(out1, device).to(device)
        out2_mask = dist_mask(out2, device).to(device)
        out1_1 = self.gcn_h1(out1, out1_mask)
        out2_1 = self.gcn_l1(out2, out2_mask)
        out1 = out1_1+out1
        out2 = out2_1+out2

        # 计算零阶矩阵
        out3 = rearrange(out3, 'n c h w -> n (h w) c')
        out4 = rearrange(out4, 'n c h w -> n (h w) c')
        Q_h = dist_mask(out3, device).to(device)
        Q_l = dist_mask(out4, device).to(device)

        # GCN_hsi
        out_h = self.gcn_h(out3, Q_h)
        # GCN_lidar
        out_l = self.gcn_l(out4, Q_l)
        out_h = out_h+out3
        out_l = out_l+out4

        # out_l2  out_h   out1
        out_l2 = torch.cat((out_l, out2), dim=2)

        output = self.gated_unit(out1, out_l2, out_h)
        # 分类
        out = rearrange(output, 'b h c->b (h c)')
        out = F.softmax(self.linear1(out), dim=-1)
        out = F.softmax(self.linear4(out), dim=-1)

        return out

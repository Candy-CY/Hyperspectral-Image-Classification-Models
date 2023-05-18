import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummaryX import summary
import math
from utils.ssn_sp import ssn_iter
import time

np.set_printoptions(threshold=np.inf)

class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3BNReLU, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(16,out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        return x


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.conv_q = nn.Conv1d(in_features, in_features//4, kernel_size=1, stride=1)
        self.conv_k = nn.Conv1d(in_features, in_features//4, kernel_size=1, stride=1)
        self.conv_v = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):

        batch, s, c = x.size()

        x = x.permute(0,2,1)

        query = self.conv_q(x)
        key = self.conv_k(x).permute(0, 2, 1)
        adj = F.softmax(torch.bmm(key, query), dim=-1)

        value = self.conv_v(x).permute(0, 2, 1)

        support = torch.matmul(value, self.weight)
        output = torch.bmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


def CalSpMean(sp, feature):

    c, h, w = feature.size()
    feature2d = feature.permute(1, 2, 0).reshape(h * w, c)
    sp_values = torch.unique(sp)
    sp_num = torch.unique(sp).shape[0]

    sp_mean = torch.zeros(sp_num, feature2d.shape[1]).cuda()

    for sp_idx in range(sp_num):
        region_pos_2d = torch.nonzero(sp == int(sp_values[sp_idx]))
        region_pos_1d = region_pos_2d[:, 0] * w + region_pos_2d[:, 1]
        sp_mean[sp_idx, :] = torch.mean(feature2d[region_pos_1d, :], dim=0)

    return sp_mean, sp_num


class SAGRN(nn.Module):
    def __init__(self, args, in_channel):

        super(SAGRN,self).__init__()

        self.args = args

        self.gcn = GraphConvolution(in_channel, in_channel)

        #self.conv = nn.Conv1d(in_channel, args.groups, kernel_size=1, stride=1)

        #self.conv= nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1)

        self.conv = Conv3x3BNReLU(in_channel, in_channel)

        self.conv_q = nn.Conv1d(in_channel, in_channel//4, kernel_size=1, stride=1)
        self.conv_k = nn.Conv1d(in_channel, in_channel//4, kernel_size=1, stride=1)
        self.conv_v = nn.Conv1d(in_channel, in_channel, kernel_size=1, stride=1)

    def forward(self, x):

        batch, c, h, w = x.size()

        xs = x#self.conv(x)

        _, sp, _ = ssn_iter(xs, num_spixels=self.args.sa_groups, n_iter=10)

        sp = sp.reshape(batch, h, w)

        output = []

        for i in range(batch):

            sp_mean, sp_num = CalSpMean(sp[i], xs[i])

            xpos = self.gcn(sp_mean.unsqueeze(0)).permute(0, 2, 1)

            query = self.conv_q(xpos)
            key = self.conv_k(x.reshape(batch,c,-1)).permute(0, 2, 1)
            atten = F.softmax(torch.bmm(key, query), dim=-1)
            value = self.conv_v(xpos).permute(0, 2, 1)

            output.append(torch.bmm(atten, value).permute(0, 2, 1).reshape(batch, c, h, w))

        output = torch.cat(output, dim=0)

        return [output, xs]


class SEGRN(nn.Module):

    def __init__(self, args, height, width):

        super(SEGRN,self).__init__()

        self.args = args

        self.pool = nn.AdaptiveAvgPool2d((height//4, width//4))

        pix_num = int((height//4) * (width//4))

        self.gcn = GraphConvolution(pix_num, pix_num)

        self.conv_q=nn.Conv1d(pix_num,pix_num//4,kernel_size=1,stride=1)
        self.conv_k=nn.Conv1d(pix_num,pix_num//4,kernel_size=1,stride=1)
        self.conv_v=nn.Conv1d(pix_num,pix_num,kernel_size=1,stride=1)

    def forward(self, x):

        batch, c, h, w = x.size()

        x = self.pool(x)

        length = int(c / self.args.se_groups)

        xpre = torch.zeros(batch, self.args.se_groups, (h//4) * (w//4)).cuda()

        start = 0

        end = start + length

        for i in range(self.args.se_groups):

            tmp=x[:,start:end,:,:]

            xpre[:,i,:]=torch.mean(tmp,dim=1).reshape(batch,-1)

            start = end

            end = start + length

        xpos = self.gcn(xpre).permute(0,2,1)

        query = self.conv_q(xpos)
        key = self.conv_k(x.reshape(batch,c,-1).permute(0, 2, 1)).permute(0, 2, 1)
        atten = F.softmax(torch.bmm(key, query), dim=-1)
        value = self.conv_v(xpos).permute(0, 2, 1)

        x = torch.bmm(atten, value).permute(0,2,1).reshape(batch, c, h//4, w//4)

        x = F.interpolate(x, size=(h, w), mode='bilinear',align_corners=True)

        return x


class SSGRN(nn.Module):

    def __init__(self, args, spec_band, num_classes, init_weights=True, in_channel=256):

        super(SSGRN, self).__init__()

        self.args=args

        self.backbone=nn.Sequential(
            Conv3x3BNReLU(spec_band, 64),
            nn.MaxPool2d(kernel_size=2,stride=2),
            Conv3x3BNReLU(64, 128),
            Conv3x3BNReLU(128, in_channel),
        )

        height = args.input_size[0] // 2
        width = args.input_size[1] // 2

        pix_num = int(height * width)

        if self.args.network == 'segrn':

            self.segrn = SEGRN(args, height, width)

            self.cls_se = nn.Sequential(
                Conv3x3BNReLU(in_channel, in_channel // 2),
                nn.Conv2d(in_channel // 2, num_classes, kernel_size=1, stride=1),
            )
        elif self.args.network == 'sagrn':

            self.sagrn = SAGRN(args,in_channel)

            self.cls_sa = nn.Sequential(
                Conv3x3BNReLU(in_channel, in_channel // 2),
                nn.Conv2d(in_channel // 2, num_classes, kernel_size=1, stride=1),
            )
            self.cls_sa2 = nn.Sequential(
                Conv3x3BNReLU(in_channel, in_channel // 2),
                nn.Conv2d(in_channel // 2, num_classes, kernel_size=1, stride=1),
            )

        elif self.args.network == 'ssgrn':

            self.segrn = SEGRN(args,  height, width)
            self.sagrn = SAGRN(args,in_channel)

            self.cls_se = nn.Sequential(
                Conv3x3BNReLU(in_channel, in_channel // 2),
                nn.Conv2d(in_channel // 2, num_classes, kernel_size=1, stride=1),
            )
            self.cls_sa = nn.Sequential(
                Conv3x3BNReLU(in_channel, in_channel // 2),
                nn.Conv2d(in_channel // 2, num_classes, kernel_size=1, stride=1),
            )
            self.cls_sa2 = nn.Sequential(
                Conv3x3BNReLU(in_channel, in_channel // 2),
                nn.Conv2d(in_channel // 2, num_classes, kernel_size=1, stride=1),
            )
            self.cls_ss = nn.Sequential(
                Conv3x3BNReLU(in_channel, in_channel // 2),
                nn.Conv2d(in_channel // 2, num_classes, kernel_size=1, stride=1),
            )
        elif self.args.network == 'fcn':

            self.cls_fcn = nn.Conv2d(in_channel, num_classes, kernel_size=1, stride=1)
        else:
            raise NotImplementedError

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        _, _, H, W = x.size()


        x = self.backbone(x)

        batch, _, h, w = x.size()

        if self.args.network=='segrn':

            se = self.segrn(x)

            se = F.interpolate(self.cls_se(se), size=(H, W), mode='bilinear',align_corners=True)

            return [se]

        elif self.args.network=='sagrn':

            sa, aux = self.sagrn(x)

            sa = F.interpolate(self.cls_sa(sa), size=(H, W), mode='bilinear',align_corners=True)
            aux = F.interpolate(self.cls_sa2(aux), size=(H, W), mode='bilinear',align_corners=True)

            return [sa,aux]

        elif self.args.network=='ssgrn':

            se = self.segrn(x)

            sa, aux = self.sagrn(x)

            re = se + sa + x

            re = F.interpolate(self.cls_ss(re), size=(H, W), mode='bilinear',align_corners=True)
            se = F.interpolate(self.cls_se(se), size=(H, W), mode='bilinear',align_corners=True)
            sa = F.interpolate(self.cls_sa(sa), size=(H, W), mode='bilinear',align_corners=True)
            aux = F.interpolate(self.cls_sa2(aux), size=(H, W), mode='bilinear')

            return [re,se,sa,aux]

        elif self.args.network == 'fcn':

            re = F.interpolate(self.cls_fcn(x), size=(H, W), mode='bilinear',align_corners=True)

            return [re]

        else:
            return NotImplementedError

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layers import GraphConvolution
import math

class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self):
        super(CAFM, self).__init__()
        
        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)
        
        self.avg1 = nn.Conv2d(192, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(192, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(192, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(192, 64, 1, stride=1, padding=0)
        
        self.avg11 = nn.Conv2d(64, 192, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, 192, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, 192, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, 192, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 =  F.relu(self.avg2(avg_2))
        max_2 =  F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2

        f1 = f1.squeeze(0)
        f2 = f2.squeeze(0)

        return f1.transpose(0, 1), f2.transpose(0, 1)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MultiHeadBlockGCN(nn.Module):  # Multihead Attention Fusion Module for GCN
    def __init__(self, num_attention_heads, input_size, hidden_dropout_prob=0.5):
        super(MultiHeadBlockGCN, self).__init__()
        hidden_size = input_size
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        attention_probs_dropout_prob = 0.5
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(0)

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = hidden_states.squeeze(0)
        return hidden_states

class MultiHeadBlockCNN(nn.Module):  # Multihead Attention Fusion Module for GCN
    def __init__(self, num_heads, input_size):
        super(MultiHeadBlockCNN, self).__init__()
        if input_size % num_heads != 0:
            raise ValueError(
                "The CNN input size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (input_size, num_heads))
        self.num_heasds = num_heads
        self.head_size = int(input_size / num_heads)
        self.attn_pool = []
        self.attn_conv = []

        for i in range(num_heads):
            conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
            setattr(self, 'attn_conv%i' % i, conv)  # 将该层添加到这个Module中，setattr函数用来设置属性，其中第一个参数为继承的类别，第二个为名称，第三个是数值
            self.attn_conv.append(conv)

    def cut_to_heads(self, x):
        # x.shape [b, c, h, w] -> [n, b, c/n, h, w]
        (b, c, h, w) = x.shape
        x = x.reshape([b, self.num_heasds, self.head_size, h, w])
        return x.permute(1, 0, 2, 3, 4)

    def forward(self, input_tensor):
        multi_tensor = self.cut_to_heads(input_tensor)
        output_list = []
        for i in range(self.num_heasds):
            avg_out = torch.mean(multi_tensor[i], dim=1, keepdim=True)
            x = self.attn_conv[i](avg_out)
            out = torch.sigmoid(x).mul(multi_tensor[i])
            output_list.append(out)
        output_tensor = torch.cat(output_list, dim=1)
        return output_tensor

class GCNConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(GCNConvBlock, self).__init__()
        self.conv1 = GraphConvolution(ch_in, ch_out)
        self.conv2 = GraphConvolution(ch_out, ch_out)
        self.drop_prob = 0.3
        self.drop = nn.Dropout(self.drop_prob)
        self.act = nn.LeakyReLU()

    def forward(self, x, adj):
        x = self.drop(self.act(self.conv1(x, adj)))
        x = self.drop(self.act(self.conv2(x, adj)))
        return x

class CNNConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k, h, w):
        super(CNNConvBlock, self).__init__()
        self.BN = nn.BatchNorm2d(ch_in)
        self.conv_in = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv_out = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k//2, stride=1, groups=ch_out)
        self.pool = nn.AvgPool2d(3, padding=1, stride=1)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.BN(x)
        x = self.act(self.conv_in(x))
        x = self.pool(x)
        x = self.act(self.conv_out(x))

        return x

class Net(nn.Module):
    def __init__(self,
                 height: int,
                 width: int,
                 channel: int,
                 class_count: int,
                 GCN_nhid: int,
                 CNN_nhid: int,
                 Q: torch.Tensor,
                 nnodes: int,
                 bands: int):
        super(Net, self).__init__()

        self.class_count = class_count 
        self.channel = channel
        self.height = height
        self.width = width
        self.GCN_nhid = GCN_nhid
        self.CNN_nhid = CNN_nhid
        self.Q = Q
        self.nnodes = nnodes
        self.bands = bands

        # GCN Conv
        self.feat_in = nn.Linear(channel, GCN_nhid)
        self.BN_GCN = nn.BatchNorm1d(GCN_nhid)
        self.BN_GCN1 = nn.BatchNorm1d(GCN_nhid)
        self.BN_GCN2 = nn.BatchNorm1d(GCN_nhid)
        self.BN_GCN3 = nn.BatchNorm1d(GCN_nhid)

        self.GCNBlock1 = GCNConvBlock(GCN_nhid, GCN_nhid)
        self.GCNBlock2 = GCNConvBlock(GCN_nhid, GCN_nhid)
        self.GCNBlock3 = GCNConvBlock(GCN_nhid, GCN_nhid)

        # CNN Conv
        self.CNNlayerA1 = CNNConvBlock(bands, CNN_nhid, 7, self.height, self.width)
        self.CNNlayerA2 = CNNConvBlock(CNN_nhid, CNN_nhid, 7, self.height, self.width)
        self.CNNlayerA3 = CNNConvBlock(CNN_nhid, CNN_nhid, 7, self.height, self.width)

        self.CNNlayerB1 = CNNConvBlock(bands, CNN_nhid, 5, self.height, self.width)
        self.CNNlayerB2 = CNNConvBlock(CNN_nhid, CNN_nhid, 5, self.height, self.width)
        self.CNNlayerB3 = CNNConvBlock(CNN_nhid, CNN_nhid, 5, self.height, self.width)

        self.CNNlayerC1 = CNNConvBlock(bands, CNN_nhid, 3, self.height, self.width)
        self.CNNlayerC2 = CNNConvBlock(CNN_nhid, CNN_nhid, 3, self.height, self.width)
        self.CNNlayerC3 = CNNConvBlock(CNN_nhid, CNN_nhid, 3, self.height, self.width)

        # Attn and Cross
        GCN_nhead = 6
        CNN_nhead = 6
        self.GCN_hidden_size = 3*GCN_nhid
        self.CNN_hidden_size = 3*CNN_nhid

        self.GCN_Multihead = MultiHeadBlockGCN(GCN_nhead, self.GCN_hidden_size)
        self.CNN_Multihead = MultiHeadBlockCNN(CNN_nhead, self.CNN_hidden_size)
        self.cross = CAFM()

        # Out
        self.GCN_fc_out = nn.Linear(self.GCN_hidden_size, class_count)
        self.CNN_fc_out = nn.Linear(self.CNN_hidden_size, class_count)
        self.Cross_fc_out = nn.Linear(self.GCN_hidden_size+self.CNN_hidden_size, class_count)

    def forward(self,
                feat: torch.Tensor,
                adj1: torch.Tensor,
                adj2: torch.Tensor,
                adj3: torch.Tensor,
                data: torch.Tensor):

        # GCN Conv, MGCsN
        feat = self.feat_in(feat)
        feat = self.BN_GCN(feat)

        mid1 = self.GCNBlock1(feat, adj1)
        mid1 = mid1 + feat
        mid1 = self.BN_GCN1(mid1)

        mid2 = self.GCNBlock2(mid1, adj2)
        mid2 = mid2 + mid1
        mid2 = self.BN_GCN2(mid2)

        out = self.GCNBlock3(mid2, adj3)
        out = out + mid2
        out = self.BN_GCN3(out)

        GCNout = torch.cat([mid1, mid2, out], dim=-1)

        # CNN Conv, PMCsN
        CNNin = torch.unsqueeze(data.permute([2, 0, 1]), 0)

        CNNmid1_A = self.CNNlayerA1(CNNin)
        CNNmid1_B = self.CNNlayerB1(CNNin)
        CNNmid1_C = self.CNNlayerC1(CNNin)

        CNNin = CNNmid1_A + CNNmid1_B + CNNmid1_C

        CNNmid2_A = self.CNNlayerA2(CNNin)
        CNNmid2_B = self.CNNlayerB2(CNNin)
        CNNmid2_C = self.CNNlayerC2(CNNin)

        CNNin = CNNmid2_A + CNNmid2_B + CNNmid2_C

        CNNout_A = self.CNNlayerA3(CNNin)
        CNNout_B = self.CNNlayerB3(CNNin)
        CNNout_C = self.CNNlayerC3(CNNin)

        CNNout = torch.cat([CNNout_A, CNNout_B, CNNout_C], dim=1)
        
        # GCN Attn, MAFM for GCN
        GCNout = self.GCN_Multihead(GCNout)
        GCNout = torch.matmul(self.Q, GCNout)

        # CNN Attn, MAFM for CNN
        CNNout = self.CNN_Multihead(CNNout)
        CNNout = torch.squeeze(CNNout, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])

        # Cross Attn
        CNNout = CNNout.transpose(0, 1).reshape([self.CNN_hidden_size, self.height, self.width]).unsqueeze(0)
        GCNout = GCNout.transpose(0, 1).reshape([self.GCN_hidden_size, self.height, self.width]).unsqueeze(0)

        CNNout, GCNout = self.cross(CNNout, GCNout)

        CNN_out = F.softmax(self.CNN_fc_out(CNNout), dim=-1)
        GCN_out = F.softmax(self.GCN_fc_out(GCNout), dim=-1)

        # Out
        Cross = torch.cat([GCNout, CNNout], dim=-1)
        Cross = F.softmax(self.Cross_fc_out(Cross), dim=-1)
        
        return Cross, CNN_out, GCN_out
    
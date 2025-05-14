import copy
import numpy as np
import torch
from torch import nn
from torch.nn import LayerNorm, Linear, Dropout, Softmax
import torch.nn.functional as F
import math


class Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, img_size, in_channels=64):
        super(Embeddings, self).__init__()

        n_patches = img_size * img_size
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3,
                                          stride=1, padding=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, 64))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))
        self.dropout = Dropout(0.5)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class Mlp(nn.Module):
    def __init__(self, encoder_dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(encoder_dim, encoder_dim)
        self.act_fn = nn.GELU()
        self.dropout1 = Dropout(0.5)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout1(x)
        return x


class Attentionbase(nn.Module):
    def __init__(self, encoder_dim):
        super(Attentionbase, self).__init__()
        self.num_attention_heads = 4
        self.attention_head_size = int(encoder_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(encoder_dim, self.all_head_size)
        self.key = Linear(encoder_dim, self.all_head_size)
        self.value = Linear(encoder_dim, self.all_head_size)

        self.out = Linear(encoder_dim, encoder_dim)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Block(nn.Module):
    def __init__(self, encoder_dim):
        super(Block, self).__init__()
        self.attention_norm = LayerNorm(encoder_dim, eps=1e-6)
        self.ffn_norm = LayerNorm(encoder_dim, eps=1e-6)
        self.ffn = Mlp(encoder_dim)
        self.attn = Attentionbase(encoder_dim)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class RelativeCoordPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape

        mask = torch.sum(x, dim=1)
        size = H

        mask = mask.view(N, H * W)
        thresholds = torch.mean(mask, dim=1, keepdim=True)
        binary_mask = (mask > thresholds).float()
        binary_mask = binary_mask.view(N, H, W)

        masked_x = x * binary_mask.view(N, 1, H, W)
        masked_x = masked_x.view(N, C, H * W).transpose(1, 2).contiguous()  # (N, S, C)
        _, reduced_x_max_index = torch.max(torch.mean(masked_x, dim=-1), dim=-1)

        basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda().type(torch.long)

        basic_label = torch.from_numpy(self.build_basic_label(size)).float()
        # Build Label
        label = basic_label.cuda()
        label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H * W, 2).type(torch.long)  # (N, S, 2)
        basic_anchor = label[basic_index, reduced_x_max_index, :].unsqueeze(1)  # (N, 1, 2)
        relative_coord = label - basic_anchor
        relative_coord = relative_coord / size
        relative_dist = torch.sqrt(torch.sum(relative_coord ** 2, dim=-1))  # (N, S)
        relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)
        relative_angle = (relative_angle / np.pi + 1) / 2  # (N, S) in (0, 1)

        binary_relative_mask = binary_mask.view(N, H * W)
        relative_dist = relative_dist * binary_relative_mask
        relative_angle = relative_angle * binary_relative_mask

        basic_anchor = basic_anchor.squeeze(1)  # (N, 2)

        relative_coord_total = torch.cat((relative_dist.unsqueeze(2), relative_angle.unsqueeze(2)), dim=-1)

        position_weight = torch.mean(masked_x, dim=-1)
        position_weight = position_weight.unsqueeze(2)
        position_weight = torch.matmul(position_weight, position_weight.transpose(1, 2))

        return relative_coord_total, basic_anchor, position_weight

    def build_basic_label(self, size):
        basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        return basic_label


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dropout=0.1):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        weight = self.weight.float()
        support = torch.matmul(input, weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return self.dropout(self.relu(output + self.bias))
        else:
            return self.dropout(self.relu(output))


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        return x


class Part_Structure(nn.Module):
    def __init__(self, encoder_dim):
        super(Part_Structure, self).__init__()
        self.act_fn = nn.ReLU()
        self.dropout = Dropout(0.1)
        self.relative_coord_predictor = RelativeCoordPredictor()
        self.gcn = GCN(2, 32, encoder_dim, dropout=0.2)

    def forward(self, hidden_states, attention_map):
        B, C, H, W = attention_map.shape
        structure_info, basic_anchor, position_weight = self.relative_coord_predictor(
            attention_map)
        structure_info = self.gcn(structure_info, position_weight)

        for i in range(B):
            index = int(basic_anchor[i, 0] * H + basic_anchor[i, 1])
            hidden_states[i, 0] = hidden_states[i, 0] + structure_info[i, index, :]

        return hidden_states


class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        last_map = x[0]
        last_map = last_map[:, :, 0, 1:]
        B, C = last_map.size(0), last_map.size(1)
        patch_num = last_map.size(-1)
        H = patch_num ** 0.5
        H = int(H)
        attention_map = last_map.view(B, C, H, H)

        return attention_map


class Encoder(nn.Module):
    def __init__(self, encoder_dim, depth):
        super(Encoder, self).__init__()
        self.attention_norm = LayerNorm(encoder_dim, eps=1e-6)
        self.layer = nn.ModuleList()
        for _ in range(depth):
            layer = Block(encoder_dim)
            self.layer.append(copy.deepcopy(layer))
        self.part_select = Part_Attention()
        self.part_layer = Block(encoder_dim)
        self.part_norm = LayerNorm(encoder_dim, eps=1e-6)
        self.part_structure = Part_Structure(encoder_dim)

    def forward(self, hidden_states):
        for i, layer in enumerate(self.layer):
            hidden_states, weights = layer(hidden_states)

            temp_weight = []
            temp_weight.append(weights)
            a_map = self.part_select(temp_weight)
            hidden_states = self.part_structure(hidden_states, a_map)
            hidden_states = self.part_norm(hidden_states)

        return hidden_states


class BaseFeatureExtraction(nn.Module):
    def __init__(self, encoder_dim, depth):
        super(BaseFeatureExtraction, self).__init__()
        self.encoder = Encoder(encoder_dim, depth)

    def forward(self, input):
        part_encoded = self.encoder(input)
        # print(hid.shape)
        return part_encoded


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, encoder_dim):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=int(encoder_dim / 2), oup=int(encoder_dim / 2), expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=int(encoder_dim / 2), oup=int(encoder_dim / 2), expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=int(encoder_dim / 2), oup=int(encoder_dim / 2), expand_ratio=2)
        self.shffleconv = nn.Conv2d(encoder_dim, encoder_dim, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, encoder_dim, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(encoder_dim) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


class CNN_Encoder(nn.Module):
    """
    多尺度编码器：逐波段提取多源数据的多尺度信息
    """
    def __init__(self, l, middle_dim):
        super(CNN_Encoder, self).__init__()

        self.conv11 = nn.Sequential(
            nn.Conv2d(l, middle_dim, 3, 1, 1),
            nn.BatchNorm2d(middle_dim),
            nn.ReLU(),  # No effect on order
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(l, middle_dim, 5, 1, 2),
            nn.BatchNorm2d(middle_dim),
            nn.ReLU(),  # No effect on order
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(l, middle_dim, 7, 1, 3),
            nn.BatchNorm2d(middle_dim),
            nn.ReLU(),  # No effect on order
        )

        self.conv = nn.Conv2d(3, 1, 3, padding=1, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()

        x_add1 = self.conv11(x)
        x_add2 = self.conv12(x)
        x_add3 = self.conv13(x)

        p = x_add3.shape[2]
        dim = x_add3.shape[1]

        num1 = x_add1.shape[1] // dim
        num2 = x_add2.shape[1] // dim
        num3 = x_add3.shape[1] // dim

        x_out = torch.empty(x.shape[0], dim, p, p).cuda()
        for i in range(dim):
            x1_tmp = x_add1[:, i * num1:(i + 1) * num1, :, :]
            x2_tmp = x_add2[:, i * num2:(i + 1) * num2, :, :]
            x3_tmp = x_add3[:, i * num3:(i + 1) * num3, :, :]

            x_tmp = torch.cat((x1_tmp, x2_tmp, x3_tmp), dim=1)
            addout = x1_tmp + x2_tmp + x3_tmp
            avgout = torch.mean(x_tmp, dim=1, keepdim=True)
            maxout, _ = torch.max(x_tmp, dim=1, keepdim=True)
            x_tmp = torch.cat([addout, avgout, maxout], dim=1)
            x_tmp = self.conv(x_tmp)
            x_out[:, i:i + 1, :, :] = x_tmp

        return x_out


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print('ECAAttention x:',x.shape)
        y = x.unsqueeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        print('ECAAttention y:',y.squeeze(1).shape)
        out = x * y.squeeze(1)
        print('ECAAttention out:',out.shape)
        return out


class NCGLF(nn.Module):
    def __init__(self, num_classes, patch_size, encoder_dim, depth, c1, c2):
        super(NCGLF, self).__init__()

        self.CNN_Encoder1 = CNN_Encoder(c1, encoder_dim)
        self.CNN_Encoder2 = CNN_Encoder(c2, encoder_dim)

        self.embeddings = Embeddings(img_size=patch_size)

        self.BaseFuseLayer = BaseFeatureExtraction(encoder_dim=encoder_dim, depth=depth)
        self.BaseFuseLayer1 = BaseFeatureExtraction(encoder_dim=encoder_dim * 2, depth=1)

        self.detailFeature = DetailFeatureExtraction(encoder_dim=encoder_dim, num_layers=depth)
        self.detailFeature1 = DetailFeatureExtraction(encoder_dim=encoder_dim * 2, num_layers=1)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(encoder_dim * 2, int(encoder_dim / 2))
        self.trans_norm = nn.LayerNorm(encoder_dim * 2)
        self.trans_cls_head = nn.Linear(encoder_dim * 2, int(encoder_dim / 2)) if num_classes > 0 else nn.Identity()

        self.Classifier = nn.Sequential(
            Linear(encoder_dim, num_classes),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_classes),
        )

        self.feature_extract_1 = nn.Sequential(
            nn.Conv2d(c1, 64, 3, 1, 1),
        )
        self.feature_extract_2 = nn.Sequential(
            nn.Conv2d(c2, 64, 3, 1, 1),
        )

        self.loss_fun2 = nn.MSELoss()  # 测试浅层特征与深层特征之间的相似性

        self.eca = ECAAttention(kernel_size=3)

    def forward(self, x):
        [b, c, h, w] = x.shape
        hsi = x[:, 0:-1, :, :]
        lidar = x[:, -1, :, :][:, None, :, :]

        # Mulitscale Information Fusion
        data1 = self.CNN_Encoder1(hsi)
        data2 = self.CNN_Encoder2(lidar)

        # INN模块:Local Feature Learning
        embedding_output1 = self.embeddings(data1)
        base_feature1 = self.BaseFuseLayer(embedding_output1)

        embedding_output2 = self.embeddings(data2)
        base_feature2 = self.BaseFuseLayer(embedding_output2)

        base_feature = torch.cat([base_feature1, base_feature2], dim=2)

        basefeature = self.BaseFuseLayer1(base_feature)
        con_loss1 = self.loss_fun2(base_feature1, base_feature2)

        x_t = self.trans_norm(basefeature)
        print('x_t:',x_t.shape)
        print('x_t[:, 0]:',x_t[:, 0].shape)
        tran_cls = self.trans_cls_head(x_t[:, 0])
        print('tran_cls:',tran_cls.shape)
        # transformer: Global Feature Learning
        detail_feature1 = self.detailFeature(data1)
        detail_feature2 = self.detailFeature(data2)

        detail_feature = torch.cat([detail_feature1, detail_feature2], dim=1)

        detailfeature = self.detailFeature1(detail_feature)
        print('detail_feature:',detail_feature.shape)
        con_loss2 = self.loss_fun2(detail_feature1, detail_feature2)

        detailfeature = self.pooling(detailfeature).flatten(1)
        conv_cls = self.conv_cls_head(detailfeature)# conv classification

        # fusion: Golbal-local Feature Fusion
        final_class = torch.cat([tran_cls, conv_cls], dim=1)
        print('final_class ori:',final_class.shape)
        final_class = self.eca(final_class)

        final_class = self.Classifier(final_class)
        print('final_class:',final_class.shape)
        con_loss = con_loss2 ** 2 / (con_loss1 + 1.01)
        final_class = final_class.view(b, -1)
        print('final_out:',final_class.shape)
        return final_class, con_loss


if __name__ == '__main__':
    x1 = torch.randn(3, 40, 11, 11).cuda()
    print(x1.shape)
    b, c1, m, n = x1.shape
    x2 = torch.randn(3, 1, 11, 11).cuda()
    print(x2.shape)
    b, c2, m, n = x2.shape

    data = torch.cat((x1, x2), dim=1)  # HSI与Lidar数据叠加
    net = NCGLF(num_classes=15, patch_size=m, encoder_dim=64, depth=2, c1=c1, c2=c2).cuda()
    output, loss = net(data)
    print(output.shape)



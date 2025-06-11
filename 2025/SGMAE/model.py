import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch.nn import Linear
import numpy as np
from torch.nn import Sequential
from timm.models.layers import DropPath
import torch
import torch.nn as nn
from einops import rearrange
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class Stem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = rearrange(x, 'b d h w -> b h w d')

        return x

class DL(nn.Module):
    def __init__(self,
                 in_chans=30,
                 num_classes=23,
                 depths=[1],
                 dims=64,
                 drop_rate=0.0,
                 norm_layer=nn.LayerNorm,):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = Stem(in_chans=in_chans, embed_dim=self.embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = torch.flatten(x, 1, 2)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = x.squeeze(1)
        x = self.forward_features(x)
        x = self.head(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgae(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', num_nodes=1000):
        super(GCN_mgae, self).__init__()
        self.decoder_mask = decoder_mask

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, add_self_loops=False))

        self.dropout = dropout

        if decoder_mask == 'mask':
            self.n_emb = torch.nn.Embedding(num_nodes, out_channels)
            self.mask_lins = torch.nn.ModuleList()
            self.mask_lins.append(torch.nn.Linear(out_channels * 2, out_channels * 2))
            self.mask_lins.append(torch.nn.Linear(out_channels * 2, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # if self.decoder_mask == 'mask':
        #     for conv in self.mask_lins:
        #         conv.reset_parameters()
            # torch.nn.init.xavier_uniform_(self.n_emb.weight.data)
            # torch.nn.init.normal_(self.n_emb.weight, std=0.1)

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        return x

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def generate_emb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        x = torch.cat(xx, dim=1)
        return x


class SAGE_mgae(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask',  num_nodes=1000):
        super(SAGE_mgae, self).__init__()
        self.decoder_mask = decoder_mask
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

        if decoder_mask == 'mask':
            self.n_emb = torch.nn.Embedding(num_nodes, out_channels)
            self.mask_lins = torch.nn.ModuleList()
            self.mask_lins.append(torch.nn.Linear(out_channels * 2, out_channels * 2))
            self.mask_lins.append(torch.nn.Linear(out_channels * 2, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        return x

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgae_ablation(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_input='last', num_nodes=1000):
        super(GCN_mgae_ablation, self).__init__()
        self.decoder_input = decoder_input

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_input == 'last':
            return x
        else:
            x = torch.cat(xx, dim=-1)
            return x

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def generate_emb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        x = torch.cat(xx, dim=1)
        return x


class SAGE_mgae_ablation(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_input='last',  num_nodes=1000):
        super(SAGE_mgae_ablation, self).__init__()
        self.decoder_input = decoder_input
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_input == 'last':
            return x
        else:
            x = torch.cat(xx, dim=-1)
            return x

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgaev2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', num_nodes=1000):
        super(GCN_mgaev2, self).__init__()
        self.decoder_mask = decoder_mask

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def generate_emb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        x = torch.cat(xx, dim=1)
        return x


class SAGE_mgaev2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE_mgaev2, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class SAGE_mgaev33(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask',  num_nodes=1000):
        super(SAGE_mgaev33, self).__init__()
        self.decoder_mask = decoder_mask
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            xx.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

class GIN_mgaev2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', eps: float = 0.,  bias=True, xavier=True):
        super(GIN_mgaev2, self).__init__()
        self.decoder_mask = decoder_mask
        self.initial_eps = eps
        self.convs = torch.nn.ModuleList()
        self.act = torch.nn.ReLU()
        for i in range(num_layers - 1):
            start_dim = hidden_channels if i else in_channels
            nn = Sequential(Linear(start_dim, hidden_channels, bias=bias),
                            self.act,
                            Linear(hidden_channels, hidden_channels, bias=bias))
            # if xavier:
            #     self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
        nn = Sequential(Linear(hidden_channels, hidden_channels, bias=bias),
                        self.act,
                        Linear(hidden_channels, out_channels, bias=bias))
        # if xavier:
        #     self.weights_init(nn)
        conv = GINConv(nn)
        self.convs.append(conv)

        self.dropout = dropout

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for conv in self.convs:
            # self.weights_init(conv.nn)
            # conv.eps.data.fill_(self.initial_eps)
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GIN_mgaev33(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', eps: float = 0.,  bias=True, xavier=True):
        super(GIN_mgaev33, self).__init__()
        self.decoder_mask = decoder_mask
        self.initial_eps = eps
        self.convs = torch.nn.ModuleList()
        self.act = torch.nn.ReLU()
        for i in range(num_layers - 1):
            start_dim = hidden_channels if i else in_channels
            nn = Sequential(Linear(start_dim, hidden_channels, bias=bias),
                            self.act,
                            Linear(hidden_channels, hidden_channels, bias=bias))
            # if xavier:
            #     self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
        nn = Sequential(Linear(hidden_channels, hidden_channels, bias=bias),
                        self.act,
                        Linear(hidden_channels, out_channels, bias=bias))
        # if xavier:
        #     self.weights_init(nn)
        conv = GINConv(nn)
        self.convs.append(conv)

        self.dropout = dropout

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for conv in self.convs:
            # self.weights_init(conv.nn)
            # conv.eps.data.fill_(self.initial_eps)
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            xx.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgaev3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN_mgaev3, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgaev33(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', num_nodes=1000):
        super(GCN_mgaev33, self).__init__()
        self.decoder_mask = decoder_mask

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            xx.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def generate_emb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        x = torch.cat(xx, dim=1)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout, de_v='v1'):
        super(LPDecoder, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        if de_v == 'v1':
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels * n_layer, in_channels * n_layer))
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LPDecoderAbs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout, abs_num=1, de_v='v1'):
        super(LPDecoderAbs, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.abs_num = abs_num
        self.lins = torch.nn.ModuleList()
        if abs_num == 1:
            self.lins.append(torch.nn.Linear(in_channels * encoder_layer, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        elif abs_num == 2:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:

            self.lins.append(GCNConv(in_channels * encoder_layer, hidden_channels, cached=False, add_self_loops=False))
            for _ in range(num_layers - 2):
                self.lins.append(
                    GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
            self.lins.append(GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge, adj_t=None):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        if self.abs_num == 1:
            src_x = torch.cat(src_x, dim=-1)
            dst_x = torch.cat(dst_x, dim=-1)
            x = src_x * dst_x
        elif self.abs_num == 2:
            src_x = src_x[-1]
            dst_x = dst_x[-1]
            x = src_x * dst_x
        else:
            x = torch.cat(h, dim=-1)

        if self.abs_num == 3:
            for lin in self.lins[:-1]:
                x = lin(x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x, adj_t)
            src_x = x[edge[0]]
            dst_x = x[edge[1]]
            x = (src_x * dst_x).sum(dim=-1)
        else:
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)

        return torch.sigmoid(x)


class LPDecoder_ogb(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout):
        super(LPDecoder_ogb, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        # self.lins.append(torch.nn.Linear(in_channels * n_layer, in_channels * n_layer))
        # self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
        # self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LPDecoder_addition(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers, dropout):
        super(LPDecoder_addition, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels*2, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        # 将起始节点和终了节点的向量拼接并求和
        concatenated = [torch.cat((xi, xj), dim=1) for xi, xj in zip(x_1, x_2)]
        # 使用 torch.stack 将列表转换为张量
        concatenated = torch.stack(concatenated, dim=0)
        # 对拼接后的向量取和
        return torch.sum(concatenated, dim=0)

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class LPDecoder_average(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers, dropout):
        super(LPDecoder_average, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels*2, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        # 将起始节点和终了节点的向量拼接
        concatenated = [torch.cat((xi, xj), dim=1) for xi, xj in zip(x_1, x_2)]
        # 使用 torch.stack 将列表转换为张量
        concatenated = torch.stack(concatenated, dim=0)
        # 对拼接后的向量逐元素取平均
        return torch.mean(concatenated, dim=0)

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

import torch
import torch.nn.functional as F

class LPDecoder_maximal(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers, dropout):
        super(LPDecoder_maximal, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels*2, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        # 将起始节点和终了节点的向量拼接
        concatenated = [torch.cat((xi, xj), dim=1) for xi, xj in zip(x_1, x_2)]
        # 使用 torch.stack 将列表转换为张量
        concatenated = torch.stack(concatenated, dim=0)
        # 对拼接后的向量逐元素取最大
        return torch.max(concatenated, dim=0).values

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class LPDecoder_multiplication(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers, dropout):
        super(LPDecoder_multiplication, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels*encoder_layer, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        # 逐层处理起始节点和终了节点的向量
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]  # 获取起始节点的向量
            xj = x_2[i]  # 获取终了节点的向量
            # 先串联拼接，再逐元素相乘
            concatenated = torch.cat((xi, xj), dim=1)
            multiplied = torch.mul(xi, xj)
            bi_layer.append(multiplied)  # 将乘积加入列表
        bi_layer = torch.cat(bi_layer, dim=1)  # 将所有结果拼接为一个Tensor
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class LPDecoder_concat(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers, dropout):
        super(LPDecoder_concat, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * n_layer * 2, hidden_channels))  # 修改输入维度
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                # 将起始节点和终了节点的向量进行串联拼接
                bi_layer.append(torch.cat((xi, xj), dim=1))  # 直接串联拼接
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class LPDecoder_ogb_layer3(torch.nn.Module):
    def __init__(self, in_channels, hid1, hid2, out_channels, encoder_layer, dropout):
        super(LPDecoder_ogb_layer3, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()

        self.lins.append(torch.nn.Linear(in_channels * n_layer, hid1))
        self.lins.append(torch.nn.Linear(hid1, hid2))
        self.lins.append(torch.nn.Linear(hid2, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class LPDecoder_ogb_sage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout, v2='v2'):
        super(LPDecoder_ogb_sage, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        if v2 == 'v1':
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        elif v2 == 'v2':
            self.lins.append(torch.nn.Linear(in_channels * n_layer, in_channels * n_layer))
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels * n_layer, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class LPDecoder_ablation(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout, num_layers, decoder_input='last', decoder_type='inner'):
        super(LPDecoder_ablation, self).__init__()
        self.decoder_type = decoder_type
        if decoder_type == 'inner':
            pass
        else:
            if decoder_input == 'last':
                in_channels = in_channels
            else:
                in_channels = in_channels * num_layers
            self.lins = torch.nn.ModuleList()
            self.lins.append(torch.nn.Linear(in_channels * 2, in_channels * 2))
            self.lins.append(torch.nn.Linear(in_channels * 2, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

            self.dropout = dropout

    def reset_parameters(self):
        if self.decoder_type == 'mlp':
            for lin in self.lins:
                lin.reset_parameters()

    def decode_inner(self, x_i, x_j):
        return (x_i * x_j).sum(dim=-1)

    def decode_mlp(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def forward(self, h, edge):
        src_x = h[edge[0]]
        dst_x = h[edge[1]]
        if self.decoder_type == 'inner':
            x = self.decode_inner(src_x, dst_x)
        else:
            x = self.decode_mlp(src_x, dst_x)
        return torch.sigmoid(x)


class FeatPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(FeatPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x



class GCN_tune(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 dropout, src_layer, dst_layer):
        super(GCN_tune, self).__init__()
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        num_layers = max(self.src_layer, self.dst_layer)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        return x_hidden[self.src_layer - 1], x_hidden[self.dst_layer - 1]


class SAGE_tune(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 dropout, src_layer, dst_layer):
        super(SAGE_tune, self).__init__()

        self.src_layer = src_layer
        self.dst_layer = dst_layer
        num_layers = max(self.src_layer, self.dst_layer)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        return x_hidden[self.src_layer - 1], x_hidden[self.dst_layer - 1]


class AutoLink_l2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type):
        super(AutoLink_l2, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.num_layers = 3
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def compute_loss(self, h, atten_matrix, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, atten_matrix, train_edge)
        neg_out = self.compute_pred(h, atten_matrix, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        # if torch.isnan(loss):
        #     print('----nan occur in pos loss={}'.format(pos_loss.item()))
        #     print('----nan occur in neg loss={}'.format(neg_loss.item()))
        #     print('### pos_out is \n')
        #     print(pos_out.data.cpu().numpy())
        #     print('### neg_out is \n')
        #     print(neg_out.data.cpu().numpy())
        #     pass
        return loss

    def compute_pred(self, h, atten_matrix, train_edge):
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h[train_edge[0]], h[train_edge[1]])
        return pos_pred

class SearchGraph_l2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_nodes,
                 temperature=0.07):
        super(SearchGraph_l2, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        self.arc = torch.nn.Parameter(torch.ones(size=[num_nodes, hidden_channels], dtype=torch.float) / self.num_nodes)
        # self.trans = torch.nn.ModuleList()
        # for i in range(num_layers - 1):
        #     if i == 0:
        #         self.trans.append(Linear(in_channels, hidden_channels, bias=False))
        #     else:
        #         self.trans.append(Linear(hidden_channels, hidden_channels, bias=False))
        # self.trans.append(Linear(hidden_channels, 1, bias=False))

    # def reset_parameters(self):
    #     for conv in self.trans:
    #         conv.reset_parameters()

    def forward(self, x):
        # x with shape [batch, num_layer, dim]
        # for conv in self.trans[:-1]:
        #     x = conv(x)
        #     x = F.relu(x)
        # x = self.trans[-1](x)
        # x = torch.squeeze(self.arc, dim=2)
        x = self.arc
        arch_set = torch.softmax(x / self.temperature, dim=1)
        device = arch_set.device
        if not self.training:
            n, c = arch_set.shape
            eyes_atten = torch.eye(c)
            atten_, atten_indice = torch.max(arch_set, dim=1)
            arch_set = eyes_atten[atten_indice]
            arch_set = arch_set.to(device)

        return arch_set


class SearchGraph_rs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, arch_layers,
                 temperature=0.07):
        super(SearchGraph_rs, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.arch_layers = arch_layers

        self.search_len = num_layers**2
        self.search_space = torch.eye(self.search_len)
        self.choice_list = torch.arange(self.search_len, dtype=torch.float)

    def forward(self, x, grad=False):
        # x with shape [batch, num_layer, dim]
        n, c, d = x.shape
        device = x.device
        # rs_indice = torch.multinomial(self.choice_list, n, replacement=True)
        rs_indice = torch.randint(0, self.search_len, (n,), dtype=torch.long)
        arch_set = self.search_space[rs_indice]
        arch_set = arch_set.to(device)
        return arch_set


class SearchGraph_qa(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, arch_layers,
                 temperature=0.07):
        super(SearchGraph_qa, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.arch_layers = arch_layers

        self.search_len = sum([num_layers - i for i in range(num_layers)])
        self.search_space = torch.eye(self.search_len)
        self.choice_list = torch.arange(self.search_len, dtype=torch.float)

    def forward(self, x):
        # x with shape [batch, num_layer, dim]
        n, c, d = x.shape
        device = x.device
        # rs_indice = torch.multinomial(self.choice_list, n, replacement=True)
        rs_indice = torch.randint(0, self.search_len, (n,), dtype=torch.long)
        arch_set = self.search_space[rs_indice]
        arch_set = arch_set.to(device)
        return arch_set


class SearchGraph_l22(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 temperature=0.07):
        super(SearchGraph_l22, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers

        self.trans = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.trans.append(Linear(in_channels, hidden_channels, bias=False))
            else:
                self.trans.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.trans.append(Linear(hidden_channels, 1, bias=False))

    def reset_parameters(self):
        for conv in self.trans:
            conv.reset_parameters()

    def forward(self, x):
        # x with shape [batch, num_layer, dim]
        for conv in self.trans[:-1]:
            x = conv(x)
            x = F.relu(x)
        x = self.trans[-1](x)
        x = torch.squeeze(x, dim=2)
        arch_set = torch.softmax(x / self.temperature, dim=1)
        device = arch_set.device
        if not self.training:
            n, c = arch_set.shape
            eyes_atten = torch.eye(c)
            atten_, atten_indice = torch.max(arch_set, dim=1)
            arch_set = eyes_atten[atten_indice]
            arch_set = arch_set.to(device)

        return arch_set


class AutoLink_l3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, lin_layers=3, cat_type='multi'):
        super(AutoLink_l3, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.cat_type = cat_type
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge, True)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg, True)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg


class AutoLink_l3Seal(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, lin_layers=3, cat_type='multi'):
        super(AutoLink_l3Seal, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.cat_type = cat_type
        self.hidden_channels = hidden_channels
        arch_vector = []
        for i in range(1, num_layers + 1):
            for j in range(1, num_layers + 1):
                arch_vector.append([i, j])
        self.arch_vector = torch.from_numpy(np.array(arch_vector))

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge, True)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg, True)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_edge(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        _, index_sub = torch.max(atten_matrix, dim=1)
        subgraphs = self.arch_vector[index_sub]
        return subgraphs

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg


class AutoLink_l3scale(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, lin_layers=3, cat_type='multi'):
        super(AutoLink_l3scale, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.cat_type = cat_type
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv_layer(in_channels, hidden_channels))
                else:
                    self.convs.append(GCNConv_layer(hidden_channels, hidden_channels))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def complte_forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def compute_loss(self, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(arch_net, train_edge, True)
        neg_out = self.compute_pred(arch_net, train_edge_neg, True)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(pos_atten, train_edge)
        neg_out = self.compute_pred_arch(neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, arch_net, train_edge, grad=False):
        h = self.cross_pair(train_edge[0], train_edge[1])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_pred_arch(self, atten_matrix, train_edge):
        h = self.cross_pair(train_edge[0], train_edge[1])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, arch_net, train_edge):
        h = self.cross_pair(train_edge[0], train_edge[1])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg

class AutoLink_l3Table(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, num_node, lin_layers=2, cat_type='multi'):
        super(AutoLink_l3Table, self).__init__()
        self.gnn_type = gnn_type
        self.num_node = num_node
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.hidden_channels = hidden_channels
        self.cat_type = cat_type

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.x = torch.nn.Embedding(num_node, in_channels)
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.x.weight)

    def forward(self, adj_t):
        x = self.x.weight
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg


class AutoLink_l3TableSeal(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, num_node, lin_layers=2, cat_type='multi'):
        super(AutoLink_l3TableSeal, self).__init__()
        self.gnn_type = gnn_type
        self.num_node = num_node
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.hidden_channels = hidden_channels
        self.cat_type = cat_type
        arch_vector = []
        for i in range(1, num_layers + 1):
            for j in range(1, num_layers + 1):
                arch_vector.append([i, j])
        self.arch_vector = torch.from_numpy(np.array(arch_vector))

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.x = torch.nn.Embedding(num_node, in_channels)
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.x.weight)

    def forward(self, adj_t):
        x = self.x.weight
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_edge(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        _, index_sub = torch.max(atten_matrix, dim=1)
        subgraphs = self.arch_vector[index_sub]
        return subgraphs

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg


class AutoLink_l3Rs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type):
        super(AutoLink_l3Rs, self).__init__()

        self.gnn_type = gnn_type
        self.num_layers = 2
        self.input_layer = num_layers
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.x.weight)

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                x.append(x_i[:, i, :] * x_j[:, j, :])
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred


class SearchGraph_l31(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, cat_type='multi',
                 temperature=0.07):
        super(SearchGraph_l31, self).__init__()
        self.temperature = temperature
        self.num_layers = num_layers
        self.cat_type = cat_type
        if self.cat_type == 'multi':
            in_channels = in_channels
        else:
            in_channels = in_channels * 2
        self.trans = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.trans.append(Linear(in_channels, hidden_channels, bias=False))
            else:
                self.trans.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.trans.append(Linear(hidden_channels, 1, bias=False))

    def reset_parameters(self):
        for conv in self.trans:
            conv.reset_parameters()

    def forward(self, x, grad=False):
        # x with shape [batch, num_layer, dim]
        for conv in self.trans[:-1]:
            x = conv(x)
            x = F.relu(x)
        x = self.trans[-1](x)
        x = torch.squeeze(x, dim=2)
        arch_set = torch.softmax(x / self.temperature, dim=1)
        if not self.training:
            if grad:
                return arch_set.detach()
            else:
                device = arch_set.device
                n, c = arch_set.shape
                eyes_atten = torch.eye(c)
                atten_, atten_indice = torch.max(arch_set, dim=1)
                arch_set = eyes_atten[atten_indice]
                arch_set = arch_set.to(device)
                return arch_set
        else:
            return arch_set


class AutoLink_Seal(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, lin_layers=3, cat_type='multi'):
        super(AutoLink_Seal, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.cat_type = cat_type
        self.hidden_channels = hidden_channels
        arch_vector = []
        for i in range(1, num_layers + 1):
            for j in range(1, num_layers + 1):
                arch_vector.append([i, j])
        self.arch_vector = torch.from_numpy(np.array(arch_vector))

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge, True)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg, True)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_edge(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        _, index_sub = torch.max(atten_matrix, dim=1)
        subgraphs = self.arch_vector[index_sub]
        return subgraphs

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg

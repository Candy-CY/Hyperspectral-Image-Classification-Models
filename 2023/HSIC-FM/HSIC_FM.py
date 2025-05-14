import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Reduce


def exists(val):
    return val is not None

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )

class LocalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        local_aggr_kernel = 5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.BatchNorm2d(dim)

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias = False)

        self.rel_pos = nn.Conv2d(heads, heads, local_aggr_kernel, padding = local_aggr_kernel // 2, groups = heads)

        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        b, c, *_, h = *x.shape, self.heads

        # to values
        v = self.to_v(x)

        # split out heads
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h = h)

        # aggregate by relative positions
        out = self.rel_pos(v)

        out = rearrange(out, '(b c) h ... -> b (c h) ...', b = b)

        return self.to_out(out)

class GlobalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        shape, h = x.shape, self.heads

        x = rearrange(x, 'b c ... -> b c (...)')

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = h), (q, k, v))  # head: the number of 'q, k'

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n', h = h)

        out = self.to_out(out)

        return out.view(*shape)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mhsa_type = 'g',
        local_aggr_kernel = 5,
        dim_head = 64,
        ff_mult = 4,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            print('depth: {}, mhsa_type: {}'.format(depth, mhsa_type))
            if mhsa_type == 'l':
                attn = LocalMHRA(dim, heads = heads, dim_head = dim_head, local_aggr_kernel = local_aggr_kernel)
            elif mhsa_type == 'g':
                attn = GlobalMHRA(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv2d(dim, dim, (3, 3), padding=1),
                attn,
                FeedForward(dim, mult=ff_mult, dropout=ff_dropout),
            ]))

    def forward(self, x):
        for dpe, attn, ff in self.layers:
            x = dpe(x) + x

            attnMap = attn(x)

            x = attn(x) + x
            x = ff(x) + x

        return x, attnMap


class FullModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dims = (64, 128, 256, 512),
        depths = (3, 4, 8, 3),
        mhsa_types = ('l', 'l', 'g', 'g'),
        channels = 1,
        ff_mult = 4,
        dim_head = 64,
        ff_dropout = 0.,
        attn_dropout = 0.,
        args,
        input_size,
        bands = 103
    ):
        super().__init__()
        if args.vanillarnn:
            net = nn.RNN
        elif args.lstm:
            net = nn.LSTM
        elif args.gru:
            net = nn.GRU
        d_out = args.d

        self.recurrent = net(input_size, args.d, args.depth)
        feat_size = d_out * args.numseq
        feat_size2 = d_out * args.numseq*bands
        self.bn = nn.BatchNorm1d(feat_size2)
        feat_size3 = (bands+81)*64
        feat_size4 = (bands+81*4)*64
        feat_size5 = (bands+81*2)*64
        self.bn_Down = nn.BatchNorm1d(feat_size3)
        self.bn_Down_Spe = nn.BatchNorm1d(feat_size2)
        self.bn_Down_allstages = nn.BatchNorm1d(feat_size4)
        self.bn_Down_2stages = nn.BatchNorm1d(feat_size5)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(feat_size, num_classes)
        self.fc1 = nn.Linear(feat_size2, 512)
        self.fc_Down = nn.Linear(feat_size3, 512)
        self.fc_Down_Spe = nn.Linear(feat_size2, 512)
        self.fc_Down_allstages = nn.Linear(feat_size4, 512)
        self.fc_Down_2stages = nn.Linear(feat_size5, 512)


        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv2d(channels, init_dim, (3, 3), padding=(1, 1))
        self.to_tokens2 = nn.Conv2d(init_dim, init_dim, (3, 3), padding=(1, 1))
        self.to_tokens3 = nn.MaxPool2d((3, 3), padding=1)

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))
        mhsa_types = tuple(map(lambda t: t.lower(), mhsa_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mhsa_type) in enumerate(zip(depths, mhsa_types)):
            print('depths, mhsa_types', depths, mhsa_types)
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Transformer(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mhsa_type = mhsa_type,
                    ff_mult = ff_mult,
                    ff_dropout = ff_dropout,
                    attn_dropout = attn_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv2d(stage_dim, dims[ind + 1], (1, 1), stride = (1, 1)),
                ) if not is_last else None
            ]))


        self.to_logits = nn.Sequential(
            Reduce('b c -> b c', 'mean'),
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, num_classes)
        )
        self.to_logits1 = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
        )
        self.to_logits_outputSpa = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, x_spe, x_spa):
        x = self.to_tokens(x_spa)
        x = self.to_tokens3(x)

        for transformer, conv in self.stages:
            x, attnMap = transformer(x)
            if transformer == self.stages[0][0]:
                Down1 = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2).cpu()
            elif transformer == self.stages[1][0]:
                Down2 = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
                Down = Down2
                Down2 = Down2.reshape(Down2.shape[0] * Down2.shape[1], -1).cpu()
                Down2 = nn.Linear(in_features=Down2.shape[1], out_features=64)(Down2)
                Down2 = Down2.reshape(Down.shape[0], Down.shape[1], -1)
            elif transformer == self.stages[2][0]:
                Down3 = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
                Down = Down3
                Down3 = Down3.reshape(Down3.shape[0] * Down3.shape[1], -1).cpu()
                Down3 = nn.Linear(in_features=Down3.shape[1], out_features=64)(Down3)
                Down3 = Down3.reshape(Down.shape[0], Down.shape[1], -1)
            elif transformer == self.stages[3][0]:
                Down4 = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
                Down = Down4
                Down4 = Down4.reshape(Down4.shape[0] * Down4.shape[1], -1).cpu()
                Down4 = nn.Linear(in_features=Down4.shape[1], out_features=64)(Down4)
                Down4 = Down4.reshape(Down.shape[0], Down.shape[1], -1)

            if exists(conv):
                x = conv(x)

        ######################## feature map visualization ########################
        x_visualize, index = torch.max(x, dim=1)
        x_visualize = x_visualize.reshape(x_visualize.shape[0], 1, x_visualize.shape[1], x_visualize.shape[2])
        x_visualize = torch.nn.functional.interpolate(x_visualize, size=(27, 27), mode='bilinear', align_corners=False)
        ###############################################################################

        Down_all = torch.cat([Down1, Down2, Down3, Down4], 1)

        output_Spa = self.to_logits_outputSpa(x)

        x = self.to_logits1(x)
        x_Spa = x

        x_spe = torch.transpose(x_spe, 0, 2).contiguous()
        x_spe = torch.transpose(x_spe, 1, 2).contiguous()

        x_RNN = self.recurrent(x_spe)[0]

        x_RNN = x_RNN.permute(1, 0, 2).contiguous()

        Down2 = Down2.cuda()
        x_RNN = torch.cat([x_RNN, Down2], 1)

        x_RNN = x_RNN.view(x_RNN.size(0), -1)

        x_RNN = self.tanh(self.bn_Down(x_RNN))

        x_RNN = self.fc_Down(x_RNN)

        output_Spe = self.to_logits(x_RNN)

        x = x_RNN + x

        x = self.to_logits(x)

        return x, output_Spe, output_Spa, x_visualize

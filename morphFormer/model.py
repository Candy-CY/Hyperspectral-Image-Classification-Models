from torch import einsum
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchsummary import summary
from einops import rearrange,repeat
import math
from torch.nn import LayerNorm,Linear,Dropout,Softmax
import copy

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
FM = 16

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

def conv_orth_dist(kernel, stride = 1):
    [o_c, i_c, w, h] = kernel.shape
    assert (w == h),"Do not support rectangular kernel"
    #half = np.floor(w/2)
    assert stride<w,"Please use matrix orthgonality instead"
    new_s = stride*(w-1) + w#np.int(2*(half+np.floor(half/stride))+1)
    temp = torch.eye(new_s*new_s*i_c).reshape((new_s*new_s*i_c, i_c, new_s,new_s)).cuda()
    out = (F.conv2d(temp, kernel, stride=stride)).reshape((new_s*new_s*i_c, -1))
    Vmat = out[np.floor(new_s**2/2).astype(int)::new_s**2, :]
    temp= np.zeros((i_c, i_c*new_s**2))
    for i in range(temp.shape[0]):temp[i,np.floor(new_s**2/2).astype(int)+new_s**2*i]=1
    return torch.norm( Vmat@torch.t(out) - torch.from_numpy(temp).float().cuda() )

def orth_dist(mat, stride=None):
    mat = mat.reshape( (mat.shape[0], -1) )
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)
    return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).cuda())



class Morphology(nn.Module):
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)


    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)
        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))
        # erosion
        weight = self.weight.view(self.out_channels, -1) # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)
        if   self.type == 'erosion2d':  x = weight - x # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d': x = weight + x # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError
        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False) # (B, Cout, L)
        else:
            x = torch.logsumexp(x*self.beta, dim=2, keepdim=False) / self.beta # (B, Cout, L)
        if self.type == 'erosion2d': x = -1 * x
        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)
        return x



class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')



class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')



class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = None, bias = None,p = 64, g = 64):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,groups=g,padding = kernel_size//3, stride = stride)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1,groups=p, stride = stride)
#         self.pwc = eSEModule(in_channels,bias=True,stride = 1 )
    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class SpectralMorph(nn.Module):
    def __init__(self,FM,NC, kernel = 3):
        super(SpectralMorph, self).__init__()

        self.erosion = Erosion2d(NC, FM, kernel, soft_max=False)
        self.conv1 = nn.Conv2d(FM,FM,1,padding = 0)
        self.dilation = Dilation2d(NC, FM, kernel, soft_max=False)
        self.conv2 = nn.Conv2d(FM,FM,1,padding = 0)
    def forward(self, x):
        z1 = self.erosion(x)
        z1 = self.conv1(z1)
        z2 = self.dilation(x)
        z2 = self.conv2(z2)
        return z1 + z2

class SpatialMorph(nn.Module):
    def __init__(self,FM,NC, kernel = 3):
        super(SpatialMorph, self).__init__()
        self.erosion = Erosion2d(NC, FM, kernel, soft_max=False)
        self.conv1 = nn.Conv2d(FM,FM,3,padding = 1)
        self.dilation = Dilation2d(NC, FM, kernel, soft_max=False)
        self.conv2 = nn.Conv2d(FM,FM,3,padding = 1)
    def forward(self, x):
        z1 = self.erosion(x)
        z1 = self.conv1(z1)
        z2 = self.dilation(x)
        z2 = self.conv2(z2)
        return z1 + z2


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):
    def __init__(self, dim, blockNum = 0):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        kernels = [3,5]
        self.cls_norm = LayerNorm(dim, eps=1e-6)
        self.spec_morph = nn.Sequential(SpectralMorph(FM,FM*2,kernels[blockNum]),nn.BatchNorm2d(FM),nn.GELU())
        self.spat_morph = nn.Sequential(SpatialMorph(FM,FM*2,kernels[blockNum]),nn.BatchNorm2d(FM),nn.GELU())
        self.attn = CrossAttention(dim)
    def forward(self, x):
        ht,w = x.shape[2:]
        rest = x[:,1:]
        rest1 = rest
        rest1 = self.spec_morph(rest1)
        rest2 = rest
        rest2 = self.spat_morph(rest2)
        rest = torch.cat([rest1,rest2],dim = 1)
        x = torch.cat([x[:,0:1,:],rest],dim = 1)
        clsTok = x[:,0:1]
        h = clsTok
        clsTok= self.attn(self.attention_norm(x.reshape(x.shape[0],x.shape[1],-1))).reshape(x.shape[0],1,ht,w)
        clsTok = clsTok + h
        clsTok = self.cls_norm(clsTok.reshape(clsTok.shape[0],clsTok.shape[1],-1)).reshape(clsTok.shape)
        x = torch.cat([clsTok,x[:,1:]],dim = 1)
        return x



class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads= 8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for i in range(2):
            layer = Block(dim,i)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)
        x= x.reshape(x.shape[0],x.shape[1],-1)
        x = self.encoder_norm(x)
        return x[:,0]



class CNN(nn.Module):
    def __init__(self, FM, NC, Classes, HSIOnly):
        super(CNN, self).__init__()

        self.HSIOnly = HSIOnly
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0,1,1), stride = 1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            HetConv(8 * (NC - 8), FM*4,
                p = 1,
                g = (FM*4)//4 if (8 * (NC - 8))%FM == 0 else (FM*4)//8,
                   ),
            nn.BatchNorm2d(FM*4),
            nn.ReLU()
        )
        self.ca = CrossAttentionBlock(FM*4)
        self.out3 = nn.Linear(FM*4 , Classes)
        torch.nn.init.xavier_uniform_(self.out3.weight)
        torch.nn.init.normal_(self.out3.bias, std=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, FM*4))
        self.position_embeddings = nn.Parameter(torch.zeros(1, FM*2 + 1, FM*4))
        self.dropout = nn.Dropout(0.1)
        self.FM = FM
        self.token_wA = nn.Parameter(torch.empty(1, FM*2, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

    def forward(self, x1):
        x1 = x1.reshape(x1.shape[0],-1,patchsize,patchsize)
        x1 = x1.unsqueeze(1)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0],-1,patchsize,patchsize)
        x1 = self.conv6(x1)
        cls_tokens = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = x1.flatten(2)
        x1 = x1.transpose(-1, -2)
        wa = self.token_wA.expand(x1.shape[0],-1,-1)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x1, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)
        wv = self.token_wV.expand(x1.shape[0],-1,-1)
        VV = torch.einsum('bij,bjk->bik', x1, wv)
        T = torch.einsum('bij,bjk->bik', A, VV)
        x = torch.cat((cls_tokens, T), dim = 1) #[b,n+1,dim]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        x = x.reshape(x.shape[0],x.shape[1],int(math.sqrt(self.FM*4)),int(math.sqrt(self.FM*4)))
        x = self.ca(x)
        x = x.reshape(x.shape[0],-1)
        return x




patchsize = 11
cnn = CNN(16, 144, 15, False)
# cnn = cnn.cuda()
summary(cnn, (144,121), device ='cpu')

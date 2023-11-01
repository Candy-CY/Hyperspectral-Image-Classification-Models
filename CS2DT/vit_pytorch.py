import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from CrossViT_module import CrossAttention

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        if self.mode == 'DEN-1':
            for i in range(depth-1):
                self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2+i], 1, 0))
                
        

    def forward(self, x, mask = None):
        #print(x.shape)
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'DEN-1': 
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 0:
                    x=x.unsqueeze(3)
                    for j in range(nl):
                        x=(torch.cat([x, last_output[nl-1-j].unsqueeze(3)], dim=3))
                    x = self.skipcat[nl-1](x).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1


        return x
    

class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self,Snum_patches,Lnum_patches, mode, small_dim = 96, small_depth = 4, small_heads =3, small_dim_head = 32, small_mlp_dim = 384,
                 large_dim = 192, large_depth = 1, large_heads = 3, large_dim_head = 64, large_mlp_dim = 768,
                 cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0.):
        super().__init__()
        self.transformer_enc_small = Transformer(small_dim, small_depth, small_heads, small_dim_head, small_mlp_dim,dropout, Snum_patches, mode)
        self.transformer_enc_large = Transformer(large_dim, large_depth, large_heads, large_dim_head, large_mlp_dim,dropout, Lnum_patches, mode)
        
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = cross_attn_heads, dim_head = large_dim_head, dropout = dropout)),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, heads = cross_attn_heads, dim_head = small_dim_head, dropout = dropout)),
            ]))


    def forward(self, xs, xl,mask = None):
        xs = self.transformer_enc_small(xs,mask)
        xl = self.transformer_enc_large(xl,mask)
        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch

            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        return xs, xl








class ViT(nn.Module):
    def __init__(self, image_size, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', dim_head = 16, dropout=0., emb_dropout=0., mode='ViT',cross_attn_depth = 1, ssf_enc_depth = 0):
        super().__init__()
    
        patch_dim = image_size ** 2
        self.pos_embedding1 = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding1 = nn.Linear(patch_dim, dim)
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout1 = nn.Dropout(emb_dropout)
        self.pool1 = pool
        self.to_latent1 = nn.Identity()

        self.pos_embedding2 = nn.Parameter(torch.randn(1, patch_dim + 1, dim))
        self.patch_to_embedding2 = nn.Linear(num_patches, dim)
        
        
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout2 = nn.Dropout(emb_dropout)
        
        self.pool2 = pool
        self.to_latent2 = nn.Identity()
        if ssf_enc_depth > 0:
             self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            )
             
             self.mlp_head2 = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
                )
             
        self.mlp_head3 = nn.Sequential(
                nn.LayerNorm(dim*2),
                nn.Linear(dim*2, num_classes)
                )
        
        self.ssf_enc_depth=ssf_enc_depth
        if ssf_enc_depth > 0:
            self.ssf_transformers = nn.ModuleList([])
            for _ in range(ssf_enc_depth):
                self.ssf_transformers.append(MultiScaleTransformerEncoder(Snum_patches=num_patches, Lnum_patches=patch_dim,mode=mode,
                                                                              small_dim=dim, small_depth=depth,
                                                                              small_heads=heads, small_dim_head=dim_head,
                                                                              small_mlp_dim=mlp_dim,
                                                                              large_dim=dim, large_depth=depth,
                                                                              large_heads=heads, large_dim_head=dim_head,
                                                                              large_mlp_dim=mlp_dim,
                                                                              cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                                                                              dropout=dropout))
        
        else:
            self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)
            self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, patch_dim, mode)

        
    
        
    def forward(self, x, mask = None):
       
        x=rearrange(x,'b w h c -> b (w h) c')
        x2=x
        x1=x.transpose(-1, -2)

       
        x1 = self.patch_to_embedding1(x1) #[b,n,dim]
        b, n, _ = x1.shape
        # add position embedding
        cls_tokens1 = repeat(self.cls_token1, '() n d -> b n d', b = b) #[b,1,dim]
        x1 = torch.cat((cls_tokens1, x1), dim = 1) #[b,n+1,dim]
        x1 += self.pos_embedding1[:, :(n + 1)]
        x1 = self.dropout1(x1)


        x2 = self.patch_to_embedding2(x2) #[b,n,dim]
        b, n, _ = x2.shape
        # add position embedding
        cls_tokens2 = repeat(self.cls_token2, '() n d -> b n d', b = b) #[b,1,dim]
        x2 = torch.cat((cls_tokens2, x2), dim = 1) #[b,n+1,dim]
        x2 += self.pos_embedding2[:, :(n + 1)]
        x2 = self.dropout2(x2)

        if self.ssf_enc_depth >0:
            xs=x1
            xl=x2
            for ssf_transformer in self.ssf_transformers:
                xs, xl = ssf_transformer(xs, xl)
            xs = xs.mean(dim = 1) if self.pool1 == 'mean' else xs[:, 0]
            xl = xl.mean(dim = 1) if self.pool2 == 'mean' else xl[:, 0]


            x3=torch.cat((xs, xl), dim = 1)
            x3=self.mlp_head3(x3)
        else:
            x1 = self.transformer1(x1, mask)
            x1 = self.to_latent1(x1[:,0])
            x2 = self.transformer2(x2, mask)

            x2 = self.to_latent2(x2[:,0])
            x3=torch.cat((x1, x2), dim = 1)
            x3=self.mlp_head3(x3)

        return x3

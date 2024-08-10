# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
# from util.logging import master_print as print

from util.video_vit import Attention, Block, PatchEmbed, Linear_Block, Linear_Attention


class VisionTransformer(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        num_frames,
        t_patch_size,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        no_qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        dropout=0, #0.5
        sep_pos_embed=False,
        cls_embed=False,
        **kwargs,
    ):
        super().__init__()
        # print(locals())
        #
        self.sep_pos_embed = sep_pos_embed
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size
        self.cls_embed = cls_embed

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim), requires_grad=True
            )  # fixed or not?

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    attn_func=partial(
                        Attention,
                        input_size=self.patch_embed.input_size,
                    ),
                )
                for i in range(depth)
                # Linear_Block(
                #     embed_dim,
                #     num_heads,
                #     mlp_ratio,
                #     qkv_bias=not no_qkv_bias,
                #     qk_scale=None,
                #     norm_layer=norm_layer,
                #     drop_path=dpr[i],
                #     attn_func=partial(
                #         Linear_Attention,
                #         input_size=self.patch_embed.input_size,
                #     ),
                # )
                # for i in range(depth)
            ]
        )



        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        torch.nn.init.normal_(self.head.weight, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "pos_embed_spatial",
            "pos_embed_temporal",
            "pos_embed_class",
        }

    def forward(self, x):
        # embed patches
        # x = x[:, :-1, :, :]  # 切片处理数据维度
        # print(x.shape)

        x = torch.unsqueeze(x, dim=1)
        x = self.patch_embed(x)
        N, T, L, C = x.shape  # T: temporal; L: spatial

        x = x.view([N, T * L, C])

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            pos_embed = self.pos_embed[:, :, :]
        x = x + pos_embed
        # reshape to [N, T, L, C] or [N, T*L, C]
        requires_t_shape = (
            len(self.blocks) > 0  # support empty decoder
            and hasattr(self.blocks[0].attn, "requires_t_shape")
            and self.blocks[0].attn.requires_t_shape
        )
        if requires_t_shape:
            x = x.view([N, T, L, C])

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if requires_t_shape:
            x = x.view([N, T * L, C])

        # classifier
        x = x[:, 1:, :].mean(dim=1)  # global pool
        x = self.norm(x)
        # x = self.fc_norm(x)
        x = self.dropout(x)
        x = self.head(x)


        return x


# def vit_base_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs,
#     )
#     return model





def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=96,
        in_chans=1,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch8(**kwargs):
    model = VisionTransformer(
        img_size=96,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch8_128(**kwargs):
    model = VisionTransformer(
        img_size=128,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch8_channel10(**kwargs):
    model = VisionTransformer(
        img_size=128,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=10,
        t_patch_size=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch16_128(**kwargs):
    model = VisionTransformer(
        img_size=128,
        in_chans=1,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=8,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
def vit_large_patch8_128(**kwargs):
    model = VisionTransformer(
        img_size=128,
        in_chans=1,
        patch_size=8,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_huge_patch8_128(**kwargs):
    model = VisionTransformer(
        img_size=128,
        in_chans=1,
        patch_size=8,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch8_120(**kwargs):
    model = VisionTransformer(
        img_size=120,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
if __name__ == '__main__':
    input = torch.rand(2, 12, 128, 128)
    model = vit_base_patch8_128()
    output = model(input)
    print(output.shape)


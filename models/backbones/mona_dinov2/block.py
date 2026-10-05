# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

# In CricaVPR, MulConvAdapter is inserted into the standard transformer block for adaptation.

import logging
from typing import Callable, List, Any, Tuple, Dict

import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

import torch.nn.functional as F
from timm.models.layers import DropPath
import math

logger = logging.getLogger("dinov2")

try:
    from xformers.ops import fmha
    from xformers.ops import scaled_index_add, index_select_cat

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class DepthwiseChannelWeighting(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // reduction, 1), 1),  # 防止除零
            nn.ReLU(),
            nn.Conv2d(max(in_channels // reduction, 1), in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pooled = self.global_pool(x)
        weights = self.conv(pooled)
        return x * weights


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=4, dilation=4, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=5, dilation=5, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=6, dilation=6, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

        # modified
        # self.weighted_layer = DepthwiseChannelWeighting(in_features * 4)
        self.adjusted_layer = nn.Conv2d(3 * in_features, in_features, 1)

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        # 特征融合流程
        fused = torch.cat([conv1_x, conv2_x, conv3_x], dim=1)
        # weighted_x = self.weighted_layer(fused)
        adjusted_x = self.adjusted_layer(fused)  # 通道数调整
        x = adjusted_x + identity
        # x = (conv1_x + conv2_x + conv3_x + conv4_x) + identity

        identity = x

        x = self.projector(x)

        return identity + x


class MonaOpMask(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=9, padding=9 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=11, padding=11 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=13, padding=13 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)
        self.adjusted_layer = nn.Conv2d(3 * in_features, in_features, 1)

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        # 特征融合流程
        fused = torch.cat([conv1_x, conv2_x, conv3_x], dim=1)
        # weighted_x = self.weighted_layer(fused)
        adjusted_x = self.adjusted_layer(fused)  # 通道数调整
        x = adjusted_x + identity
        # x = (conv1_x + conv2_x + conv3_x + conv4_x) + identity

        identity = x
        x = self.projector(x)
        x = x + identity

        # for the mask part
        b, c, h, w = x.shape
        max_size = max(h, w)
        mask_x = torch.arange(w) - w // 2
        mask_y = torch.arange(h) - h // 2
        mask_X, mask_Y = torch.meshgrid(mask_x, mask_y)
        mask_X = mask_X.to(x.device).unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)  # [B, C, H, W]
        mask_Y = mask_Y.to(x.device).unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
        # 特征图方差越大，代表飞行高度越高（特征分布更多），对应的掩码的高斯方差应当越小，使得特征集中
        channel_variance = 1 / x.var(dim=(-2, -1), keepdim=True, unbiased=False)  # [B, C, 1, 1] sigma = 0.5
        denominator = 2 * channel_variance * (max_size / 2) ** 2  # [B, C, 1, 1]
        exponent = -(mask_X ** 2 + mask_Y ** 2) / denominator  # [B, C, H, W]
        mask = torch.exp(exponent)  # [B, C, H, W]

        return x * mask


class MonaOpRaw(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=11, padding=11 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) + identity

        identity = x

        x = self.projector(x)

        return identity + x


class GlobalContextBlock(nn.Module):
    """ 全局上下文模块，用于捕捉全局信息 """

    def __init__(self, in_channels, ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化获取全局描述子
        global_context = self.avg_pool(x)
        # 通道压缩
        global_context = self.conv1(global_context)
        global_context = self.relu(global_context)
        # 通道扩展
        global_context = self.conv2(global_context)
        # 生成空间权重
        weight = self.sigmoid(global_context)
        # 加权原始特征
        return x * weight


class MonaOpValina(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, in_features, kernel_size=3, padding=6, dilation=6, groups=in_features)
        # self.conv = GlobalContextBlock(in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        identity = x
        conv_x = self.conv(x)
        x = conv_x + identity

        identity = x
        x = self.projector(x)
        x = identity + x
        # return x

        # for the mask part
        b, c, h, w = x.shape
        max_size = max(h, w)
        mask_x = torch.arange(w) - w // 2
        mask_y = torch.arange(h) - h // 2
        mask_X, mask_Y = torch.meshgrid(mask_x, mask_y)
        mask_X = mask_X.to(x.device).unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)  # [B, C, H, W]
        mask_Y = mask_Y.to(x.device).unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
        # 特征图方差越大，代表飞行高度越高（特征分布更多），对应的掩码的高斯方差应当越小，使得特征集中
        channel_variance = 1 / x.var(dim=(-2, -1), keepdim=True, unbiased=False)  # [B, C, 1, 1] sigma = 0.5
        denominator = 2 * channel_variance * (max_size / 2) ** 2  # [B, C, 1, 1]
        exponent = -(mask_X ** 2 + mask_Y ** 2) / denominator  # [B, C, H, W]
        mask = torch.exp(exponent)  # [B, C, H, W]

        return x * mask


class Mona(nn.Module):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        self.adapter_dim = 64

        self.project1 = nn.Linear(in_dim, self.adapter_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(self.adapter_dim, in_dim)

        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = MonaOpValina(self.adapter_dim)
        # self.adapter_conv = MonaOpRaw(self.adapter_dim)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, c_x, hw_shapes=None):
        # c_x输入包含token
        x = c_x[:, 1:, :]
        clstoken = c_x[:, 0:1, :]
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        outputs = torch.cat([clstoken, identity + project2], dim=1)

        return outputs


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values=None,
            drop_path: float = 0.0,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_class: Callable[..., nn.Module] = Attention,
            ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.feat_dim = 16 * 16 + 1  # input 224*224

        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

        # input feat dim
        self.mona_adapter_attn = Mona(dim, 8)
        self.mona_adapter_ffn = Mona(dim, 8)
        self.norm_a = norm_layer(dim)
        self.ls_a = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        drop_path = 0.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            # return self.ls1(self.mona_adapter_attn(self.attn(self.norm1(x)), (16, 16)))
            # return self.ls1(self.attn(self.norm1(x)))
            # return self.ls1(self.mona_adapter_attn(self.attn(self.norm1(x)), (16, 16)))
            return self.ls1(self.attn(self.norm1(x)))
            # return self.mona_adapter_attn(self.ls1(self.attn(self.norm1(x))), (16, 16))

        def ffn_residual_func(x: Tensor) -> Tensor:
            # return self.ls2(self.mona_adapter_attn(self.mlp(self.norm2(x)), (16, 16)) + 0.2 * self.mona_adapter_ffn_sc(self.norm2(x), (16, 16)))
            # return self.ls2(self.mona_adapter(self.mlp(self.norm2(x)), (16, 16)))
            # return self.ls2(self.mlp(self.norm2(x))) + self.gamma_mlp * self.mona_adapter_ffn(self.norm2(x), (16, 16))
            return self.ls2(self.mlp(self.norm2(x)))
            # return self.mona_adapter_ffn(self.ls2(self.mlp(self.norm2(x))), (16, 16))

        if x.shape[1] <= self.feat_dim:
            main_input = x
            bypass_input = torch.ones(x.shape, device=x.device)
        elif x.shape[1] > self.feat_dim:
            main_input, bypass_input = torch.split(x, [self.feat_dim, self.feat_dim], dim=1)
        else:
            main_input = None
            bypass_input = None
        x = main_input

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        # return x

        # bypass forward
        adatper_bypass_output = self.mona_adapter_ffn(self.norm_a(x) + bypass_input, (16, 16))
        # adatper_bypass_output = self.mona_adapter_ffn(self.norm_a(x), (16, 16)) + bypass_input
        return torch.cat([x, adatper_bypass_output], dim=1)


def drop_add_residual_stochastic_depth(
        x: Tensor,
        residual_func: Callable[[Tensor], Tensor],
        sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
        x_list: List[Tensor],
        residual_func: Callable[[Tensor, Any], Tensor],
        sample_drop_ratio: float = 0.0,
        scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.drop_path(self.mona_adapter(self.attn(self.norm1(x), attn_bias=attn_bias), (16, 16)))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.drop_path(self.mona_adapter(self.mlp(self.norm2(x)), (16, 16)))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)) + self.drop_path(0.2 * self.adapter(self.norm2(x))))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError

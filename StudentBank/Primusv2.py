# Copyright (c) MIC-DKFZ and contributors.
# SPDX-License-Identifier: Apache-2.0
#
# PrimusV2-M architecture module for:
#   from dynamic_network_architectures.architectures.primusv2 import PrimusV2M
#
# Important:
#   As of the public nnU-Net Primus documentation available when this file was
#   written, official PrimusV2 trainers had not yet been released. This module is
#   a practical PrimusV2-M-style network implementation based on the public
#   Primus architecture and the paper description of PrimusV2's Minimal Residual
#   Downsampling tokenizer.
#
# It expects the rest of dynamic_network_architectures to be installed, especially:
#   dynamic_network_architectures.building_blocks.eva
#   dynamic_network_architectures.building_blocks.patch_encode_decode
#   dynamic_network_architectures.initialization.weight_init

from typing import Tuple

import torch
from einops import rearrange
from timm.layers import RotaryEmbeddingCat
from torch import nn

from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    LayerNormNd,
    PatchDecode,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


PRIMUSV2_M_EMBED_DIM = 864
PRIMUSV2_M_EVA_DEPTH = 16
PRIMUSV2_M_EVA_NUM_HEADS = 12
PRIMUSV2_M_PATCH_EMBED_SIZE = (8, 8, 8)


def _norm(num_channels: int) -> nn.Module:
    return nn.InstanceNorm3d(num_channels, eps=1e-5, affine=True)


def _act() -> nn.Module:
    return nn.LeakyReLU(negative_slope=1e-2, inplace=True)


class ConvNormAct3d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            _norm(out_channels),
            _act(),
        )


class MinimalResidualBlock3d(nn.Module):
    """A small 3D residual block for the PrimusV2-M tokenizer."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            _norm(out_channels),
        )

        skip_layers = []
        if stride != 1:
            skip_layers.append(nn.AvgPool3d(kernel_size=stride, stride=stride))
        if in_channels != out_channels:
            skip_layers.extend(
                [
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    _norm(out_channels),
                ]
            )
        self.skip = nn.Identity() if not skip_layers else nn.Sequential(*skip_layers)
        self.nonlin = _act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return self.nonlin(x)


class MinimalResidualTokenizer3d(nn.Module):
    """
    PrimusV2-style Minimal Residual Downsampling tokenizer.

    It replaces Primus' single strided 8x8x8 projection with:
      stem:        residual block, stride 1, channels 32
      downsample:  residual block, stride 2, channels 32
      downsample:  residual block, stride 2, channels 64
      downsample:  residual block, stride 2, channels 128
      projection:  1x1x1 conv to 864 embedding channels

    The three stride-2 blocks produce the same 8x spatial downsampling as the
    default Primus 8x8x8 patch embedding.
    """

    def __init__(
        self,
        input_channels: int,
        embed_dim: int = PRIMUSV2_M_EMBED_DIM,
        stem_channels: int = 32,
        stage_channels: Tuple[int, int, int] = (32, 64, 128),
    ):
        super().__init__()
        self.stem = MinimalResidualBlock3d(
            input_channels,
            stem_channels,
            stride=1,
        )
        self.down_blocks = nn.Sequential(
            MinimalResidualBlock3d(stem_channels, stage_channels[0], stride=2),
            MinimalResidualBlock3d(stage_channels[0], stage_channels[1], stride=2),
            MinimalResidualBlock3d(stage_channels[1], stage_channels[2], stride=2),
        )
        self.proj = nn.Conv3d(stage_channels[2], embed_dim, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down_blocks(x)
        return self.proj(x)


class PrimusV2M(AbstractDynamicNetworkArchitectures):
    """
    PrimusV2-M-style 3D medical image segmentation network.

    Structure:
      Minimal Residual Downsampling tokenizer -> EVA/ViT encoder -> PatchDecode.

    M configuration:
      embed_dim = 864
      EVA depth = 16
      attention heads = 12
      patch_embed_size = (8, 8, 8)

    Example:
      from dynamic_network_architectures.architectures.primusv2 import PrimusV2M

      model = PrimusV2M(
          input_channels=1,
          output_channels=2,
          patch_embed_size=(8, 8, 8),
          input_shape=(96, 96, 96),
      )
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...] = PRIMUSV2_M_PATCH_EMBED_SIZE,
        input_shape: Tuple[int, ...] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
    ):
        assert input_shape is not None, "input_shape must be provided"
        assert len(input_shape) == 3, "Currently only 3D inputs are supported"
        assert len(patch_embed_size) == 3, "patch_embed_size must be a 3-tuple"
        assert tuple(patch_embed_size) == PRIMUSV2_M_PATCH_EMBED_SIZE, (
            "This PrimusV2-M tokenizer implementation currently supports "
            "patch_embed_size=(8, 8, 8)"
        )
        assert all(
            size % patch == 0 for patch, size in zip(patch_embed_size, input_shape)
        ), "Each input dimension must be divisible by the corresponding patch size"

        super().__init__()

        self.key_to_encoder = "eva"
        self.key_to_stem = "down_projection"
        self.keys_to_in_proj = ("down_projection.proj",)
        self.key_to_lpe = "eva.pos_embed"

        self.down_projection = MinimalResidualTokenizer3d(
            input_channels=input_channels,
            embed_dim=PRIMUSV2_M_EMBED_DIM,
        )
        self.up_projection = PatchDecode(
            patch_embed_size,
            PRIMUSV2_M_EMBED_DIM,
            output_channels,
            norm=decoder_norm,
            activation=decoder_act,
        )

        ref_feat_shape = tuple(
            size // patch for size, patch in zip(input_shape, patch_embed_size)
        )
        self.eva = Eva(
            embed_dim=PRIMUSV2_M_EMBED_DIM,
            depth=PRIMUSV2_M_EVA_DEPTH,
            num_heads=PRIMUSV2_M_EVA_NUM_HEADS,
            ref_feat_shape=ref_feat_shape,
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )

        self.register_buffer("mask_token", torch.zeros(1, 1, PRIMUSV2_M_EMBED_DIM))
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_register_tokens, PRIMUSV2_M_EMBED_DIM)
            )
            nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None

        self.down_projection.apply(InitWeights_He(1e-2))
        self.up_projection.apply(InitWeights_He(1e-2))

    def restore_full_sequence(self, x, keep_indices, num_patches):
        """Restore a sequence after patch dropping by inserting mask tokens."""
        if keep_indices is None:
            return x, None

        batch_size, num_kept, channels = x.shape
        device = x.device
        num_masked = num_patches - num_kept
        mask_tokens = self.mask_token.repeat(batch_size, num_masked, 1)

        restored = torch.zeros(
            batch_size,
            num_patches,
            channels,
            device=device,
            dtype=x.dtype,
        )
        restored_mask = torch.zeros(
            batch_size,
            num_patches,
            dtype=torch.bool,
            device=device,
        )

        all_indices = torch.arange(num_patches, device=device)
        for batch_idx in range(batch_size):
            kept_pos = keep_indices[batch_idx]
            mask = torch.ones(num_patches, device=device, dtype=torch.bool)
            mask[kept_pos] = False
            masked_pos = all_indices[mask]

            restored[batch_idx, kept_pos] = x[batch_idx]
            restored[batch_idx, masked_pos] = mask_tokens[batch_idx, : len(masked_pos)]
            restored_mask[batch_idx, kept_pos] = True

        return restored, restored_mask

    def forward(self, x: torch.Tensor, ret_mask: bool = False):
        full_w, full_h, full_d = x.shape[2:]

        x = self.down_projection(x)
        batch_size, channels, w, h, d = x.shape
        num_patches = w * h * d

        x = rearrange(x, "b c w h d -> b (w h d) c")

        if self.register_tokens is not None:
            x = torch.cat(
                (self.register_tokens.expand(batch_size, -1, -1), x),
                dim=1,
            )

        x, keep_indices = self.eva(x)

        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1] :]

        restored_x, restoration_mask = self.restore_full_sequence(
            x,
            keep_indices,
            num_patches,
        )
        x = rearrange(restored_x, "b (w h d) c -> b c w h d", w=w, h=h, d=d)

        if restoration_mask is not None:
            mask = rearrange(restoration_mask, "b (w h d) -> b w h d", w=w, h=h, d=d)
            full_mask = (
                mask.repeat_interleave(full_w // w, dim=1)
                .repeat_interleave(full_h // h, dim=2)
                .repeat_interleave(full_d // d, dim=3)
            )
            full_mask = full_mask[:, None, ...]
        else:
            full_mask = None

        dec_out = self.up_projection(x)
        return (dec_out, full_mask) if ret_mask else dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError(
            "PrimusV2M feature map size accounting is not implemented"
        )


# Optional aliases for users who want to import PrimusV2 but only keep the M scale.
PrimusV2 = PrimusV2M
PrimusV2_M = PrimusV2M
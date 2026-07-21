from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, int(channel / reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.bn3(self.conv3(x))
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(self.se(out + residual), inplace=True)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.r1 = ResidualBlock(in_channels + out_channels, out_channels)
        self.r2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.r2(self.r1(torch.cat((x, skip), dim=1)))


class SpatialInteractionMix(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int) -> None:
        super().__init__()
        ratio_0_to_1 = layer_id / max(1, n_layer - 1)
        ratio_1_to_almost0 = 1.0 - layer_id / n_layer
        decay = torch.ones(n_embd)
        for index in range(n_embd):
            decay[index] = -5 + 8 * (index / max(1, n_embd - 1)) ** (
                0.7 + 1.3 * ratio_0_to_1
            )
        zigzag = torch.tensor([(index + 1) % 3 - 1 for index in range(n_embd)]) * 0.5
        self.spatial_decay = nn.Parameter(decay)
        self.spatial_first = nn.Parameter(torch.ones(n_embd) * math.log(0.3) + zigzag)
        mix = torch.arange(n_embd, dtype=torch.float32).view(1, 1, -1) / n_embd
        self.spatial_mix_k = nn.Parameter(torch.pow(mix, ratio_1_to_almost0))
        self.spatial_mix_v = nn.Parameter(
            torch.pow(mix, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
        )
        self.spatial_mix_r = nn.Parameter(torch.pow(mix, 0.5 * ratio_1_to_almost0))
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The woBrwkv model does not execute this branch.")


class SpectralMixer(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int) -> None:
        super().__init__()
        ratio = 1.0 - layer_id / n_layer
        mix = torch.arange(n_embd, dtype=torch.float32).view(1, 1, -1) / n_embd
        self.spatial_mix_k = nn.Parameter(torch.pow(mix, ratio))
        self.spatial_mix_r = nn.Parameter(torch.pow(mix, ratio))
        hidden = 4 * n_embd
        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The woBrwkv model does not execute this branch.")


class VEncBlock(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int) -> None:
        super().__init__()
        self.gamma1 = nn.Parameter(torch.ones(n_embd))
        self.gamma2 = nn.Parameter(torch.ones(n_embd))
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = SpatialInteractionMix(n_embd, n_layer, layer_id)
        self.ffn = SpectralMixer(n_embd, n_layer, layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The woBrwkv model does not execute this branch.")


class BinaryOrientatedRWKV2D(nn.Module):
    def __init__(self, n_embd: int, n_layer: int = 12, **_: object) -> None:
        super().__init__()
        self.block_forward_rwkv = VEncBlock(n_embd, n_layer, 0)
        self.block_backward_rwkv = VEncBlock(n_embd, n_layer, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The woBrwkv model does not execute this branch.")


class VEnc_4_5_woBrwkv(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        num_classes: int = 1,
        dims=(24, 48, 96, 192, 384),
        **_: object,
    ) -> None:
        super().__init__()
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3])
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=dims[4])
        self.m1 = nn.Sequential(nn.Conv2d(dims[3], 1, 1), nn.Sigmoid())
        self.m2 = nn.Sequential(nn.Conv2d(dims[2], 1, 1), nn.Sigmoid())
        self.m3 = nn.Sequential(nn.Conv2d(dims[1], 1, 1), nn.Sigmoid())
        self.m4 = nn.Sequential(nn.Conv2d(dims[0], 1, 1), nn.Sigmoid())
        self.m5 = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())
        self.output1 = nn.Conv2d(16, num_classes, 1)
        self.output2 = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)
        s1, a1 = self.s1(p5, x5), self.a1(p5, x5)
        x6 = s1 * self.m1(a1)
        s2, a2 = self.s2(x6, x4), self.a2(a1, x4)
        x7 = s2 * self.m2(a2)
        s3, a3 = self.s3(x7, x3), self.a3(a2, x3)
        x8 = s3 * self.m3(a3)
        s4, a4 = self.s4(x8, x2), self.a4(a3, x2)
        x9 = s4 * self.m4(a4)
        s5, a5 = self.s5(x9, x1), self.a5(a4, x1)
        return self.output1(s5 * self.m5(a5))


__all__ = ["VEnc_4_5_woBrwkv"]

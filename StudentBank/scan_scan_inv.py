from __future__ import annotations

import torch
from einops import rearrange


def horizontal_forward_scan(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c").contiguous()


def horizontal_forward_scan_inv(x: torch.Tensor, shape) -> torch.Tensor:
    b, c, h, w = shape
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()


def horizontal_backward_scan(x: torch.Tensor) -> torch.Tensor:
    return rearrange(torch.flip(x, dims=[-1]), "b c h w -> b (h w) c").contiguous()


def horizontal_backward_scan_inv(x: torch.Tensor, shape) -> torch.Tensor:
    b, c, h, w = shape
    y = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
    return torch.flip(y, dims=[-1])


def vertical_forward_scan(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x.transpose(-1, -2).contiguous(), "b c h w -> b (h w) c").contiguous()


def vertical_forward_scan_inv(x: torch.Tensor, shape) -> torch.Tensor:
    b, c, h, w = shape
    y = rearrange(x, "b (h w) c -> b c h w", h=w, w=h).contiguous()
    return y.transpose(-1, -2).contiguous()


def vertical_backward_scan(x: torch.Tensor) -> torch.Tensor:
    y = torch.flip(x, dims=[-2]).transpose(-1, -2).contiguous()
    return rearrange(y, "b c h w -> b (h w) c").contiguous()


def vertical_backward_scan_inv(x: torch.Tensor, shape) -> torch.Tensor:
    b, c, h, w = shape
    y = rearrange(x, "b (h w) c -> b c h w", h=w, w=h).contiguous()
    return torch.flip(y.transpose(-1, -2).contiguous(), dims=[-2])

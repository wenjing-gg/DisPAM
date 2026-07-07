from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
):
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()
    x = u.new_zeros((u.shape[0], u.shape[1], A.shape[1]), dtype=torch.float32)
    ys = []
    for i in range(u.shape[-1]):
        delta_i = delta[:, :, i].unsqueeze(-1)
        u_i = u[:, :, i].unsqueeze(-1)
        B_i = B[:, :, i].unsqueeze(1)
        C_i = C[:, :, i].unsqueeze(1)
        x = torch.exp(delta_i * A.unsqueeze(0)) * x + delta_i * u_i * B_i
        y = torch.sum(x * C_i, dim=-1)
        ys.append(y)
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D.view(1, -1, 1)
    if z is not None:
        y = y * F.silu(z)
    return y.to(dtype=u.dtype)


def _causal_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    if weight.dim() == 3:
        weight = weight.squeeze(1)
    kernel_size = int(weight.shape[-1])
    x = F.pad(x, (kernel_size - 1, 0))
    return F.conv1d(x, weight.unsqueeze(1), bias=bias, groups=weight.shape[0])


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2).contiguous()
        x = F.silu(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(x.transpose(1, 2).reshape(-1, self.d_inner))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).view(batch, seqlen, self.d_inner).transpose(1, 2).contiguous()
        B = B.view(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        C = C.view(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        A = -torch.exp(self.A_log.float())
        y = selective_scan_ref(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z.transpose(1, 2).contiguous(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return self.out_proj(y.transpose(1, 2).contiguous())


__all__ = ["Mamba", "selective_scan_ref"]

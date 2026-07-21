from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    """Mamba v1-compatible module with a pure PyTorch selective scan.

    The parameter names and shapes match mamba_ssm.Mamba, so the July
    checkpoints load strictly. The scan is chunked to bound activation memory.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        del use_fast_path
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = (
            math.ceil(self.d_model / 16) if str(dt_rank) == "auto" else int(dt_rank)
        )
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        dt_init_std = self.dt_rank**-0.5 * float(dt_scale)
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init={dt_init!r}")
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(float(dt_max)) - math.log(float(dt_min)))
            + math.log(float(dt_min))
        ).clamp(min=float(dt_init_floor))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        a = torch.arange(
            1, self.d_state + 1, dtype=torch.float32, device=device
        ).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    @staticmethod
    def _scan_chunk(
        x: torch.Tensor,
        delta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        h0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_decay = delta[..., None] * a[None, None, :, :]
        log_prefix = torch.cumsum(log_decay, dim=1)
        prefix = torch.exp(log_prefix)
        drive = delta[..., None] * b[:, :, None, :] * x[..., None]
        weighted = torch.cumsum(drive * torch.exp(-log_prefix), dim=1)
        states = prefix * (h0[:, None, :, :] + weighted)
        y = torch.einsum("bldn,bln->bld", states, c)
        return y, states[:, -1]

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        delta = F.softplus(delta.float() + self.dt_proj.bias.float())
        b = b.float()
        c = c.float()
        a = -torch.exp(self.A_log.float())
        h = x.new_zeros((x.shape[0], self.d_inner, self.d_state))
        outputs = []
        for start in range(0, x.shape[1], int(chunk_size)):
            stop = min(start + int(chunk_size), x.shape[1])
            y, h = self._scan_chunk(
                x[:, start:stop],
                delta[:, start:stop],
                a,
                b[:, start:stop],
                c[:, start:stop],
                h,
            )
            outputs.append(y)
        y = torch.cat(outputs, dim=1)
        y = y + x * self.D.float()[None, None, :]
        return y.to(dtype=input_dtype)

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("Incremental Mamba inference is not used here.")
        batch, length, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :length]).transpose(1, 2)
        z = F.silu(z.transpose(1, 2))
        projected = self.x_proj(x)
        dt_raw, b, c = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.linear(dt_raw, self.dt_proj.weight, bias=None)
        y = self._selective_scan(x, delta, b, c)
        return self.out_proj(y * z)


__all__ = ["Mamba"]

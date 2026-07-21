"""Registry for the four target-specific heterogeneous students.

Paper mapping:
  Hard-Core Dual Distillation. CNN, Transformer, Mamba, and RWKV students are
  instantiated as VISTA3D, PrimusV2, SegMambaV2, and U-RWKV, respectively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

MODEL_NAMES: Tuple[str, ...] = ("vista3d", "primusv2", "segmambav2", "urwkv")


def canonical_model_name(name: str) -> str:
    aliases = {
        "vista": "vista3d",
        "vista3d": "vista3d",
        "primus": "primusv2",
        "primusv2": "primusv2",
        "transformer": "primusv2",
        "segmamba": "segmambav2",
        "segmambav2": "segmambav2",
        "mamba": "segmambav2",
        "urwkv": "urwkv",
        "rwkv": "urwkv",
    }
    key = str(name).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported model {name!r}; expected {MODEL_NAMES}.")
    return aliases[key]


class StandardLayoutAdapter(nn.Module):
    """Adapt [B,C,H,W,D] DisPAM tensors to standard [B,C,D,H,W] models."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_hwd: torch.Tensor) -> torch.Tensor:
        image_zyx = image_hwd.permute(0, 1, 4, 2, 3).contiguous()
        output = self.model(image_zyx)
        if isinstance(output, (tuple, list)):
            output = next((value for value in output if isinstance(value, torch.Tensor)), None)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("The student did not return tensor logits.")
        return output.permute(0, 1, 3, 4, 2).contiguous()


class SliceLayoutAdapter(nn.Module):
    """Apply the U-RWKV 2D network independently to every axial slice."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_hwd: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width, depth = image_hwd.shape
        slices = image_hwd.permute(0, 4, 1, 2, 3).reshape(
            batch * depth, channels, height, width
        )
        output = self.model(slices)
        if isinstance(output, (tuple, list)):
            output = next((value for value in output if isinstance(value, torch.Tensor)), None)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("U-RWKV did not return tensor logits.")
        return output.view(batch, depth, 1, height, width).permute(
            0, 2, 3, 4, 1
        ).contiguous()


def build_model(
    name: str,
    target_label_id: int,
    in_channels: int,
    patch_size: Sequence[int],
    encoder_embed_dim: int = 48,
) -> nn.Module:
    model_name = canonical_model_name(name)
    if model_name == "vista3d":
        from .vista3d import build_single_target_seg_model

        return build_single_target_seg_model(
            target_id=int(target_label_id),
            in_channels=int(in_channels),
            encoder_embed_dim=int(encoder_embed_dim),
        )
    if model_name == "primusv2":
        from .primusv2 import PrimusV2M

        return StandardLayoutAdapter(
            PrimusV2M(
                input_channels=int(in_channels),
                output_channels=1,
                input_shape=tuple(int(value) for value in patch_size),
            )
        )
    if model_name == "segmambav2":
        from .segmambav2 import SegMamba

        base = max(8, int(encoder_embed_dim))
        return StandardLayoutAdapter(
            SegMamba(
                in_chans=int(in_channels),
                out_chans=1,
                feat_size=[base, base * 2, base * 4, base * 8],
                hidden_size=base * 16,
            )
        )
    from .urwkv import VEnc_4_5_woBrwkv

    return SliceLayoutAdapter(
        VEnc_4_5_woBrwkv(input_channel=int(in_channels), num_classes=1)
    )


def _binary_output_layers(model: nn.Module) -> Tuple[nn.Module, ...]:
    candidates = [
        module
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Conv3d))
        and int(module.out_channels) == 1
    ]
    if not candidates:
        raise RuntimeError("Could not identify the model's final binary output layer.")
    return tuple(candidates)


def forward_with_decoder_features(
    model: nn.Module,
    image_hwd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return logits and the input feature map of the final segmentation layer.

    Paper mapping:
      Consensus-Divergence Contrastive Learning. The captured tensor is the
      final decoder representation immediately before binary prediction.
    """
    captured: Dict[str, torch.Tensor] = {}

    def capture_input(
        _module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
    ) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["features"] = inputs[0]

    handles = [
        layer.register_forward_pre_hook(capture_input)
        for layer in _binary_output_layers(model)
    ]
    try:
        logits = model(image_hwd)
    finally:
        for handle in handles:
            handle.remove()
    if "features" not in captured:
        raise RuntimeError("The decoder feature hook did not receive a tensor.")
    features = captured["features"]
    if features.ndim == 5:
        if features.shape[-3:] == logits.shape[-3:]:
            features_hwd = features
        elif features.shape[-3:] == (
            logits.shape[-1],
            logits.shape[-3],
            logits.shape[-2],
        ):
            features_hwd = features.permute(0, 1, 3, 4, 2).contiguous()
        else:
            features_hwd = torch.nn.functional.interpolate(
                features,
                size=(logits.shape[-1], logits.shape[-3], logits.shape[-2]),
                mode="trilinear",
                align_corners=False,
            ).permute(0, 1, 3, 4, 2).contiguous()
    elif features.ndim == 4:
        batch, _, _, _, depth = image_hwd.shape
        if features.shape[0] != batch * depth:
            raise RuntimeError(
                "Unexpected slice-model decoder feature shape "
                f"{tuple(features.shape)} for batch={batch}, depth={depth}."
            )
        features_hwd = features.view(
            batch,
            depth,
            features.shape[1],
            features.shape[2],
            features.shape[3],
        ).permute(0, 2, 3, 4, 1).contiguous()
    else:
        raise RuntimeError(f"Unsupported decoder feature shape {tuple(features.shape)}.")
    if features_hwd.shape[-3:] != logits.shape[-3:]:
        standard = features_hwd.permute(0, 1, 4, 2, 3).contiguous()
        standard = torch.nn.functional.interpolate(
            standard,
            size=(logits.shape[-1], logits.shape[-3], logits.shape[-2]),
            mode="trilinear",
            align_corners=False,
        )
        features_hwd = standard.permute(0, 1, 3, 4, 2).contiguous()
    return logits, features_hwd


def save_model_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    model_name: str,
    metadata: Dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "model_name": canonical_model_name(model_name),
            "metadata": dict(metadata),
        },
        path,
    )


def _load_state_dict_compatible(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as original_error:
        if hasattr(model, "model") and not any(key.startswith("model.") for key in state_dict):
            model.load_state_dict(
                {f"model.{key}": value for key, value in state_dict.items()},
                strict=True,
            )
            return
        raise original_error


def load_model_checkpoint(
    path: Path,
    name: str,
    target_label_id: int,
    in_channels: int,
    patch_size: Sequence[int],
    device: torch.device,
    encoder_embed_dim: int = 48,
) -> Tuple[nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    model = build_model(
        name=name,
        target_label_id=target_label_id,
        in_channels=in_channels,
        patch_size=patch_size,
        encoder_embed_dim=encoder_embed_dim,
    )
    _load_state_dict_compatible(model, checkpoint["model_state_dict"])
    return model.to(device), dict(checkpoint)

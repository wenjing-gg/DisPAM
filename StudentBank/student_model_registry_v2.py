from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn

AVAILABLE_STUDENT_MODELS: Tuple[str, ...] = ("vista3d", "segmambav2", "primusv2", "urwkv")


def canonical_student_model_name(name: str | None) -> str:
    text = str(name or "vista3d").strip().lower()
    aliases = {
        "vista": "vista3d",
        "vista3d": "vista3d",
        "segmamba": "segmambav2",
        "segmambav2": "segmambav2",
        "mamba": "segmambav2",
        "primus": "primusv2",
        "primusv2": "primusv2",
        "transformer": "primusv2",
        "urwkv": "urwkv",
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported student model {name!r}. Expected one of {AVAILABLE_STUDENT_MODELS}."
        )
    return aliases[text]


def parse_student_model_names(raw_values: Sequence[str] | None) -> List[str]:
    names: List[str] = []
    seen = set()
    values = list(raw_values or [])
    if not values:
        values = list(AVAILABLE_STUDENT_MODELS)
    for raw in values:
        for part in str(raw).split(","):
            name = canonical_student_model_name(part)
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError("At least one student model must be specified.")
    return names


def parse_patch_size_arg(value: Sequence[int] | str | None) -> Tuple[int, int, int]:
    if value is None:
        return (96, 96, 96)
    if isinstance(value, str):
        parts = [int(v) for v in str(value).replace(",", " ").split()]
    else:
        parts = [int(v) for v in value]
    if len(parts) != 3:
        raise ValueError(f"patch_size must contain exactly 3 integers, got {value!r}.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def checkpoint_filename(student_model: str, target_name: str) -> str:
    model_name = canonical_student_model_name(student_model)
    if model_name == "vista3d":
        return f"seg_model_{target_name}_best.pth"
    return f"seg_model_{model_name}_{target_name}_best.pth"


def checkpoint_last_filename(student_model: str, target_name: str) -> str:
    model_name = canonical_student_model_name(student_model)
    if model_name == "vista3d":
        return f"seg_model_{target_name}_last.pth"
    return f"seg_model_{model_name}_{target_name}_last.pth"


def model_input_shape_from_patch_size(patch_size_zyx: Sequence[int]) -> Tuple[int, int, int]:
    z, y, x = parse_patch_size_arg(patch_size_zyx)
    return int(z), int(y), int(x)


class Standard3DLayoutAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        if input_images.ndim != 5:
            raise ValueError(
                f"input_images must have shape [B, C, H, W, D], got {tuple(input_images.shape)}."
            )
        x = input_images.permute(0, 1, 4, 2, 3).contiguous()
        output = self.model(x)
        if isinstance(output, torch.Tensor):
            logits = output
        elif isinstance(output, (tuple, list)) and output:
            logits = None
            for item in output:
                if isinstance(item, torch.Tensor):
                    logits = item
                    break
            if logits is None:
                raise RuntimeError("Wrapped model returned tuple/list without tensor logits.")
        else:
            raise RuntimeError(f"Wrapped model returned unsupported output type: {type(output)!r}")
        if logits.ndim != 5:
            raise RuntimeError(f"Wrapped model must return 5D logits, got {tuple(logits.shape)}.")
        return logits.permute(0, 1, 3, 4, 2).contiguous()


def _build_vista3d_model(
    target_id: int,
    in_channels: int,
    encoder_embed_dim: int,
) -> torch.nn.Module:
    from .VISTA3D import build_single_target_seg_model

    return build_single_target_seg_model(
        target_id=int(target_id),
        in_channels=int(in_channels),
        encoder_embed_dim=int(encoder_embed_dim),
    )


def _build_segmambav2_model(
    in_channels: int,
    encoder_embed_dim: int,
) -> torch.nn.Module:
    return Standard3DLayoutAdapter(
        _build_raw_segmambav2_model(
            in_channels=in_channels,
            encoder_embed_dim=encoder_embed_dim,
        )
    )


def _build_raw_segmambav2_model(
    in_channels: int,
    encoder_embed_dim: int,
) -> torch.nn.Module:
    _prefer_local_mamba_ssm()
    from .SegMambav2 import SegMamba

    base = int(max(8, encoder_embed_dim))
    feat_size = [base, base * 2, base * 4, base * 8]
    return SegMamba(
        in_chans=int(in_channels),
        out_chans=1,
        feat_size=feat_size,
        hidden_size=base * 16,
    )


def _prefer_local_mamba_ssm() -> None:
    studentbank_root = Path(__file__).resolve().parent
    local_pkg = studentbank_root / "mamba_ssm" / "__init__.py"
    if not local_pkg.is_file():
        return
    root_text = str(studentbank_root)
    if sys.path[:1] != [root_text]:
        sys.path = [p for p in sys.path if p != root_text]
        sys.path.insert(0, root_text)
    loaded = sys.modules.get("mamba_ssm")
    loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve() if loaded is not None else None
    if loaded_path is not None and studentbank_root not in loaded_path.parents:
        for name in list(sys.modules):
            if name == "mamba_ssm" or name.startswith("mamba_ssm."):
                del sys.modules[name]


def _build_primusv2_model(
    in_channels: int,
    patch_size: Sequence[int],
) -> torch.nn.Module:
    return Standard3DLayoutAdapter(
        _build_raw_primusv2_model(
            in_channels=in_channels,
            patch_size=patch_size,
        )
    )


def _build_raw_primusv2_model(
    in_channels: int,
    patch_size: Sequence[int],
) -> torch.nn.Module:
    from .Primusv2 import PrimusV2M

    return PrimusV2M(
        input_channels=int(in_channels),
        output_channels=1,
        input_shape=model_input_shape_from_patch_size(patch_size),
    )


class SliceWise3DLayoutAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        if input_images.ndim != 5:
            raise ValueError(
                f"input_images must have shape [B, C, H, W, D], got {tuple(input_images.shape)}."
            )
        b, c, h, w, d = input_images.shape
        x = input_images.permute(0, 4, 1, 2, 3).contiguous().view(b * d, c, h, w)
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = next((item for item in output if isinstance(item, torch.Tensor)), None)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"Wrapped model returned unsupported output type: {type(output)!r}")
        if output.ndim != 4:
            raise RuntimeError(f"Wrapped model must return 4D logits, got {tuple(output.shape)}.")
        if int(output.shape[1]) != 1:
            raise RuntimeError(f"URWKV adapter expects single-channel output, got {int(output.shape[1])}.")
        if int(output.shape[2]) != h or int(output.shape[3]) != w:
            raise RuntimeError(
                f"URWKV adapter expects spatial output {h}x{w}, got {int(output.shape[2])}x{int(output.shape[3])}."
            )
        return output.view(b, d, 1, h, w).permute(0, 2, 3, 4, 1).contiguous()


def _build_urwkv_model(
    in_channels: int,
    patch_size: Sequence[int],
) -> torch.nn.Module:
    from .URWKV import VEnc_4_5_woBrwkv

    if int(in_channels) != 1:
        raise ValueError("URWKV student currently expects single-channel input.")
    _ = model_input_shape_from_patch_size(patch_size)
    return SliceWise3DLayoutAdapter(
        VEnc_4_5_woBrwkv(
            input_channel=int(in_channels),
            num_classes=1,
        )
    )


def build_single_target_student_model(
    student_model: str,
    target_id: int,
    in_channels: int = 1,
    encoder_embed_dim: int = 48,
    patch_size: Sequence[int] = (96, 96, 96),
) -> torch.nn.Module:
    model_name = canonical_student_model_name(student_model)
    try:
        if model_name == "vista3d":
            return _build_vista3d_model(
                target_id=target_id,
                in_channels=in_channels,
                encoder_embed_dim=encoder_embed_dim,
            )
        if model_name == "segmambav2":
            return _build_segmambav2_model(
                in_channels=in_channels,
                encoder_embed_dim=encoder_embed_dim,
            )
        if model_name == "primusv2":
            return _build_primusv2_model(
                in_channels=in_channels,
                patch_size=patch_size,
            )
        if model_name == "urwkv":
            return _build_urwkv_model(
                in_channels=in_channels,
                patch_size=patch_size,
            )
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import dependencies for student_model={model_name!r}: {exc}"
        ) from exc
    raise ValueError(f"Unsupported student model: {student_model!r}")


def build_legacy_layout_student_model(
    student_model: str,
    target_id: int,
    in_channels: int = 1,
    encoder_embed_dim: int = 48,
    patch_size: Sequence[int] = (96, 96, 96),
) -> torch.nn.Module:
    model_name = canonical_student_model_name(student_model)
    if model_name == "segmambav2":
        return _build_raw_segmambav2_model(
            in_channels=in_channels,
            encoder_embed_dim=encoder_embed_dim,
        )
    if model_name == "primusv2":
        return _build_raw_primusv2_model(
            in_channels=in_channels,
            patch_size=patch_size,
        )
    return build_single_target_student_model(
        student_model=model_name,
        target_id=target_id,
        in_channels=in_channels,
        encoder_embed_dim=encoder_embed_dim,
        patch_size=patch_size,
    )


def checkpoint_needs_legacy_layout(model_name: str, state_dict: Dict[str, torch.Tensor]) -> bool:
    canonical = canonical_student_model_name(model_name)
    if canonical not in {"segmambav2", "primusv2"}:
        return False
    return not any(str(key).startswith("model.") for key in state_dict.keys())


def load_state_dict_compatible(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as original_exc:
        original_error = original_exc

    has_model_prefix = any(str(key).startswith("model.") for key in state_dict.keys())
    if hasattr(model, "model") and not has_model_prefix:
        prefixed = {f"model.{key}": value for key, value in state_dict.items()}
        try:
            model.load_state_dict(prefixed, strict=True)
            return
        except RuntimeError:
            pass

    if hasattr(model, "model") and has_model_prefix:
        stripped = {
            str(key)[6:] if str(key).startswith("model.") else str(key): value
            for key, value in state_dict.items()
        }
        try:
            model.model.load_state_dict(stripped, strict=True)
            return
        except RuntimeError:
            pass

    raise original_error


def load_single_target_student_model(
    checkpoint_path: Path,
    target_label_id: int,
    device: torch.device,
    student_model: str | None = None,
    fallback_patch_size: Sequence[int] = (96, 96, 96),
) -> tuple[torch.nn.Module, Dict[str, Any], str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = dict(checkpoint.get("args", {}))
    model_name = canonical_student_model_name(
        student_model or train_args.get("student_model", "vista3d")
    )
    state_dict = checkpoint["model_state_dict"]
    patch_size = parse_patch_size_arg(train_args.get("patch_size", fallback_patch_size))
    encoder_embed_dim = int(train_args.get("encoder_embed_dim", 48))
    legacy_layout = checkpoint_needs_legacy_layout(model_name, state_dict)
    if legacy_layout:
        model = build_legacy_layout_student_model(
            student_model=model_name,
            target_id=int(target_label_id),
            in_channels=1,
            encoder_embed_dim=encoder_embed_dim,
            patch_size=patch_size,
        ).to(device)
    else:
        model = build_single_target_student_model(
            student_model=model_name,
            target_id=int(target_label_id),
            in_channels=1,
            encoder_embed_dim=encoder_embed_dim,
            patch_size=patch_size,
        ).to(device)
    load_state_dict_compatible(model, state_dict)
    train_args["_loaded_layout"] = "legacy_hwd_raw" if legacy_layout else "standard_adapter"
    model.eval()
    return model, train_args, model_name

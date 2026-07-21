"""Array and tensor layout helpers."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def image_czyx_to_model_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert [C,D,H,W] to the model layout [C,H,W,D]."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W], got {array.shape}.")
    return torch.from_numpy(np.transpose(array, (0, 2, 3, 1)).copy())


def mask_zyx_to_model_tensor(mask: np.ndarray) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected [D,H,W], got {array.shape}.")
    return torch.from_numpy(np.transpose(array, (1, 2, 0)).copy()).unsqueeze(0)


def model_tensor_to_zyx(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError(f"Expected [H,W,D], got {array.shape}.")
    return np.transpose(array, (2, 0, 1))


def crop_with_pad(
    array: np.ndarray,
    start_zyx: Sequence[int],
    patch_size: Sequence[int],
    pad_value: float = 0.0,
) -> np.ndarray:
    source = np.asarray(array)
    start = tuple(int(value) for value in start_zyx)
    patch = tuple(int(value) for value in patch_size)
    if len(start) != 3 or len(patch) != 3:
        raise ValueError("start_zyx and patch_size must each contain three values.")
    spatial_shape = source.shape[-3:]
    output_shape = source.shape[:-3] + patch
    output = np.full(output_shape, pad_value, dtype=source.dtype)

    source_slices = []
    target_slices = []
    for axis, (origin, span, limit) in enumerate(zip(start, patch, spatial_shape)):
        source_lo = max(0, origin)
        source_hi = min(int(limit), origin + span)
        target_lo = source_lo - origin
        target_hi = target_lo + max(0, source_hi - source_lo)
        source_slices.append(slice(source_lo, source_hi))
        target_slices.append(slice(target_lo, target_hi))
    if all(item.stop > item.start for item in source_slices):
        output[(..., *target_slices)] = source[(..., *source_slices)]
    return output


def resize_zyx(array: np.ndarray, output_shape: Sequence[int], mode: str) -> np.ndarray:
    source = np.asarray(array, dtype=np.float32)
    tensor = torch.from_numpy(source).view(1, 1, *source.shape)
    align_corners = False if mode == "trilinear" else None
    resized = F.interpolate(
        tensor,
        size=tuple(int(value) for value in output_shape),
        mode=mode,
        align_corners=align_corners,
    )
    return resized[0, 0].numpy()

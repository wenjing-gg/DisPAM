"""Connected-component voting and full-volume inference.

Paper mapping:
  Consensus--Divergence Zone Determination. Each heterogeneous model is
  binarized and component-filtered independently before voxel support counting.
  Divergence uses the support rule OR the population standard-deviation rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .component_selection import label_components_6
from .tensor_ops import image_czyx_to_model_tensor, model_tensor_to_zyx


def _triple(values: Sequence[int], name: str) -> Tuple[int, int, int]:
    output = tuple(int(value) for value in values)
    if len(output) != 3 or any(value <= 0 for value in output):
        raise ValueError(f"{name} must contain three positive integers.")
    return output


def filter_small_components(
    mask: np.ndarray,
    min_component_voxels: int,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    labels, count = label_components_6(mask)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    keep = [
        component_id
        for component_id in range(1, count + 1)
        if int(sizes[component_id]) >= int(min_component_voxels)
    ]
    lookup = np.zeros(count + 1, dtype=bool)
    if keep:
        lookup[np.asarray(keep, dtype=np.int64)] = True
    rows = [
        {"component_id": int(component_id), "voxels": int(sizes[component_id])}
        for component_id in keep
    ]
    rows.sort(key=lambda row: row["voxels"], reverse=True)
    return lookup[labels], rows


@dataclass(frozen=True)
class RegionBatch:
    consensus: torch.Tensor
    divergence: torch.Tensor
    filtered_binary: torch.Tensor
    support_count: torch.Tensor
    probability_std: torch.Tensor


def build_consensus_divergence(
    probabilities: Sequence[torch.Tensor],
    probability_threshold: float,
    consensus_min_votes: int,
    probability_std_threshold: float,
    min_component_voxels: int,
) -> RegionBatch:
    """Build fixed regions from the four dual-distilled probability maps."""
    # CDFT zone determination: component support and probability dispersion
    # instantiate the paper's Consensus and Divergence region definitions.
    if len(probabilities) != 4:
        raise ValueError(f"CDFT requires four model probabilities, got {len(probabilities)}.")
    shapes = {tuple(probability.shape) for probability in probabilities}
    if len(shapes) != 1:
        raise ValueError(f"Probability shapes differ: {sorted(shapes)}")
    device = probabilities[0].device
    filtered_models: List[torch.Tensor] = []
    for probability in probabilities:
        rows = []
        binary = probability.detach().cpu().numpy() >= float(probability_threshold)
        for batch_index in range(binary.shape[0]):
            zyx = model_tensor_to_zyx(binary[batch_index, 0])
            filtered, _ = filter_small_components(zyx, min_component_voxels)
            rows.append(
                torch.from_numpy(np.transpose(filtered.astype(np.float32), (1, 2, 0)))
                .unsqueeze(0)
            )
        filtered_models.append(torch.stack(rows, dim=0).to(device=device))
    filtered_binary = torch.stack(filtered_models, dim=0)
    support_count = filtered_binary.sum(dim=0)
    probability_stack = torch.stack([value.detach() for value in probabilities], dim=0)
    probability_std = torch.std(probability_stack, dim=0, correction=0)
    component_union = support_count > 0
    low_dispersion = probability_std < float(probability_std_threshold)
    consensus = (support_count >= int(consensus_min_votes)) & low_dispersion
    divergence = (
        ((support_count > 0) & (support_count < int(consensus_min_votes)))
        | (component_union & (probability_std >= float(probability_std_threshold)))
    ) & ~consensus
    return RegionBatch(
        consensus=consensus.float(),
        divergence=divergence.float(),
        filtered_binary=filtered_binary.float(),
        support_count=support_count,
        probability_std=probability_std,
    )


def component_masks(
    region: torch.Tensor,
    min_component_voxels: int,
) -> List[List[torch.Tensor]]:
    """Return retained component masks for every batch item."""
    output: List[List[torch.Tensor]] = []
    device = region.device
    binary = region.detach().cpu().numpy() > 0.5
    for batch_index in range(binary.shape[0]):
        zyx = model_tensor_to_zyx(binary[batch_index, 0])
        labels, count = label_components_6(zyx)
        masks: List[torch.Tensor] = []
        for component_id in range(1, count + 1):
            component = labels == component_id
            if int(component.sum()) < int(min_component_voxels):
                continue
            hwd = np.transpose(component.astype(np.float32), (1, 2, 0))
            masks.append(torch.from_numpy(hwd).unsqueeze(0).to(device=device))
        output.append(masks)
    return output


def component_masks_from_labels(
    component_labels: torch.Tensor,
) -> List[List[torch.Tensor]]:
    """Recover cropped masks while retaining full-volume component identities."""
    output: List[List[torch.Tensor]] = []
    labels = component_labels.to(dtype=torch.int64)
    for batch_index in range(labels.shape[0]):
        sample = labels[batch_index, 0]
        component_ids = torch.unique(sample)
        masks = [
            (sample == component_id).unsqueeze(0).float()
            for component_id in component_ids
            if int(component_id.item()) > 0
        ]
        output.append(masks)
    return output


def sliding_window_starts(length: int, patch: int, overlap: float) -> List[int]:
    if length <= patch:
        return [0]
    stride = max(1, int(round(patch * (1.0 - float(overlap)))))
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def gaussian_importance_map(
    patch_size: Sequence[int],
    sigma_scale: float = 0.125,
    minimum: float = 1e-3,
) -> np.ndarray:
    patch = _triple(patch_size, "patch_size")
    output = np.ones(patch, dtype=np.float32)
    for axis, length in enumerate(patch):
        coordinates = np.arange(length, dtype=np.float32)
        center = (length - 1) / 2.0
        sigma = max(length * float(sigma_scale), 1e-6)
        profile = np.exp(-0.5 * ((coordinates - center) / sigma) ** 2)
        shape = [1, 1, 1]
        shape[axis] = length
        output *= profile.reshape(shape)
    output /= max(float(output.max()), 1e-12)
    return np.maximum(output, float(minimum)).astype(np.float32)


def sliding_window_probability(
    model: torch.nn.Module,
    image_czyx: np.ndarray,
    patch_size: Sequence[int],
    overlap: float,
    sw_batch_size: int,
    device: torch.device,
    amp: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Predict one binary target over a complete multi-modal volume."""
    image = np.asarray(image_czyx, dtype=np.float32)
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W], got {image.shape}.")
    patch = _triple(patch_size, "patch_size")
    spatial = image.shape[-3:]
    padded_shape = tuple(max(spatial[axis], patch[axis]) for axis in range(3))
    pad_width = ((0, 0),) + tuple(
        (0, padded_shape[axis] - spatial[axis]) for axis in range(3)
    )
    padded = np.pad(image, pad_width, mode="constant")
    starts = [
        sliding_window_starts(padded_shape[axis], patch[axis], overlap)
        for axis in range(3)
    ]
    coordinates = [
        (z, y, x)
        for z in starts[0]
        for y in starts[1]
        for x in starts[2]
    ]
    importance = gaussian_importance_map(patch)
    probability_sum = np.zeros(padded_shape, dtype=np.float32)
    weight_sum = np.zeros(padded_shape, dtype=np.float32)
    use_amp = bool(amp and device.type == "cuda")
    was_training = model.training
    model.eval()
    start_time = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(coordinates), max(1, int(sw_batch_size))):
            batch_coordinates = coordinates[offset : offset + max(1, int(sw_batch_size))]
            patches = [
                image_czyx_to_model_tensor(
                    padded[
                        :,
                        z : z + patch[0],
                        y : y + patch[1],
                        x : x + patch[2],
                    ]
                )
                for z, y, x in batch_coordinates
            ]
            inputs = torch.stack(patches).to(device=device, dtype=torch.float32)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inputs)
            probabilities = torch.sigmoid(logits).float().cpu()
            for batch_index, (z, y, x) in enumerate(batch_coordinates):
                probability = model_tensor_to_zyx(probabilities[batch_index, 0])
                target = np.s_[z : z + patch[0], y : y + patch[1], x : x + patch[2]]
                probability_sum[target] += probability * importance
                weight_sum[target] += importance
    if was_training:
        model.train()
    probability = probability_sum / np.maximum(weight_sum, 1e-8)
    cropped = probability[: spatial[0], : spatial[1], : spatial[2]]
    return cropped.astype(np.float32), {
        "window_count": len(coordinates),
        "seconds": float(time.perf_counter() - start_time),
        "patch_size": list(patch),
        "overlap": float(overlap),
    }

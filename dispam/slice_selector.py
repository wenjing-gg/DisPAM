"""Anatomy-Agent Adaptive Slice Selector.

Paper mapping:
  Propagation-Aware Slice Estimation ranks slices using anatomical support,
  transition-balanced interior position, and local boundary stability. Agent
  Selection then applies target-specific visibility screening in that order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


@dataclass(frozen=True)
class SliceRanking:
    order: Tuple[int, ...]
    score: np.ndarray
    support: np.ndarray
    interior: np.ndarray
    boundary_stability: np.ndarray


def _unit_interval(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    low = float(array.min())
    high = float(array.max())
    if high - low <= 1e-8:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def _robust_modality_mean(image_czyx: np.ndarray) -> np.ndarray:
    image = np.asarray(image_czyx, dtype=np.float32)
    if image.ndim == 3:
        image = image[None]
    normalized = []
    for channel in image:
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            normalized.append(np.zeros_like(channel))
            continue
        low, high = np.percentile(finite, [0.5, 99.5])
        normalized.append(_unit_interval(np.clip(channel, low, high)))
    return np.mean(np.stack(normalized, axis=0), axis=0)


def propagation_aware_slice_estimation(
    image_czyx: np.ndarray,
    smoothing_sigma: float = 1.0,
) -> SliceRanking:
    """Compute the four equations in Propagation-Aware Slice Estimation."""
    volume = _robust_modality_mean(image_czyx)
    occupancy = np.zeros(volume.shape[0], dtype=np.float32)
    for index, image_slice in enumerate(volume):
        if float(image_slice.max() - image_slice.min()) <= 1e-8:
            continue
        threshold = float(threshold_otsu(image_slice))
        foreground = image_slice > threshold
        occupancy[index] = float(np.mean(foreground))
    support = _unit_interval(
        ndi.gaussian_filter1d(occupancy, sigma=float(smoothing_sigma), mode="nearest")
    )

    transition = np.zeros(volume.shape[0], dtype=np.float32)
    if volume.shape[0] > 1:
        adjacent = np.mean(np.abs(np.diff(volume, axis=0)), axis=(1, 2))
        transition[1:] = adjacent
        transition[0] = transition[1]
    transition = _unit_interval(
        ndi.gaussian_filter1d(transition, sigma=float(smoothing_sigma), mode="nearest")
    )
    # PASE equations: support=S(z), interior=C(z), and
    # boundary_stability=B(z); score is Omega(z)=S(z)C(z)B(z).
    mass = transition / max(float(transition.sum()), 1e-8)
    position = np.cumsum(mass) - mass / 2.0
    interior = np.sqrt(np.maximum(np.sin(np.pi * np.clip(position, 0.0, 1.0)), 0.0))
    boundary_stability = np.sqrt(np.clip(1.0 - transition, 0.0, 1.0))
    score = support * interior * boundary_stability
    order = tuple(int(index) for index in np.argsort(-score, kind="stable"))
    return SliceRanking(
        order=order,
        score=score,
        support=support,
        interior=interior,
        boundary_stability=boundary_stability,
    )

"""PAM Cache serialization and foreground-aware training patches.

Paper mapping:
  Multi-Perturbation PAM Propagation and Hard-Core Dual Distillation. Each cache
  stores the Normal hard target, the confidence-filtered Dilation/Erosion Core
  region, and the corresponding averaged soft probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import CaseRecord
from .io import load_case_image, robust_normalize_channels
from .tensor_ops import crop_with_pad, image_czyx_to_model_tensor, mask_zyx_to_model_tensor


@dataclass(frozen=True)
class CacheFiles:
    npz: Path
    metadata: Path


@dataclass(frozen=True)
class RegionFiles:
    npz: Path
    metadata: Path


def cache_files(root: Path, split: str, target: str, case_id: str) -> CacheFiles:
    directory = Path(root) / str(split) / str(target)
    return CacheFiles(directory / f"{case_id}.npz", directory / f"{case_id}.json")


def region_files(root: Path, split: str, target: str, case_id: str) -> RegionFiles:
    directory = Path(root) / str(split) / str(target)
    return RegionFiles(directory / f"{case_id}.npz", directory / f"{case_id}.json")


def save_cache(
    files: CacheFiles,
    hard_target: np.ndarray,
    core_region: np.ndarray,
    core_probability: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    files.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        files.npz,
        hard_target=np.asarray(hard_target, dtype=np.uint8),
        core_region=np.asarray(core_region, dtype=np.uint8),
        core_probability=np.asarray(core_probability, dtype=np.float16),
    )
    files.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_cache(files: CacheFiles) -> Dict[str, np.ndarray]:
    if not files.npz.exists():
        raise FileNotFoundError(f"Missing PAM Cache: {files.npz}")
    with np.load(files.npz) as payload:
        required = {"hard_target", "core_region", "core_probability"}
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(f"{files.npz} is missing keys {sorted(missing)}.")
        return {key: np.asarray(payload[key]) for key in required}


def save_regions(
    files: RegionFiles,
    consensus_labels: np.ndarray,
    divergence_labels: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    files.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        files.npz,
        consensus_labels=np.asarray(consensus_labels, dtype=np.int32),
        divergence_labels=np.asarray(divergence_labels, dtype=np.int32),
    )
    files.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_regions(files: RegionFiles) -> Dict[str, np.ndarray]:
    if not files.npz.exists():
        raise FileNotFoundError(f"Missing CDFT region cache: {files.npz}")
    with np.load(files.npz) as payload:
        required = {"consensus_labels", "divergence_labels"}
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(f"{files.npz} is missing keys {sorted(missing)}.")
        return {key: np.asarray(payload[key]) for key in required}


def sample_patch_start(
    shape_zyx: Sequence[int],
    patch_size: Sequence[int],
    hard_target: np.ndarray,
    core_region: np.ndarray,
    rng: np.random.Generator,
    foreground_probability: float,
    core_probability: float,
) -> Tuple[int, int, int]:
    shape = np.asarray(shape_zyx, dtype=np.int64)
    patch = np.asarray(patch_size, dtype=np.int64)

    def random_start() -> np.ndarray:
        return np.asarray(
            [
                int(rng.integers(0, max(1, int(shape[axis] - patch[axis] + 1))))
                for axis in range(3)
            ],
            dtype=np.int64,
        )

    use_foreground = bool(np.any(hard_target)) and rng.random() < float(
        foreground_probability
    )
    if not use_foreground:
        return tuple(int(value) for value in random_start())
    use_core = bool(np.any(core_region)) and rng.random() < float(core_probability)
    coordinates = np.argwhere(core_region if use_core else hard_target)
    center = coordinates[int(rng.integers(0, coordinates.shape[0]))]
    start = center - patch // 2
    start = np.maximum(start, 0)
    start = np.minimum(start, np.maximum(shape - patch, 0))
    return tuple(int(value) for value in start)


class DistillationPatchDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[CaseRecord],
        cache_root: Path,
        split: str,
        target: str,
        patch_size: Sequence[int],
        samples_per_case: int,
        foreground_probability: float,
        core_probability: float,
        seed: int,
    ) -> None:
        self.items: List[Tuple[CaseRecord, CacheFiles]] = []
        for case in cases:
            files = cache_files(cache_root, split, target, case.case_id)
            if files.npz.exists():
                self.items.append((case, files))
        if not self.items:
            raise RuntimeError(f"No PAM Cache entries found for {split}/{target}.")
        self.patch_size = tuple(int(value) for value in patch_size)
        self.samples_per_case = max(1, int(samples_per_case))
        self.foreground_probability = float(foreground_probability)
        self.core_probability = float(core_probability)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.items) * self.samples_per_case

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case, files = self.items[index % len(self.items)]
        worker = torch.utils.data.get_worker_info()
        worker_offset = 0 if worker is None else int(worker.id) * 1_000_003
        rng = np.random.default_rng(self.seed + worker_offset + int(index))
        image = robust_normalize_channels(load_case_image(case.image_paths))
        cache = load_cache(files)
        hard_target = cache["hard_target"] > 0
        core_region = cache["core_region"] > 0
        core_probability = np.asarray(cache["core_probability"], dtype=np.float32)
        start = sample_patch_start(
            image.shape[-3:],
            self.patch_size,
            hard_target,
            core_region,
            rng,
            self.foreground_probability,
            self.core_probability,
        )
        return {
            "case_id": case.case_id,
            "image": image_czyx_to_model_tensor(
                crop_with_pad(image, start, self.patch_size)
            ),
            "hard_target": mask_zyx_to_model_tensor(
                crop_with_pad(hard_target, start, self.patch_size)
            ),
            "core_region": mask_zyx_to_model_tensor(
                crop_with_pad(core_region, start, self.patch_size)
            ),
            "core_probability": mask_zyx_to_model_tensor(
                crop_with_pad(core_probability, start, self.patch_size)
            ),
            "start_zyx": torch.tensor(start, dtype=torch.int64),
        }


class CDFTPatchDataset(Dataset):
    """Add fixed full-volume Consensus and Divergence labels to PAM patches."""

    def __init__(
        self,
        cases: Sequence[CaseRecord],
        cache_root: Path,
        region_root: Path,
        split: str,
        target: str,
        patch_size: Sequence[int],
        samples_per_case: int,
        foreground_probability: float,
        core_probability: float,
        seed: int,
        data_protocol: Dict[str, object] | None = None,
    ) -> None:
        self.base = DistillationPatchDataset(
            cases,
            cache_root,
            split,
            target,
            patch_size,
            samples_per_case,
            foreground_probability,
            core_probability,
            seed,
        )
        self.region_lookup = {
            case.case_id: region_files(region_root, split, target, case.case_id)
            for case, _ in self.base.items
        }
        missing = [
            case_id
            for case_id, files in self.region_lookup.items()
            if not files.npz.exists()
        ]
        if missing:
            raise RuntimeError(
                f"Missing fixed CDFT regions for {len(missing)} cases: {missing[:10]}"
            )
        if data_protocol is not None:
            for files in self.region_lookup.values():
                metadata = json.loads(files.metadata.read_text(encoding="utf-8"))
                if metadata.get("data_protocol") != data_protocol:
                    raise ValueError(
                        "CDFT regions use a different or unrecorded data split; "
                        "rebuild them from checkpoints trained with this protocol."
                    )
        self.patch_size = tuple(int(value) for value in patch_size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case, files = self.base.items[index % len(self.base.items)]
        worker = torch.utils.data.get_worker_info()
        worker_offset = 0 if worker is None else int(worker.id) * 1_000_003
        rng = np.random.default_rng(self.base.seed + worker_offset + int(index))
        image = robust_normalize_channels(load_case_image(case.image_paths))
        cache = load_cache(files)
        regions = load_regions(self.region_lookup[case.case_id])
        hard_target = cache["hard_target"] > 0
        core_region = cache["core_region"] > 0
        core_probability = np.asarray(cache["core_probability"], dtype=np.float32)
        consensus_labels = regions["consensus_labels"].astype(np.int32, copy=False)
        divergence_labels = regions["divergence_labels"].astype(np.int32, copy=False)
        cdft_support = (consensus_labels > 0) | (divergence_labels > 0)
        start = sample_patch_start(
            image.shape[-3:],
            self.patch_size,
            hard_target | cdft_support,
            core_region | (consensus_labels > 0),
            rng,
            self.base.foreground_probability,
            self.base.core_probability,
        )
        return {
            "case_id": case.case_id,
            "image": image_czyx_to_model_tensor(
                crop_with_pad(image, start, self.patch_size)
            ),
            "hard_target": mask_zyx_to_model_tensor(
                crop_with_pad(hard_target, start, self.patch_size)
            ),
            "core_region": mask_zyx_to_model_tensor(
                crop_with_pad(core_region, start, self.patch_size)
            ),
            "core_probability": mask_zyx_to_model_tensor(
                crop_with_pad(core_probability, start, self.patch_size)
            ),
            "consensus_labels": mask_zyx_to_model_tensor(
                crop_with_pad(consensus_labels, start, self.patch_size)
            ).to(dtype=torch.int64),
            "divergence_labels": mask_zyx_to_model_tensor(
                crop_with_pad(divergence_labels, start, self.patch_size)
            ).to(dtype=torch.int64),
            "start_zyx": torch.tensor(start, dtype=torch.int64),
        }

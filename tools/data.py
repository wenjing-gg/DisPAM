from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tools.volumes import load_volume_any


def resize_zyx(arr: np.ndarray, size_zyx: Sequence[int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))[None, None]
    size = tuple(int(v) for v in size_zyx)
    if mode == "nearest":
        out = F.interpolate(tensor, size=size, mode="nearest")
    else:
        out = F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)
    return np.asarray(out[0, 0].numpy(), dtype=np.float32)


def zyx_to_model_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(np.asarray(arr, dtype=np.float32), (1, 2, 0))).unsqueeze(0)


def model_hwd_to_zyx(arr_hwd: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(arr_hwd, dtype=np.float32), (2, 0, 1))


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return np.zeros_like(arr, dtype=np.float32)
    values = arr[finite]
    lo, hi = np.percentile(values, [0.5, 99.5])
    if float(hi) <= float(lo):
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~finite] = 0.0
    return out


def crop_or_pad_zyx(arr: np.ndarray, start_zyx: Sequence[int], patch_size: Sequence[int], pad_value: float = 0.0) -> np.ndarray:
    source = np.asarray(arr)
    start = np.asarray([int(v) for v in start_zyx], dtype=np.int64)
    size = np.asarray([int(v) for v in patch_size], dtype=np.int64)
    out = np.full(tuple(int(v) for v in size), pad_value, dtype=source.dtype)
    src_slices = []
    dst_slices = []
    for axis in range(3):
        src0 = max(int(start[axis]), 0)
        src1 = min(int(start[axis] + size[axis]), int(source.shape[axis]))
        dst0 = max(-int(start[axis]), 0)
        dst1 = dst0 + max(src1 - src0, 0)
        src_slices.append(slice(src0, src1))
        dst_slices.append(slice(dst0, dst1))
    if all(s.stop > s.start for s in src_slices):
        out[tuple(dst_slices)] = source[tuple(src_slices)]
    return out


def choose_foreground_aware_start(
    foreground: np.ndarray,
    volume_shape: Sequence[int],
    patch_size: Sequence[int],
    rng: np.random.Generator,
    foreground_prob: float,
) -> Tuple[int, int, int]:
    shape = np.asarray([int(v) for v in volume_shape], dtype=np.int64)
    patch = np.asarray([int(v) for v in patch_size], dtype=np.int64)
    max_start = np.maximum(shape - patch, 0)
    use_foreground = bool(np.any(foreground)) and float(rng.random()) < float(foreground_prob)
    if use_foreground:
        coords = np.argwhere(np.asarray(foreground, dtype=bool))
        center = coords[int(rng.integers(0, len(coords)))]
        jitter = rng.integers(-np.maximum(patch // 4, 1), np.maximum(patch // 4, 1) + 1, size=3)
        start = center + jitter - patch // 2
    else:
        start = np.asarray([rng.integers(0, int(v) + 1) if int(v) > 0 else 0 for v in max_start], dtype=np.int64)
    start = np.clip(start, 0, max_start)
    return int(start[0]), int(start[1]), int(start[2])


class TeacherCacheVolumeDataset(Dataset):
    """Teacher-cache dataset using foreground-aware random crops."""

    def __init__(
        self,
        case_ids: Sequence[str],
        lookup: Dict[str, Tuple[Path, Path]],
        teacher_cache_root: Path,
        split: str,
        target_name: str,
        patch_size: Sequence[int],
        samples_per_case: int,
        foreground_crop_prob: float,
        seed: int,
    ) -> None:
        self.items: list[Tuple[str, Path, Path]] = []
        for case_id in case_ids:
            if str(case_id) not in lookup:
                continue
            npz_path = teacher_cache_root / split / target_name / f"{case_id}.npz"
            if npz_path.is_file():
                self.items.append((str(case_id), lookup[str(case_id)][0], npz_path))
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_case = int(max(1, samples_per_case))
        self.foreground_crop_prob = float(np.clip(foreground_crop_prob, 0.0, 1.0))
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(len(self.items) * self.samples_per_case)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case_index = int(index) % len(self.items)
        case_id, image_path, npz_path = self.items[case_index]
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = 0 if worker_info is None else int(worker_info.id) * 1000003
        rng = np.random.default_rng(self.seed + worker_seed + int(index))
        volume = normalize_volume(np.asarray(load_volume_any(image_path), dtype=np.float32))
        with np.load(npz_path) as cache:
            final = np.asarray(cache["final_mask"] > 0, dtype=np.float32)
            core = np.asarray(cache["core_mask"] > 0, dtype=np.float32)
            teacher_prob = np.asarray(cache["mean_prob"], dtype=np.float32) if "mean_prob" in cache else final
            teacher_support = (
                np.asarray(cache["support_ratio"], dtype=np.float32)
                if "support_ratio" in cache
                else np.asarray(final > 0.5, dtype=np.float32)
            )
        foreground = np.logical_or(final > 0.5, core > 0.5)
        start = choose_foreground_aware_start(
            foreground=foreground,
            volume_shape=volume.shape,
            patch_size=self.patch_size,
            rng=rng,
            foreground_prob=self.foreground_crop_prob,
        )
        volume_patch = crop_or_pad_zyx(volume, start, self.patch_size, pad_value=0.0)
        final_patch = crop_or_pad_zyx(final, start, self.patch_size, pad_value=0.0)
        core_patch = crop_or_pad_zyx(core, start, self.patch_size, pad_value=0.0)
        prob_patch = crop_or_pad_zyx(np.clip(teacher_prob, 0.0, 1.0), start, self.patch_size, pad_value=0.0)
        support_patch = crop_or_pad_zyx(np.clip(teacher_support, 0.0, 1.0), start, self.patch_size, pad_value=0.0)
        return {
            "case_id": case_id,
            "image": zyx_to_model_tensor(volume_patch),
            "final_target": zyx_to_model_tensor(final_patch > 0.5),
            "core_target": zyx_to_model_tensor(core_patch > 0.5),
            "teacher_prob": zyx_to_model_tensor(prob_patch),
            "teacher_support": zyx_to_model_tensor(support_patch),
            "volume_shape": tuple(int(v) for v in volume.shape),
            "patch_start_zyx": tuple(int(v) for v in start),
        }


def case_lookup(dataset: str, dataset_root: str) -> Dict[str, Tuple[Path, Path]]:
    from dataset import collect_cases, resolve_dataset_config

    cfg = resolve_dataset_config(dataset=dataset, dataset_root=dataset_root)
    all_cases, _ = collect_cases(cfg)
    return {str(case_id): (Path(image_path), Path(label_path)) for case_id, image_path, label_path in all_cases}

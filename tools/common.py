from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tools.volumes import load_volume_any

EPS = 1e-6


def parse_patch_size(values: Sequence[int]) -> Tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("patch_size must contain exactly 3 integers.")
    return tuple(int(v) for v in values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < EPS:
        std = 1.0
    return ((arr - mean) / std).astype(np.float32)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def compute_bbox_zyx(mask: np.ndarray) -> List[int]:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return [0, 0, 0, 0, 0, 0]
    coords = np.argwhere(m)
    z0, y0, x0 = coords.min(axis=0).tolist()
    z1, y1, x1 = (coords.max(axis=0) + 1).tolist()
    return [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)]


def crop_with_pad_3d(
    arr: np.ndarray,
    start_zyx: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    z0, y0, x0 = [int(v) for v in start_zyx]
    dz, dy, dx = [int(v) for v in patch_size]
    src = np.asarray(arr)
    out = np.full((dz, dy, dx), pad_value, dtype=src.dtype)

    src_z0 = max(0, z0)
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_z1 = min(src.shape[0], z0 + dz)
    src_y1 = min(src.shape[1], y0 + dy)
    src_x1 = min(src.shape[2], x0 + dx)

    dst_z0 = src_z0 - z0
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_z1 = dst_z0 + max(0, src_z1 - src_z0)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)

    if src_z1 > src_z0 and src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = src[
            src_z0:src_z1,
            src_y0:src_y1,
            src_x0:src_x1,
        ]
    return out


def sample_patch_start_zyx(
    shape_zyx: Sequence[int],
    patch_size: Tuple[int, int, int],
    final_mask: np.ndarray,
    core_mask: np.ndarray,
    rng: np.random.Generator,
    fg_prob: float,
    core_focus_prob: float,
    hard_bg_prob: float = 0.0,
    hard_bg_max_fg_fraction: float = 0.0,
    hard_bg_trials: int = 16,
    force_global: bool = False,
) -> Tuple[int, int, int]:
    depth, height, width = [int(v) for v in shape_zyx]
    dz, dy, dx = [int(v) for v in patch_size]

    def _random_start(limit: int, span: int) -> int:
        upper = max(0, int(limit) - int(span))
        if upper <= 0:
            return 0
        return int(rng.integers(0, upper + 1))

    def _foreground_voxels_at(start_zyx: Tuple[int, int, int]) -> int:
        z0, y0, x0 = [int(v) for v in start_zyx]
        z1 = min(depth, z0 + dz)
        y1 = min(height, y0 + dy)
        x1 = min(width, x0 + dx)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return 0
        return int(np.sum(np.asarray(final_mask[z0:z1, y0:y1, x0:x1]) > 0.5))

    if bool(force_global):
        return (
            _random_start(depth, dz),
            _random_start(height, dy),
            _random_start(width, dx),
        )

    hard_bg_prob = min(max(float(hard_bg_prob), 0.0), 1.0)
    if hard_bg_prob > 0.0 and float(rng.random()) < hard_bg_prob:
        max_fg_voxels = max(
            0,
            int(round(float(hard_bg_max_fg_fraction) * float(dz * dy * dx))),
        )
        best_start = (
            _random_start(depth, dz),
            _random_start(height, dy),
            _random_start(width, dx),
        )
        best_fg_voxels: int | None = None
        for _ in range(max(1, int(hard_bg_trials))):
            candidate = (
                _random_start(depth, dz),
                _random_start(height, dy),
                _random_start(width, dx),
            )
            fg_voxels = _foreground_voxels_at(candidate)
            if best_fg_voxels is None or fg_voxels < best_fg_voxels:
                best_start = candidate
                best_fg_voxels = fg_voxels
            if fg_voxels <= max_fg_voxels:
                return candidate
        return best_start

    use_positive = bool(np.any(final_mask)) and float(rng.random()) < float(fg_prob)
    if use_positive:
        use_core = bool(np.any(core_mask)) and float(rng.random()) < float(core_focus_prob)
        coords = np.argwhere(core_mask if use_core else final_mask)
        if coords.size > 0:
            cz, cy, cx = coords[int(rng.integers(0, coords.shape[0]))].tolist()
            z0 = int(cz - dz // 2)
            y0 = int(cy - dy // 2)
            x0 = int(cx - dx // 2)
            z0 = min(max(z0, 0), max(0, depth - dz))
            y0 = min(max(y0, 0), max(0, height - dy))
            x0 = min(max(x0, 0), max(0, width - dx))
            return z0, y0, x0

    return (
        _random_start(depth, dz),
        _random_start(height, dy),
        _random_start(width, dx),
    )


def np_patch_to_model_input(volume_patch_zyx: np.ndarray) -> torch.Tensor:
    arr = np.asarray(volume_patch_zyx, dtype=np.float32)
    return torch.tensor(np.transpose(arr, (1, 2, 0)).tolist(), dtype=torch.float32).unsqueeze(0)


def np_patch_to_target(volume_patch_zyx: np.ndarray) -> torch.Tensor:
    arr = np.asarray(volume_patch_zyx, dtype=np.float32)
    return torch.tensor(np.transpose(arr, (1, 2, 0)).tolist(), dtype=torch.float32).unsqueeze(0)


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    if weight is None:
        weight = torch.ones_like(target)
    weight = weight.float()
    inter = torch.sum(weight * probs * target, dim=(1, 2, 3, 4))
    denom = torch.sum(weight * probs, dim=(1, 2, 3, 4)) + torch.sum(weight * target, dim=(1, 2, 3, 4))
    dice = (2.0 * inter + EPS) / (denom + EPS)
    return 1.0 - dice.mean()


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return torch.sum(loss * weight) / torch.sum(weight).clamp_min(EPS)


def soft_kl_distill_loss(
    logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    temperature: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    temp = float(max(temperature, 1e-4))
    student_log_prob_fg = F.logsigmoid(logits / temp)
    student_log_prob_bg = F.logsigmoid(-logits / temp)
    teacher_prob = teacher_prob.float().clamp(1e-5, 1.0 - 1e-5)
    teacher_bg = 1.0 - teacher_prob
    loss = teacher_prob * (torch.log(teacher_prob) - student_log_prob_fg)
    loss += teacher_bg * (torch.log(teacher_bg) - student_log_prob_bg)
    loss = loss * (temp * temp)
    if weight is None:
        return loss.mean()
    return torch.sum(loss * weight) / torch.sum(weight).clamp_min(EPS)


def binary_dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    inter = int(np.sum(pred & gt))
    denom = int(np.sum(pred)) + int(np.sum(gt))
    if denom <= 0:
        return 1.0
    return float((2.0 * inter) / max(denom, 1))


def build_hard_loss_weight(
    final_target: torch.Tensor,
    core_target: torch.Tensor,
    bg_weight: float,
    fg_weight: float,
    core_weight: float,
) -> torch.Tensor:
    weight = torch.full_like(final_target, float(bg_weight))
    weight = torch.where(final_target > 0.5, torch.full_like(weight, float(fg_weight)), weight)
    weight = torch.where(core_target > 0.5, torch.full_like(weight, float(core_weight)), weight)
    return weight


@dataclass(frozen=True)
class TeacherCacheFiles:
    npz_path: Path
    meta_path: Path


def teacher_cache_files(
    teacher_cache_root: Path,
    split: str,
    target_name: str,
    case_id: str,
) -> TeacherCacheFiles:
    npz_path = teacher_cache_root / str(split) / str(target_name) / f"{case_id}.npz"
    meta_path = teacher_cache_root / str(split) / str(target_name) / f"{case_id}.json"
    return TeacherCacheFiles(npz_path=npz_path, meta_path=meta_path)


def load_teacher_cache_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


class SingleTargetTeacherPatchDataset(Dataset):
    def __init__(
        self,
        case_specs: Sequence[Tuple[str, Path, TeacherCacheFiles]],
        patch_size: Tuple[int, int, int],
        samples_per_case: int,
        fg_prob: float,
        core_focus_prob: float,
        seed: int,
        hard_bg_prob: float = 0.0,
        hard_bg_max_fg_fraction: float = 0.0,
        hard_bg_trials: int = 16,
    ) -> None:
        self.case_specs = []
        for spec in case_specs:
            if len(spec) == 3:
                target_id, img_path, cache_files = spec
                teacher_weight = 1.0
                teacher_quality_score = 1.0
            elif len(spec) == 5:
                target_id, img_path, cache_files, teacher_weight, teacher_quality_score = spec
            else:
                raise ValueError(f"Unexpected teacher case spec length={len(spec)} for {spec!r}")
            self.case_specs.append(
                (
                    str(target_id),
                    Path(img_path),
                    cache_files,
                    float(teacher_weight),
                    float(teacher_quality_score),
                )
            )
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_case = int(max(1, samples_per_case))
        self.fg_prob = float(fg_prob)
        self.core_focus_prob = float(core_focus_prob)
        self.hard_bg_prob = float(hard_bg_prob)
        self.hard_bg_max_fg_fraction = float(hard_bg_max_fg_fraction)
        self.hard_bg_trials = int(hard_bg_trials)
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(len(self.case_specs) * self.samples_per_case)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case_idx = int(index) % len(self.case_specs)
        sample_slot = int(index) // len(self.case_specs)
        case_id, image_path, cache_files, teacher_weight, teacher_quality_score = self.case_specs[case_idx]
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = 0 if worker_info is None else int(worker_info.id) * 1000003
        rng = np.random.default_rng(self.seed + worker_seed + int(index))

        volume = normalize_volume(np.asarray(load_volume_any(image_path), dtype=np.float32))
        cache = load_teacher_cache_npz(cache_files.npz_path)
        final_mask = np.asarray(cache["final_mask"], dtype=np.float32)
        core_mask = np.asarray(cache["core_mask"], dtype=np.float32)
        # Keep distillation consumers compatible with minimal teacher caches:
        # final_mask/core_mask are the only required teacher targets.
        teacher_prob = np.asarray(final_mask > 0.5, dtype=np.float32)
        teacher_support = np.asarray(final_mask > 0.5, dtype=np.float32)

        start_zyx = sample_patch_start_zyx(
            shape_zyx=volume.shape,
            patch_size=self.patch_size,
            final_mask=final_mask > 0.5,
            core_mask=core_mask > 0.5,
            rng=rng,
            fg_prob=self.fg_prob,
            core_focus_prob=self.core_focus_prob,
            hard_bg_prob=self.hard_bg_prob,
            hard_bg_max_fg_fraction=self.hard_bg_max_fg_fraction,
            hard_bg_trials=self.hard_bg_trials,
        )
        image_patch = crop_with_pad_3d(volume, start_zyx, self.patch_size, pad_value=0.0)
        final_patch = crop_with_pad_3d(final_mask, start_zyx, self.patch_size, pad_value=0.0)
        core_patch = crop_with_pad_3d(core_mask, start_zyx, self.patch_size, pad_value=0.0)
        prob_patch = crop_with_pad_3d(teacher_prob, start_zyx, self.patch_size, pad_value=0.0)
        support_patch = crop_with_pad_3d(teacher_support, start_zyx, self.patch_size, pad_value=0.0)

        return {
            "case_id": case_id,
            "image": np_patch_to_model_input(image_patch),
            "final_target": np_patch_to_target(final_patch),
            "core_target": np_patch_to_target(core_patch),
            "teacher_prob": np_patch_to_target(prob_patch),
            "teacher_support": np_patch_to_target(support_patch),
            "teacher_weight": torch.tensor(float(teacher_weight), dtype=torch.float32),
            "teacher_quality_score": torch.tensor(float(teacher_quality_score), dtype=torch.float32),
        }

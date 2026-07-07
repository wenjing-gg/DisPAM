#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import collect_cases, load_split_case_ids, resolve_dataset_config, resolve_split_config_path
from tools.profiles import get_target_profile
from tools.pam import (
    BOX_PROMPT_VARIANT_BASELINE,
    FINAL_PROMPT_PERTURBATION_SCHEDULE,
    build_pam_backend,
    build_prompt_mask_from_label_boxes,
    build_selector_config,
    dice_coefficient,
    normalize_box2mask_runtime_mode,
    normalize_box_prompt_variant,
    run_final_prompt_ensemble,
    set_seed,
)
from tools.common import compute_bbox_zyx, teacher_cache_files, write_json
from tools.volumes import load_volume_any

DEFAULT_PAM_ROOT = REPO_ROOT / "PAM"
DEFAULT_PROP_CKPT = DEFAULT_PAM_ROOT / "tutorials" / "checkpoint" / "propmask.pth"
DEFAULT_BOX_CKPT = DEFAULT_PAM_ROOT / "tutorials" / "checkpoint" / "box2mask.pth"


TARGET_ORDER: Dict[str, Tuple[str, ...]] = {
    "kits19": ("kidney", "tumor"),
    "lits": ("liver", "tumor"),
    "brats18_flair": ("et", "tc", "wt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build box-prompt PAM teacher cache from trusted slice-cache anchors. "
            "For each target, the selected_slice_id is copied from the source slice cache, "
            "the GT mask on that exact slice is converted to component boxes, and PAM "
            "box2mask+propmask is rerun. No fallback slice selection is used."
        )
    )
    parser.add_argument("--dataset", default="kits19")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--run-config", default=None)
    parser.add_argument("--source-slice-cache-root", required=True)
    parser.add_argument("--teacher-cache-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val"])
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--case-ids-file", default=None)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--target-names", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pam-root", default=str(DEFAULT_PAM_ROOT))
    parser.add_argument(
        "--propmask-ckpt",
        default=str(DEFAULT_PROP_CKPT),
    )
    parser.add_argument(
        "--box2mask-ckpt",
        default=str(DEFAULT_BOX_CKPT),
    )
    parser.add_argument("--box-prompt-variant", default=BOX_PROMPT_VARIANT_BASELINE)
    parser.add_argument("--runtime-mode", default="aligned")
    parser.add_argument("--crop-ratio", type=float, default=1.3)
    parser.add_argument("--eval-prob-thresh", type=float, default=0.3)
    parser.add_argument("--final-core-support-thresh", type=float, default=0.7)
    parser.add_argument("--final-candidate-support-thresh", type=float, default=0.25)
    parser.add_argument("--final-keep-lcc", type=int, default=1)
    parser.add_argument(
        "--final-prompt-perturbations",
        nargs="+",
        default=list(FINAL_PROMPT_PERTURBATION_SCHEDULE),
    )
    parser.add_argument("--prop-batch-size", type=int, default=64)
    parser.add_argument("--emb-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def normalize_target_names(dataset: str, raw_targets: Sequence[str] | None) -> Tuple[str, ...]:
    profile = get_target_profile(dataset)
    available = tuple(TARGET_ORDER.get(resolve_dataset_config(dataset=dataset).name, tuple(profile.branch_source_label_ids)))
    if not available:
        available = tuple(profile.branch_source_label_ids)
    if raw_targets is None:
        return available
    out: List[str] = []
    for raw in raw_targets:
        for item in str(raw).replace(",", " ").split():
            target = item.strip().lower()
            if not target:
                continue
            if target not in profile.branch_source_label_ids:
                raise ValueError(
                    f"Unsupported target '{target}' for dataset={dataset}. "
                    f"Expected one of {tuple(profile.branch_source_label_ids)}."
                )
            if target not in out:
                out.append(target)
    if not out:
        raise ValueError("--target-names resolved to an empty target list.")
    return tuple(out)


def requested_case_ids(args: argparse.Namespace, run_config_path: Path, split: str) -> List[str]:
    case_ids: List[str] = []
    if args.case_ids:
        case_ids.extend(str(item).strip() for item in args.case_ids if str(item).strip())
    if args.case_ids_file:
        for line in Path(args.case_ids_file).read_text().splitlines():
            text = line.strip()
            if text:
                case_ids.append(text)
    if not case_ids:
        case_ids = [str(item) for item in load_split_case_ids(run_config_path, split)]
    if int(args.max_cases) > 0:
        case_ids = case_ids[: int(args.max_cases)]
    return case_ids


def support_z_range(mask: np.ndarray) -> List[int]:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3 or not np.any(m):
        return [0, 0]
    z_any = np.where(m.reshape(m.shape[0], -1).any(axis=1))[0]
    return [int(z_any[0]), int(z_any[-1] + 1)]


def output_cache_complete(cache_root: Path, split: str, target_name: str, case_id: str) -> bool:
    files = teacher_cache_files(cache_root, split, target_name, case_id)
    if not files.npz_path.is_file() or not files.meta_path.is_file():
        return False
    try:
        meta = load_json(files.meta_path)
    except Exception:
        return False
    if str(meta.get("prompt_annotation_type", "")).strip().lower() != "box":
        return False
    if str(meta.get("source_prompt_annotation_type", "")).strip().lower() != "slice_anchor":
        return False
    return True


def build_pam(args: argparse.Namespace, pam_modality: str) -> Any:
    selector_args = build_selector_config(
        device=str(args.device),
        pam_root=str(args.pam_root),
        pam_ckpt=str(args.propmask_ckpt),
        pam_box2mask_ckpt=str(args.box2mask_ckpt),
        pam_box2mask_runtime_mode=normalize_box2mask_runtime_mode(args.runtime_mode),
        prompt_annotation_type="box",
        box_prompt_variant=normalize_box_prompt_variant(args.box_prompt_variant),
        seed=int(args.seed),
        eval_prob_thresh=float(args.eval_prob_thresh),
        final_core_support_thresh=float(args.final_core_support_thresh),
        final_candidate_support_thresh=float(args.final_candidate_support_thresh),
        final_keep_lcc=int(args.final_keep_lcc),
    )
    selector_args.dataset = str(args.dataset)
    selector_args.pam_modality = str(pam_modality)
    selector_args.pam_dynamic_crop_ratio = float(args.crop_ratio)
    selector_args.pam_prop_batch_size = int(args.prop_batch_size)
    selector_args.pam_emb_batch_size = int(args.emb_batch_size)
    set_seed(int(args.seed))
    return build_pam_backend(selector_args)


def empty_pack(
    *,
    volume_shape: Sequence[int],
    target_name: str,
    label_id: int,
    source_label_ids: Sequence[int],
    source_meta: Dict[str, Any],
) -> Dict[str, Any]:
    shape = tuple(int(v) for v in volume_shape)
    final_mask = np.zeros(shape, dtype=bool)
    return {
        "target_name": str(target_name),
        "target_absent": True,
        "teacher_status": "target_absent_gt_zero",
        "label_id": int(label_id),
        "source_label_ids": [int(v) for v in source_label_ids],
        "selected_slice_id": int(source_meta.get("selected_slice_id", -1)),
        "selected_score": float(source_meta.get("selected_score", 0.0)),
        "selected_quality_score": float(source_meta.get("selected_quality_score", 0.0)),
        "final_mask": final_mask,
        "core_mask": final_mask.copy(),
        "final_dice": 1.0,
        "core_dice": 1.0,
        "gt_voxel_count": 0,
        "prompt_gt_area": 0,
        "prompt_mask_area": 0,
        "prompt_component_area": 0,
        "prompt_box_count": 0,
        "prompt_box_area": 0,
        "box2mask_prompt_area": 0,
        "prompt_boxes": [],
        "final_voxels": 0,
        "core_voxels": 0,
        "ensemble_size": 0,
        "weak_perturbation_count": 0,
        "strong_perturbation_count": 0,
        "final_support_mean": 0.0,
        "core_support_mean": 0.0,
        "support_peak": 0.0,
    }


def build_box_pack(
    *,
    volume: np.ndarray,
    label: np.ndarray,
    pam: Any,
    target_name: str,
    label_id: int,
    source_label_ids: Sequence[int],
    source_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    target_mask = np.isin(np.asarray(label, dtype=np.int32), [int(v) for v in source_label_ids])
    gt_voxel_count = int(np.sum(target_mask))
    if gt_voxel_count <= 0:
        return empty_pack(
            volume_shape=label.shape,
            target_name=target_name,
            label_id=label_id,
            source_label_ids=source_label_ids,
            source_meta=source_meta,
        )

    if "selected_slice_id" not in source_meta:
        raise RuntimeError(f"{target_name}: source slice cache metadata has no selected_slice_id.")
    sid = int(source_meta["selected_slice_id"])
    if sid < 0 or sid >= int(label.shape[0]):
        raise RuntimeError(
            f"{target_name}: selected_slice_id={sid} is invalid for label shape={tuple(label.shape)}."
        )
    prompt_gt_area = int(np.sum(target_mask[sid]))
    if prompt_gt_area <= 0:
        raise RuntimeError(
            f"{target_name}: selected_slice_id={sid} has empty target GT mask; no fallback is allowed."
        )

    prompt_mask, prompt_meta = build_prompt_mask_from_label_boxes(
        volume=np.asarray(volume, dtype=np.float32),
        prompt_slice_id=int(sid),
        label_slice_2d=np.asarray(label[int(sid)], dtype=np.int32),
        pam=pam,
        prob_thresh=float(args.eval_prob_thresh),
        label_ids=[int(v) for v in source_label_ids],
        box_prompt_variant=str(args.box_prompt_variant),
    )
    prompt_mask = np.asarray(prompt_mask > 0, dtype=bool)
    if int(np.sum(prompt_mask)) <= 0:
        raise RuntimeError(f"{target_name}: PAM box2mask produced an empty 2D prompt.")

    ensemble = run_final_prompt_ensemble(
        volume=np.asarray(volume, dtype=np.float32),
        pam=pam,
        prompt_slice_id=int(sid),
        prompt_mask_2d=prompt_mask,
        prob_thresh=float(args.eval_prob_thresh),
        core_support_thresh=float(args.final_core_support_thresh),
        candidate_support_thresh=float(args.final_candidate_support_thresh),
        keep_lcc=bool(int(args.final_keep_lcc)),
        seed=int(args.seed),
        perturbation_schedule=list(args.final_prompt_perturbations),
        anchor_to_prompt=True,
    )
    final_mask = np.asarray(ensemble["final_mask"], dtype=bool)
    core_mask = np.asarray(ensemble["core_mask"], dtype=bool)
    final_dice = float(dice_coefficient(final_mask.astype(np.uint8), target_mask.astype(np.uint8)))
    core_dice = float(dice_coefficient(core_mask.astype(np.uint8), target_mask.astype(np.uint8)))
    return {
        "target_name": str(target_name),
        "target_absent": False,
        "teacher_status": "ok",
        "label_id": int(label_id),
        "source_label_ids": [int(v) for v in source_label_ids],
        "selected_slice_id": int(sid),
        "selected_score": float(source_meta.get("selected_score", 0.0)),
        "selected_quality_score": float(
            source_meta.get("selected_quality_score", source_meta.get("selected_score", 0.0))
        ),
        "final_mask": final_mask,
        "core_mask": core_mask,
        "final_dice": final_dice,
        "core_dice": core_dice,
        "gt_voxel_count": int(gt_voxel_count),
        "prompt_gt_area": int(prompt_gt_area),
        "prompt_mask_area": int(prompt_meta["prompt_mask_area"]),
        "prompt_component_area": int(prompt_meta["sum_component_area"]),
        "prompt_box_count": int(prompt_meta["box_count"]),
        "prompt_box_area": int(prompt_meta["union_box_area"]),
        "box2mask_prompt_area": int(prompt_meta["prompt_mask_area"]),
        "prompt_boxes": list(prompt_meta.get("boxes", [])),
        "final_voxels": int(ensemble["final_voxels"]),
        "core_voxels": int(ensemble["core_voxels"]),
        "ensemble_size": int(ensemble["ensemble_size"]),
        "weak_perturbation_count": int(ensemble["weak_perturbation_count"]),
        "strong_perturbation_count": int(ensemble["strong_perturbation_count"]),
        "final_support_mean": float(ensemble["final_support_mean"]),
        "core_support_mean": float(ensemble["core_support_mean"]),
        "support_peak": float(ensemble["support_peak"]),
    }


def save_target_cache(
    *,
    cache_root: Path,
    source_root: Path,
    source_meta_path: Path,
    split: str,
    case_id: str,
    target_name: str,
    image_path: Path,
    label_path: Path,
    pack: Dict[str, Any],
    source_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    files = teacher_cache_files(cache_root, split, target_name, case_id)
    files.npz_path.parent.mkdir(parents=True, exist_ok=True)
    final_mask = np.asarray(pack["final_mask"], dtype=np.uint8)
    core_mask = np.asarray(pack["core_mask"], dtype=np.uint8)
    np.savez_compressed(files.npz_path, final_mask=final_mask, core_mask=core_mask)

    volume_shape = [int(v) for v in final_mask.shape]
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_id": str(case_id),
        "target_name": str(target_name),
        "split": str(split),
        "dataset": str(args.dataset),
        "target_absent": bool(pack.get("target_absent", False)),
        "teacher_status": str(pack.get("teacher_status", "ok")),
        "prompt_annotation_type": "box",
        "source_prompt_annotation_type": "slice_anchor",
        "prompt_source": "box2mask_from_gt_mask_boxes_on_slice_anchor",
        "prompt_sources": ["box2mask"],
        "anchor_to_prompt": True,
        "label_id": int(pack["label_id"]),
        "source_label_ids": [int(v) for v in pack.get("source_label_ids", [])],
        "volume_shape": volume_shape,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "selected_slice_id": int(pack["selected_slice_id"]),
        "source_selected_slice_id": int(source_meta.get("selected_slice_id", pack["selected_slice_id"])),
        "selected_score": float(pack["selected_score"]),
        "selected_quality_score": float(pack["selected_quality_score"]),
        "final_dice": float(pack["final_dice"]),
        "core_dice": float(pack["core_dice"]),
        "gt_voxel_count": int(pack["gt_voxel_count"]),
        "prompt_gt_area": int(pack["prompt_gt_area"]),
        "prompt_mask_area": int(pack["prompt_mask_area"]),
        "prompt_component_area": int(pack["prompt_component_area"]),
        "prompt_box_count": int(pack["prompt_box_count"]),
        "prompt_box_area": int(pack["prompt_box_area"]),
        "box2mask_prompt_area": int(pack["box2mask_prompt_area"]),
        "prompt_boxes": list(pack.get("prompt_boxes", [])),
        "box_prompt_variant": str(args.box_prompt_variant),
        "final_voxels": int(pack["final_voxels"]),
        "core_voxels": int(pack["core_voxels"]),
        "final_support_mean": float(pack["final_support_mean"]),
        "core_support_mean": float(pack["core_support_mean"]),
        "support_peak": float(pack["support_peak"]),
        "ensemble_size": int(pack["ensemble_size"]),
        "weak_perturbation_count": int(pack["weak_perturbation_count"]),
        "strong_perturbation_count": int(pack["strong_perturbation_count"]),
        "final_bbox_zyx": compute_bbox_zyx(final_mask),
        "core_bbox_zyx": compute_bbox_zyx(core_mask),
        "final_z_range": support_z_range(final_mask),
        "core_z_range": support_z_range(core_mask),
        "teacher_algorithm": {
            "script": "tools/cache.py",
            "prompt_annotation_type": "box",
            "source_prompt_annotation_type": "slice_anchor",
            "no_fallback": True,
            "same_selected_slice_as_source_slice_cache": True,
            "box_prompt_variant": str(args.box_prompt_variant),
            "runtime_mode": str(args.runtime_mode),
            "crop_ratio": float(args.crop_ratio),
            "eval_prob_thresh": float(args.eval_prob_thresh),
            "final_core_support_thresh": float(args.final_core_support_thresh),
            "final_candidate_support_thresh": float(args.final_candidate_support_thresh),
            "final_keep_lcc": int(args.final_keep_lcc),
            "anchor_to_prompt": 1,
            "final_prompt_perturbations": list(args.final_prompt_perturbations),
        },
        "source_slice_teacher_cache": {
            "root": str(source_root),
            "meta_json": str(source_meta_path),
            "npz": str(source_meta_path.with_suffix(".npz")),
            "prompt_annotation_type": str(source_meta.get("prompt_annotation_type", "")),
            "teacher_status": str(source_meta.get("teacher_status", "")),
            "selected_slice_id": int(source_meta.get("selected_slice_id", -1)),
        },
        "cache_files": {"npz": str(files.npz_path), "meta_json": str(files.meta_path)},
        "cache_npz_keys": ["final_mask", "core_mask"],
    }
    write_json(files.meta_path, meta)


def count_target_cache_files(
    cache_root: Path,
    splits: Sequence[str],
    target_names: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split in splits:
        split_counts: Dict[str, Dict[str, int]] = {}
        for target_name in target_names:
            target_dir = cache_root / split / target_name
            split_counts[str(target_name)] = {
                "json": int(len(list(target_dir.glob("*.json"))) if target_dir.is_dir() else 0),
                "npz": int(len(list(target_dir.glob("*.npz"))) if target_dir.is_dir() else 0),
            }
        out[str(split)] = split_counts
    return out


def main() -> None:
    args = parse_args()
    args.box_prompt_variant = normalize_box_prompt_variant(args.box_prompt_variant)
    args.runtime_mode = normalize_box2mask_runtime_mode(args.runtime_mode)

    run_config_path = resolve_split_config_path(dataset=args.dataset, run_config=args.run_config)
    if run_config_path is None:
        raise RuntimeError(f"No split config path available for dataset={args.dataset!r}.")
    cfg = resolve_dataset_config(dataset=args.dataset, dataset_root=args.dataset_root)
    profile = get_target_profile(args.dataset)
    target_names = normalize_target_names(args.dataset, args.target_names)
    cache_root = Path(args.teacher_cache_root)
    source_root = Path(args.source_slice_cache_root)

    all_cases, missing = collect_cases(cfg)
    case_lookup = {str(case_id).lower(): (str(case_id), Path(image_path), Path(label_path)) for case_id, image_path, label_path in all_cases}

    pam = build_pam(args, pam_modality=cfg.pam_modality)
    summary_rows: List[Dict[str, Any]] = []
    normalized_splits = [str(item).lower() for item in args.splits]
    for split in normalized_splits:
        case_ids = requested_case_ids(args, run_config_path, split)
        print(f"[BoxFromSliceAnchor][Split] {split} cases={len(case_ids)} targets={list(target_names)}", flush=True)
        for case_index, case_id in enumerate(case_ids, start=1):
            print(f"[BoxFromSliceAnchor][{split}][{case_index}/{len(case_ids)}] {case_id}", flush=True)
            if str(case_id).lower() not in case_lookup:
                raise RuntimeError(f"Missing case in dataset: split={split} case={case_id}")
            _, image_path, label_path = case_lookup[str(case_id).lower()]
            volume = np.asarray(load_volume_any(image_path), dtype=np.float32)
            label = np.asarray(load_volume_any(label_path), dtype=np.int32)
            if tuple(volume.shape) != tuple(label.shape):
                raise RuntimeError(
                    f"Image/label shape mismatch for case={case_id}: image={volume.shape} label={label.shape}"
                )

            status_by_target: Dict[str, str] = {}
            dice_by_target: Dict[str, float] = {}
            for target_name in target_names:
                if not bool(args.overwrite) and output_cache_complete(cache_root, split, target_name, case_id):
                    status_by_target[str(target_name)] = "cached"
                    continue
                source_files = teacher_cache_files(source_root, split, target_name, case_id)
                if not source_files.meta_path.is_file():
                    raise RuntimeError(f"Missing source slice cache metadata: {source_files.meta_path}")
                source_meta = load_json(source_files.meta_path)
                source_label_ids = tuple(int(v) for v in profile.branch_source_label_ids[str(target_name)])
                label_id = int(profile.branch_export_label_ids[str(target_name)])
                pack = build_box_pack(
                    volume=volume,
                    label=label,
                    pam=pam,
                    target_name=str(target_name),
                    label_id=label_id,
                    source_label_ids=source_label_ids,
                    source_meta=source_meta,
                    args=args,
                )
                save_target_cache(
                    cache_root=cache_root,
                    source_root=source_root,
                    source_meta_path=source_files.meta_path,
                    split=split,
                    case_id=str(case_id),
                    target_name=str(target_name),
                    image_path=image_path,
                    label_path=label_path,
                    pack=pack,
                    source_meta=source_meta,
                    args=args,
                )
                status_by_target[str(target_name)] = str(pack.get("teacher_status", "ok"))
                dice_by_target[str(target_name)] = float(pack.get("final_dice", 0.0))
                print(
                    f"[BoxFromSliceAnchor][{split}] {case_id}/{target_name} "
                    f"sid={int(pack['selected_slice_id'])} "
                    f"boxes={int(pack['prompt_box_count'])} "
                    f"prompt={int(pack['box2mask_prompt_area'])} "
                    f"final={int(pack['final_voxels'])} "
                    f"dice_vs_gt={float(pack['final_dice']):.4f}",
                    flush=True,
                )

            valid_dice = [v for k, v in dice_by_target.items() if status_by_target.get(k) != "target_absent_gt_zero"]
            summary_rows.append(
                {
                    "split": str(split),
                    "case_id": str(case_id),
                    "status": "ok",
                    "prompt_annotation_type": "box",
                    "source_prompt_annotation_type": "slice_anchor",
                    "target_names": list(target_names),
                    "teacher_status_by_target": dict(status_by_target),
                    "selected_target_dice": dict(dice_by_target),
                    "selected_macro_dice": float(np.mean(np.asarray(valid_dice, dtype=np.float32))) if valid_dice else None,
                }
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "dataset_root": str(cfg.dataset_root),
        "run_config": str(run_config_path),
        "prompt_annotation_type": "box",
        "source_prompt_annotation_type": "slice_anchor",
        "source_slice_teacher_cache_root": str(source_root),
        "teacher_cache_root": str(cache_root),
        "target_names": list(target_names),
        "splits": list(normalized_splits),
        "missing_labels": list(missing),
        "rows": summary_rows,
        "summary": {
            "total_rows": int(len(summary_rows)),
            "target_cache_files": count_target_cache_files(cache_root, normalized_splits, target_names),
        },
    }
    write_json(cache_root / "summary.json", payload)
    print("[Saved]", cache_root / "summary.json", flush=True)


if __name__ == "__main__":
    main()

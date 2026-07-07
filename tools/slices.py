#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import collect_cases, load_split_case_ids, resolve_dataset_config, resolve_split_config_path
from tools import pam as pam_tools
from tools.profiles import get_target_profile
from tools.pam import (
    DEFAULT_EVAL_PROB_THRESH,
    DEFAULT_SLICE_DYNAMIC_CROP_RATIO,
    AGENT_BASE_URL,
    AGENT_FALLBACK_BASE_URLS,
    AGENT_MAX_RETRIES,
    AGENT_MODEL,
    AGENT_MODEL_PROVIDER,
    AGENT_REFERENCE_EXAMPLES_ENABLED,
    AGENT_REFERENCE_EXAMPLES_PER_BUCKET,
    AGENT_REASONING_EFFORT,
    AGENT_REQUEST_TIMEOUT_SECONDS,
    AGENT_RETRY_BACKOFF_SECONDS,
    AGENT_USE_SYSTEM_PROXY,
    AGENT_WIRE_API,
    AGENT_CONFIG_PATH,
    ask_agent_slice_complete,
    build_pam_backend,
    build_prompt_mask_from_annotation,
    build_selector_config,
    dice_coefficient,
    load_agent_api_key_from_config,
    run_final_prompt_ensemble,
    set_seed,
)
from tools.volumes import load_volume_any


EPS = 1e-8
DEFAULT_PROP_CKPT = REPO_ROOT / "PAM" / "tutorials" / "checkpoint" / "propmask.pth"
DEFAULT_BOX_CKPT = REPO_ROOT / "PAM" / "tutorials" / "checkpoint" / "box2mask.pth"


@dataclass(frozen=True)
class TransitionEstimate:
    ranked_slices: List[Dict[str, Any]]
    score_profile: np.ndarray
    transition_profile: np.ndarray
    interior_profile: np.ndarray
    support_profile: np.ndarray
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free slice selection by adjacent-slice transition visibility, "
            "then optional PAM evaluation from the selected true single-slice prompt."
        )
    )
    parser.add_argument("--dataset", default="brats18_flair")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--run-config", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--target-names", nargs="+", default=None)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--use-agent-selection", type=int, default=1, choices=[0, 1])
    parser.add_argument("--api_key_path", type=str, default=str(AGENT_CONFIG_PATH))
    parser.add_argument("--agent_model_provider", type=str, default=AGENT_MODEL_PROVIDER)
    parser.add_argument("--agent_model", type=str, default=AGENT_MODEL)
    parser.add_argument("--agent_reasoning_effort", type=str, default=AGENT_REASONING_EFFORT)
    parser.add_argument("--agent_base_url", type=str, default=AGENT_BASE_URL)
    parser.add_argument("--agent_wire_api", type=str, default=AGENT_WIRE_API)
    parser.add_argument("--agent_fallback_base_urls", type=str, default=",".join(str(v) for v in AGENT_FALLBACK_BASE_URLS))
    parser.add_argument(
        "--agent_reference_examples_enabled",
        type=int,
        default=1 if AGENT_REFERENCE_EXAMPLES_ENABLED else 0,
        choices=[0, 1],
    )
    parser.add_argument("--agent_reference_examples_per_bucket", type=int, default=AGENT_REFERENCE_EXAMPLES_PER_BUCKET)
    parser.add_argument("--agent_max_retries", type=int, default=AGENT_MAX_RETRIES)
    parser.add_argument("--agent_retry_backoff_seconds", type=float, default=AGENT_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--agent_request_timeout_seconds", type=float, default=AGENT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--agent_use_system_proxy", type=int, default=1 if AGENT_USE_SYSTEM_PROXY else 0, choices=[0, 1])
    parser.add_argument("--eval-pam", type=int, default=1)
    parser.add_argument("--pam-device", default="cuda:7")
    parser.add_argument("--pam-root", default=str(REPO_ROOT / "PAM"))
    parser.add_argument("--propmask-ckpt", default=str(DEFAULT_PROP_CKPT))
    parser.add_argument("--box2mask-ckpt", default=str(DEFAULT_BOX_CKPT))
    parser.add_argument("--prompt-annotation-type", default="slice", choices=("slice", "box"))
    parser.add_argument("--eval-prob-thresh", type=float, default=DEFAULT_EVAL_PROB_THRESH)
    parser.add_argument("--final-core-support-thresh", type=float, default=0.7)
    parser.add_argument("--final-candidate-support-thresh", type=float, default=0.25)
    parser.add_argument("--final-keep-lcc", type=int, default=1)
    parser.add_argument("--prop-batch-size", type=int, default=64)
    parser.add_argument("--emb-batch-size", type=int, default=64)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--disable-cv2-resize", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/select/results.json")
    return parser.parse_args()


def normalize_target_names(raw_targets: Sequence[str] | None, available_targets: Sequence[str]) -> Tuple[str, ...]:
    available = tuple(str(item).strip().lower() for item in available_targets)
    if raw_targets is None:
        return available
    normalized: List[str] = []
    for raw in raw_targets:
        for item in str(raw).replace(",", " ").split():
            target = item.strip().lower()
            if not target:
                continue
            if target not in available:
                raise ValueError(f"Unsupported target '{target}'. Expected one of {available}.")
            if target not in normalized:
                normalized.append(target)
    if not normalized:
        raise ValueError("--target-names resolved to an empty target list.")
    return tuple(normalized)


def target_mask_from_label(label: np.ndarray, source_label_ids: Sequence[int]) -> np.ndarray:
    return np.asarray(np.isin(np.asarray(label, dtype=np.int32), [int(v) for v in source_label_ids]), dtype=bool)


def ensure_agent_api_key(args: argparse.Namespace) -> None:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return
    api_key_path = Path(str(getattr(args, "api_key_path", "") or "")).expanduser()
    if not str(api_key_path):
        raise RuntimeError("OPENAI_API_KEY is not set and api_key_path is empty.")
    if not api_key_path.is_file():
        raise RuntimeError(f"OPENAI_API_KEY is not set and api_key_path does not exist: {api_key_path}")
    if api_key_path.suffix.lower() == ".toml":
        value = load_agent_api_key_from_config(api_key_path)
    else:
        value = api_key_path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"Agent API key file is empty or invalid: {api_key_path}")
    os.environ["OPENAI_API_KEY"] = str(value)


def build_agent_args(args: argparse.Namespace, profile: Any, target_name: str) -> argparse.Namespace:
    target = str(target_name).strip().lower()
    prompt = str(profile.branch_agent_prompts[target])
    payload = dict(vars(args))
    payload["agent_prompt"] = prompt
    payload["agent_branch"] = target
    payload["agent_reference_examples_enabled"] = 0
    return argparse.Namespace(**payload)


def run_target_agent_selection(
    *,
    volume: np.ndarray,
    ranked_slices: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    profile: Any,
    target_name: str,
    case_id: str,
) -> Dict[str, Any]:
    if not ranked_slices:
        raise RuntimeError(f"No Top-k slice candidate available for Agent Selection: case={case_id} target={target_name}")
    agent_args = build_agent_args(args, profile, target_name)
    responses: List[Dict[str, Any]] = []
    checked_slice_ids: List[int] = []
    for row in ranked_slices:
        sid = int(row["slice_id"])
        checked_slice_ids.append(sid)
        response = ask_agent_slice_complete(
            slice_2d=np.asarray(volume[sid], dtype=np.float32),
            args=agent_args,
            reference_examples=[],
            candidate_mask_2d=None,
        )
        answer = str(response.get("answer", "no")).strip().lower()
        item = {
            "slice_id": int(sid),
            "rank": int(row.get("rank", len(checked_slice_ids))),
            "score": float(row.get("score", 0.0)),
            "answer": answer,
            "raw_text": str(response.get("raw_text", ""))[:200],
            "attempts": int(response.get("attempts", 1)),
        }
        responses.append(item)
        print(
            f"[Agent Selection][{case_id}][{target_name}] "
            f"rank={item['rank']} slice={sid} score={item['score']:.4f} answer={answer}",
            flush=True,
        )
        if answer == "yes":
            return {
                "selected_slice_id": int(sid),
                "selected_rank": int(item["rank"]),
                "selected_score": float(item["score"]),
                "agent_checked_slice_ids": checked_slice_ids,
                "agent_pass_slice_ids": [int(sid)],
                "agent_pass_count": 1,
                "agent_responses": responses,
            }
    raise RuntimeError(
        f"Agent rejected all Top-{len(ranked_slices)} candidate slices: "
        f"case={case_id} target={target_name} checked_slice_ids={checked_slice_ids}"
    )


def mean_std(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([float(v) for v in values], dtype=np.float32)
    if arr.size <= 0:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _finite_percentile(values: np.ndarray, q: float, default: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float(default)
    return float(np.percentile(finite, float(q)))


def robust_normalize(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(arr)
    if int(np.sum(finite)) <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = _finite_percentile(arr[finite], 1.0, 0.0)
    hi = _finite_percentile(arr[finite], 99.0, 1.0)
    if hi <= lo + EPS:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo + EPS), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 2 or float(sigma) <= 0.0:
        return arr.astype(np.float32)
    try:
        from scipy import ndimage as ndi

        return ndi.gaussian_filter1d(arr, sigma=float(sigma), mode="nearest").astype(np.float32)
    except Exception:
        radius = int(max(1, round(3.0 * float(sigma))))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * np.square(x / float(sigma)))
        kernel = kernel / max(float(np.sum(kernel)), EPS)
        padded = np.pad(arr, (radius, radius), mode="edge")
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def scale01(values: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if int(np.sum(finite)) <= 0:
        return out
    vals = arr[finite]
    lo = float(np.percentile(vals, float(q_low)))
    hi = float(np.percentile(vals, float(q_high)))
    if hi <= lo + EPS:
        return out
    out[finite] = np.clip((vals - lo) / (hi - lo + EPS), 0.0, 1.0)
    return out.astype(np.float32)


def otsu_threshold01(values: np.ndarray, bins: int = 128) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return 0.5
    hist, edges = np.histogram(np.clip(arr, 0.0, 1.0), bins=int(bins), range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = float(np.sum(hist))
    if total <= EPS:
        return 0.5
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, EPS)
    mean_fg = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(weight_fg, EPS)
    between = weight_bg * weight_fg * np.square(mean_bg - mean_fg)
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not bool(np.any(valid)):
        return float(np.median(arr))
    return float(np.clip(centers[int(np.argmax(np.where(valid, between, -1.0)))], 0.0, 1.0))


def foreground_support_profile(norm: np.ndarray) -> np.ndarray:
    arr = np.asarray(norm, dtype=np.float32)
    nonzero = arr[np.abs(arr) > EPS]
    if nonzero.size <= 0:
        return np.zeros((arr.shape[0],), dtype=np.float32)
    threshold = otsu_threshold01(nonzero)
    support = arr > float(threshold)
    return np.mean(support.reshape(support.shape[0], -1), axis=1).astype(np.float32)


def estimate_adjacent_transition_visibility(volume: np.ndarray, topk: int) -> TransitionEstimate:
    arr = robust_normalize(volume)
    if arr.ndim != 3:
        raise ValueError(f"volume must be [D,H,W], got {arr.shape}")
    d = int(arr.shape[0])
    if d <= 2:
        raise ValueError("volume must contain at least three slices.")

    diff = np.zeros((d,), dtype=np.float32)
    between = np.mean(np.abs(arr[1:] - arr[:-1]).reshape(d - 1, -1), axis=1).astype(np.float32)
    diff[0] = between[0]
    diff[-1] = between[-1]
    diff[1:-1] = 0.5 * (between[:-1] + between[1:])
    transition = scale01(smooth_1d(diff, sigma=max(1.0, 0.01 * float(d))), 5.0, 95.0)

    support = scale01(smooth_1d(foreground_support_profile(arr), sigma=max(1.0, 0.015 * float(d))), 5.0, 95.0)
    boundary_mass = transition / max(float(np.sum(transition)), EPS)
    coord = (np.cumsum(boundary_mass) - 0.5 * boundary_mass).astype(np.float32)
    interior = np.sqrt(np.clip(np.sin(np.pi * np.clip(coord, 0.0, 1.0)), 0.0, 1.0)).astype(np.float32)

    # A visible foreground slice should be inside the transition bracket:
    # not on the abrupt boundary itself, but between the two boundary bands.
    boundary_penalty = np.sqrt(np.clip(1.0 - transition, 0.0, 1.0)).astype(np.float32)
    score = (support * interior * boundary_penalty).astype(np.float32)
    order = np.argsort(-score, kind="mergesort")

    rows: List[Dict[str, Any]] = []
    for rank, sid in enumerate(order[: max(1, int(topk))], start=1):
        sid = int(sid)
        rows.append(
            {
                "rank": int(rank),
                "slice_id": sid,
                "score": float(score[sid]),
                "support": float(support[sid]),
                "transition": float(transition[sid]),
                "interior": float(interior[sid]),
            }
        )
    return TransitionEstimate(
        ranked_slices=rows,
        score_profile=score,
        transition_profile=transition,
        interior_profile=interior,
        support_profile=support,
        method="adjacent_transition_foreground_visibility",
    )


def build_pam(args: argparse.Namespace) -> Any:
    if bool(int(args.disable_cv2_resize)):
        pam_tools.HAS_CV2_RESIZE = False
    selector_args = build_selector_config(
        device=args.device,
        pam_root=args.pam_root,
        pam_ckpt=args.propmask_ckpt,
        pam_box2mask_ckpt=args.box2mask_ckpt if str(args.prompt_annotation_type) == "box" else None,
        prompt_annotation_type=args.prompt_annotation_type,
        seed=args.seed,
        eval_prob_thresh=args.eval_prob_thresh,
        final_core_support_thresh=args.final_core_support_thresh,
        final_candidate_support_thresh=args.final_candidate_support_thresh,
        final_keep_lcc=args.final_keep_lcc,
    )
    selector_args.dataset = args.dataset
    selector_args.pam_dynamic_crop_ratio = float(DEFAULT_SLICE_DYNAMIC_CROP_RATIO)
    selector_args.pam_prop_batch_size = int(args.prop_batch_size)
    selector_args.pam_emb_batch_size = int(args.emb_batch_size)
    selector_args.pam_prompt_batch_size = int(args.prompt_batch_size)
    selector_args.pam_modality = resolve_dataset_config(dataset=args.dataset, dataset_root=args.dataset_root).pam_modality
    return build_pam_backend(selector_args)


def evaluate_pam_for_selected_slice(
    *,
    args: argparse.Namespace,
    pam: Any,
    volume: np.ndarray,
    gt: np.ndarray,
    sid: int,
) -> Dict[str, Any]:
    prompt_source = np.asarray(gt[sid], dtype=np.uint8) if 0 <= sid < gt.shape[0] else np.zeros(gt.shape[1:], dtype=np.uint8)
    prompt_area = int(np.sum(prompt_source))
    if prompt_area <= 0:
        return {
            "pam_status": "empty_selected_slice_prompt",
            "pam_dice": None,
            "prompt_gt_area": 0,
            "prompt_mask_area": 0,
            "pred_voxels": 0,
        }
    try:
        prompt_mask, prompt_meta = build_prompt_mask_from_annotation(
            volume=volume,
            prompt_slice_id=int(sid),
            label_slice_2d=prompt_source,
            pam=pam,
            prompt_annotation_type=str(args.prompt_annotation_type),
            prob_thresh=float(args.eval_prob_thresh),
        )
        ensemble = run_final_prompt_ensemble(
            volume=volume,
            pam=pam,
            prompt_slice_id=int(sid),
            prompt_mask_2d=np.asarray(prompt_mask, dtype=bool),
            prob_thresh=float(args.eval_prob_thresh),
            core_support_thresh=float(args.final_core_support_thresh),
            candidate_support_thresh=float(args.final_candidate_support_thresh),
            keep_lcc=bool(int(args.final_keep_lcc)),
            seed=int(args.seed),
            perturbation_schedule=["original"],
            anchor_to_prompt=True,
        )
        pred = np.asarray(ensemble["final_mask"], dtype=bool)
        return {
            "pam_status": "ok",
            "pam_dice": float(dice_coefficient(pred.astype(np.uint8), gt.astype(np.uint8))),
            "prompt_gt_area": int(prompt_area),
            "prompt_mask_area": int(np.sum(np.asarray(prompt_mask, dtype=bool))),
            "prompt_source": str(prompt_meta.get("prompt_source", args.prompt_annotation_type)),
            "pred_voxels": int(np.sum(pred)),
        }
    except Exception as exc:
        return {
            "pam_status": "pam_error",
            "pam_error": f"{type(exc).__name__}: {exc}",
            "pam_dice": None,
            "prompt_gt_area": int(prompt_area),
            "prompt_mask_area": 0,
            "pred_voxels": 0,
        }


def summarize_rows(rows: Sequence[Dict[str, Any]], target_names: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for target_name in target_names:
        target_rows = [row for row in rows if row["target_name"] == str(target_name)]
        present = [row for row in target_rows if bool(row["target_present"])]
        attempted = [row for row in present if row["selection_status"] == "ok"]
        valid = [row for row in attempted if row.get("pam_status") == "ok"]
        selection_aware = [
            float(row.get("pam_dice", 0.0)) if row.get("pam_status") == "ok" else 0.0
            for row in attempted
        ]
        payload: Dict[str, Any] = {
            "case_count": int(len(target_rows)),
            "present_case_count": int(len(present)),
            "selected_case_count": int(len(attempted)),
            "empty_prompt_cases": int(sum(row.get("pam_status") == "empty_selected_slice_prompt" for row in attempted)),
            "pam_error_cases": int(sum(row.get("pam_status") == "pam_error" for row in attempted)),
        }
        for k in (1, 3, 5, 10, 16):
            vals = [float(row[f"top{k}_hit"]) for row in attempted if row.get(f"top{k}_hit") is not None]
            payload[f"top{k}_hit_mean"] = float(np.mean(vals)) if vals else None
        for metric in ("top1_area_ratio_to_max", "best_rank", "selected_slice_distance_to_target_center"):
            vals = [float(row[metric]) for row in attempted if row.get(metric) is not None]
            payload[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            payload[f"{metric}_std"] = float(np.std(vals)) if vals else None
        payload["valid_pam_dice"] = mean_std([float(row["pam_dice"]) for row in valid])
        payload["selection_aware_pam_dice"] = mean_std(selection_aware)
        payload["prompt_gt_area"] = mean_std([float(row["prompt_gt_area"]) for row in valid])
        summary[str(target_name)] = payload
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    profile = get_target_profile(args.dataset)
    target_names = normalize_target_names(args.target_names, profile.branch_source_label_ids.keys())
    if bool(int(args.use_agent_selection)):
        ensure_agent_api_key(args)
    cfg = resolve_dataset_config(dataset=args.dataset, dataset_root=args.dataset_root)
    all_cases, missing = collect_cases(cfg)
    case_lookup = {str(cid).lower(): (str(cid), img, lab) for cid, img, lab in all_cases}
    run_config_path = resolve_split_config_path(dataset=args.dataset, run_config=args.run_config)
    if run_config_path is None:
        raise RuntimeError(f"No split config path available for dataset={args.dataset!r}.")
    case_ids = [str(v).strip() for v in args.case_ids if str(v).strip()] if args.case_ids else load_split_case_ids(run_config_path, args.split)
    if int(args.max_cases) > 0:
        case_ids = case_ids[: int(args.max_cases)]

    rows: List[Dict[str, Any]] = []
    for idx, case_id in enumerate(case_ids, start=1):
        if case_id.lower() not in case_lookup:
            raise RuntimeError(f"Missing case in dataset: split={args.split} case={case_id}")
        _, image_path, label_path = case_lookup[case_id.lower()]
        print(f"[Adjacent Transition SE][{args.dataset}][{args.split}][{idx}/{len(case_ids)}] {case_id}", flush=True)
        volume = np.asarray(load_volume_any(image_path), dtype=np.float32)
        label = np.asarray(load_volume_any(label_path), dtype=np.int32)
        estimate = estimate_adjacent_transition_visibility(volume, topk=max(int(args.topk), 16))
        ranked_ids = [int(item["slice_id"]) for item in estimate.ranked_slices]

        for target_name in target_names:
            source_ids = tuple(int(v) for v in profile.branch_source_label_ids[str(target_name)])
            gt = target_mask_from_label(label, source_ids)
            gt_area = np.sum(gt.reshape(gt.shape[0], -1), axis=1).astype(np.int64)
            positive_ids = np.where(gt_area > 0)[0].astype(int)
            present = bool(positive_ids.size > 0)
            base: Dict[str, Any] = {
                "case_id": str(case_id),
                "target_name": str(target_name),
                "source_label_ids": list(source_ids),
                "target_present": bool(present),
                "gt_voxels": int(np.sum(gt)),
                "image_path": str(image_path),
                "label_path": str(label_path),
                "selection_status": "not_run",
            }
            if not present:
                base.update(
                    {
                        "selection_status": "target_absent",
                        "selected_slice_id": -1,
                        "selected_score": None,
                        "pam_status": "target_absent",
                        "pam_dice": None,
                    }
                )
                rows.append(base)
                continue
            if bool(int(args.use_agent_selection)):
                agent_pack = run_target_agent_selection(
                    volume=volume,
                    ranked_slices=estimate.ranked_slices[: max(1, int(args.topk))],
                    args=args,
                    profile=profile,
                    target_name=str(target_name),
                    case_id=str(case_id),
                )
                sid = int(agent_pack["selected_slice_id"])
                selected_rank = int(agent_pack["selected_rank"])
                selected_score = float(agent_pack["selected_score"])
                selected_row = next(
                    (item for item in estimate.ranked_slices if int(item["slice_id"]) == sid),
                    {},
                )
            else:
                agent_pack = {
                    "agent_checked_slice_ids": [],
                    "agent_pass_slice_ids": [],
                    "agent_pass_count": 0,
                    "agent_responses": [],
                }
                sid = int(ranked_ids[0]) if ranked_ids else -1
                selected_rank = 1 if sid >= 0 else -1
                selected_row = estimate.ranked_slices[0] if estimate.ranked_slices else {}
                selected_score = float(selected_row.get("score", 0.0))
            positive_set = {int(v) for v in positive_ids.tolist()}
            max_area = int(np.max(gt_area)) if gt_area.size > 0 else 0
            target_center = float(
                np.sum(np.arange(gt_area.size, dtype=np.float32) * gt_area) / max(float(np.sum(gt_area)), 1.0)
            )
            best_rank = None
            for rank, ranked_sid in enumerate(ranked_ids, start=1):
                if int(ranked_sid) in positive_set:
                    best_rank = int(rank)
                    break
            top_row = estimate.ranked_slices[0] if estimate.ranked_slices else {}
            base.update(
                {
                    "selection_status": "ok",
                    "selection_method": "agent_selection" if bool(int(args.use_agent_selection)) else "top1",
                    "selected_slice_id": int(sid),
                    "selected_rank": int(selected_rank),
                    "selected_score": float(selected_score),
                    "selected_support": float(selected_row.get("support", top_row.get("support", 0.0))),
                    "selected_transition": float(selected_row.get("transition", top_row.get("transition", 0.0))),
                    "selected_interior": float(selected_row.get("interior", top_row.get("interior", 0.0))),
                    "agent_checked_slice_ids": list(agent_pack.get("agent_checked_slice_ids", [])),
                    "agent_pass_slice_ids": list(agent_pack.get("agent_pass_slice_ids", [])),
                    "agent_pass_count": int(agent_pack.get("agent_pass_count", 0)),
                    "agent_responses": list(agent_pack.get("agent_responses", [])),
                    "target_positive_slice_count": int(positive_ids.size),
                    "target_max_slice_area": int(max_area),
                    "top1_area": int(gt_area[sid]) if 0 <= sid < gt_area.size else 0,
                    "top1_area_ratio_to_max": None if max_area <= 0 or sid < 0 else float(gt_area[sid] / max(float(max_area), 1.0)),
                    "target_center_z": float(target_center),
                    "selected_slice_distance_to_target_center": abs(float(sid) - target_center) if sid >= 0 else None,
                    "best_rank": best_rank,
                }
            )
            for k in (1, 3, 5, 10, 16):
                base[f"top{k}_hit"] = bool(any(int(ranked_sid) in positive_set for ranked_sid in ranked_ids[:k]))
            base.update({"pam_status": "not_run", "pam_dice": None, "prompt_gt_area": None})
            rows.append(base)
            print(
                f"  {target_name}: slice={sid} top1_gt_area={base['top1_area']} "
                f"best_rank={best_rank} pam_dice=NA",
                flush=True,
            )

    if bool(int(args.eval_pam)):
        pam_args = argparse.Namespace(**vars(args))
        pam_args.device = str(args.pam_device)
        pam = build_pam(pam_args)
        volume_cache: Dict[str, np.ndarray] = {}
        label_cache: Dict[str, np.ndarray] = {}
        for row in rows:
            if row.get("selection_status") != "ok" or not bool(row.get("target_present")):
                continue
            case_id = str(row["case_id"])
            image_path = str(row["image_path"])
            label_path = str(row["label_path"])
            if case_id not in volume_cache:
                volume_cache[case_id] = np.asarray(load_volume_any(Path(image_path)), dtype=np.float32)
                label_cache[case_id] = np.asarray(load_volume_any(Path(label_path)), dtype=np.int32)
            source_ids = tuple(int(v) for v in row.get("source_label_ids", []))
            gt = target_mask_from_label(label_cache[case_id], source_ids)
            row.update(
                evaluate_pam_for_selected_slice(
                    args=args,
                    pam=pam,
                    volume=volume_cache[case_id],
                    gt=gt,
                    sid=int(row.get("selected_slice_id", -1)),
                )
            )
            dice_text = "NA" if row.get("pam_dice") is None else f"{float(row['pam_dice']):.4f}"
            print(
                f"[PAM Eval][{args.dataset}][{case_id}] {row['target_name']}: "
                f"slice={row.get('selected_slice_id')} pam_dice={dice_text}",
                flush=True,
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "split": str(args.split),
        "case_count": int(len(case_ids)),
        "target_names": list(target_names),
        "method": (
            "adjacent_transition_foreground_visibility_agent_selection"
            if bool(int(args.use_agent_selection))
            else "adjacent_transition_foreground_visibility_top1"
        ),
        "leakage_contract": (
            "Slice estimation uses only raw image adjacency changes; "
            "GT prompt is used only after the selected slice is fixed."
        ),
        "algorithm_contract": (
            "Abrupt adjacent-slice changes are treated as foreground/background transition bands. "
            "Visible foreground is modeled as the stable interior between transition bands, weighted by image support. "
            "When Agent Selection is enabled, the Top-k queue is reviewed in order with the raw 2D slice "
            "and target-specific question only; no mask, box, overlay, PAM output, 3D GT, or fallback is used."
        ),
        "use_agent_selection": bool(int(args.use_agent_selection)),
        "agent_selection": {
            "model_provider": str(args.agent_model_provider),
            "model": str(args.agent_model),
            "reasoning_effort": str(args.agent_reasoning_effort),
            "wire_api": str(args.agent_wire_api),
            "topk": int(args.topk),
            "visual_prompt": False,
            "reference_examples": False,
            "no_fallback": True,
        },
        "prompt_annotation_type": str(args.prompt_annotation_type),
        "run_config": str(run_config_path),
        "missing_labels": list(missing),
        "rows": rows,
        "summary": summarize_rows(rows, target_names),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()

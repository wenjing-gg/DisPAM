#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import collect_cases, load_split_case_ids, resolve_dataset_config  # noqa: E402
from StudentBank.student_model_registry_v2 import load_single_target_student_model  # noqa: E402
from tools.profiles import get_target_profile  # noqa: E402
from tools.metrics import binary_metric_dict, summarize_metric_rows  # noqa: E402
from tools.data import crop_or_pad_zyx, model_hwd_to_zyx, normalize_volume, zyx_to_model_tensor  # noqa: E402
from tools.volumes import infer_nifti_depth_axis, load_volume_any  # noqa: E402

try:
    import nibabel as nib
except Exception:  # pragma: no cover
    nib = None

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def nifti_from_zyx(arr: np.ndarray, reference_path: Path, *, dtype: np.dtype) -> Any:
    if nib is None:  # pragma: no cover
        raise RuntimeError("nibabel is required for NIfTI export.")
    ref = nib.load(str(reference_path))
    data = np.asarray(arr, dtype=dtype)
    depth_axis = infer_nifti_depth_axis(tuple(int(v) for v in ref.shape[:3]))
    out = np.moveaxis(data, 0, depth_axis)
    header = ref.header.copy()
    header.set_data_dtype(np.dtype(dtype))
    return nib.Nifti1Image(out.astype(dtype), ref.affine, header)


def save_nifti_zyx(arr: np.ndarray, reference_path: Path, output_path: Path, *, dtype: np.dtype) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_from_zyx(arr, reference_path, dtype=dtype), str(output_path))


def rss_mb() -> float | None:
    if psutil is None:
        return None
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))


def maxrss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return float(value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0)


def cuda_peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def machine_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_count": int(os.cpu_count() or 0),
        "ram_total_gb": None,
    }
    if psutil is not None:
        info["ram_total_gb"] = float(psutil.virtual_memory().total / (1024.0**3))
    if device.type == "cuda":
        idx = int(device.index or torch.cuda.current_device())
        prop = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_device_index": idx,
                "cuda_device_name": prop.name,
                "cuda_total_memory_gb": float(prop.total_memory / (1024.0**3)),
                "cuda_capability": f"{prop.major}.{prop.minor}",
            }
        )
    return info


def parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    total = int(sum(int(p.numel()) for p in model.parameters()))
    trainable = int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))
    return {"parameters": total, "trainable_parameters": trainable}


def estimate_flops(model: torch.nn.Module, sample: torch.Tensor) -> Dict[str, Any]:
    flops = 0
    hooks = []

    def conv_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        if not isinstance(output, torch.Tensor):
            return
        batch = int(output.shape[0])
        out_channels = int(output.shape[1])
        out_spatial = int(np.prod(output.shape[2:]))
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            kernel_ops = int(np.prod(module.kernel_size)) * int(module.in_channels // module.groups)
        elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            kernel_ops = int(np.prod(module.kernel_size)) * int(module.in_channels // module.groups)
        else:
            return
        flops += int(batch * out_channels * out_spatial * (2 * kernel_ops + int(module.bias is not None)))

    def linear_hook(module: nn.Linear, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        if isinstance(output, torch.Tensor):
            flops += int(output.numel() * (2 * int(module.in_features) + int(module.bias is not None)))

    def norm_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        if isinstance(output, torch.Tensor):
            flops += int(4 * output.numel())

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(norm_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(sample)
    if was_training:
        model.train()
    for hook in hooks:
        hook.remove()
    return {
        "flops_estimated": int(flops),
        "gflops_estimated": float(flops / 1e9),
        "flops_method": "forward-hook estimate for Conv/ConvTranspose/Linear/Norm layers",
        "flops_input_shape": [int(v) for v in sample.shape],
    }


def optimizer_metadata(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    state = checkpoint.get("optimizer_state_dict")
    if not isinstance(state, Mapping):
        return {"optimizer_type": "AdamW", "optimizer_state_available": False}
    groups = []
    for group in state.get("param_groups", []):
        if not isinstance(group, Mapping):
            continue
        groups.append(
            {
                "lr": group.get("lr"),
                "betas": list(group.get("betas", [])) if isinstance(group.get("betas"), tuple) else group.get("betas"),
                "eps": group.get("eps"),
                "weight_decay": group.get("weight_decay"),
                "amsgrad": group.get("amsgrad"),
            }
        )
    return {"optimizer_type": "AdamW", "optimizer_state_available": True, "optimizer_param_groups": groups}


def checkpoint_metadata(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    args = dict(payload.get("args", {}))
    meta: Dict[str, Any] = {
        "checkpoint": str(path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_best_val_dice": payload.get("best_val_dice", payload.get("best_score")),
        "train_args": args,
        "train_epoch_count_config": args.get("epochs"),
        "train_lr": args.get("lr"),
        "train_weight_decay": args.get("weight_decay"),
        "train_batch_size": args.get("batch_size"),
        "train_num_workers": args.get("num_workers"),
        "train_seed": args.get("seed"),
        "train_run_config": args.get("run_config"),
        "train_dataset": args.get("dataset"),
        "train_1epoch_seconds": None,
        "train_1epoch_timing_note": "not_recorded_in_checkpoint",
    }
    meta.update(optimizer_metadata(payload))
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-student sliding-window probability voting without Teacher Cache foreground masking.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--target", action="append", default=[], help="Optional target filter, e.g. tumor.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--student", action="append", default=[], help="student_model=checkpoint.pth. May repeat.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--sliding-overlap", type=float, default=0.5)
    parser.add_argument("--vote-thresh", type=float, default=0.5)
    parser.add_argument("--min-votes", type=int, default=0, help="Default 0 means majority vote.")
    parser.add_argument("--out-dir", default="outputs/vote_eval")
    parser.add_argument("--save-case-artifacts", action="store_true")
    parser.add_argument("--artifact-root", default="", help="Default: <out-dir>/cases. Saves prob.nii.gz, mask.nii.gz, gt.nii.gz, metrics.json.")
    parser.add_argument("--no-profile", action="store_true", help="Disable model complexity/memory profiling.")
    return parser.parse_args()


def parse_student_specs(values: Sequence[str]) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid --student {text!r}; expected model=path.")
        name, path = text.split("=", 1)
        specs.append((name.strip(), Path(path.strip())))
    if not specs:
        raise RuntimeError("At least one --student model=checkpoint.pth is required for voting evaluation.")
    return specs


def case_lookup(dataset: str, dataset_root: str) -> Dict[str, Tuple[Path, Path]]:
    cfg = resolve_dataset_config(dataset=dataset, dataset_root=dataset_root)
    cases, _ = collect_cases(cfg)
    return {str(case_id): (Path(image_path), Path(label_path)) for case_id, image_path, label_path in cases}


def target_specs(dataset: str, raw_targets: Sequence[str]) -> List[Tuple[str, int, Tuple[int, ...]]]:
    profile = get_target_profile(dataset)
    requested = {str(v).strip() for raw in raw_targets for v in str(raw).split(",") if str(v).strip()}
    specs: List[Tuple[str, int, Tuple[int, ...]]] = []
    for target in profile.branch_priority:
        if requested and target not in requested:
            continue
        specs.append(
            (
                str(target),
                int(profile.branch_export_label_ids[target]),
                tuple(int(v) for v in profile.branch_source_label_ids[target]),
            )
        )
    if not specs:
        raise RuntimeError(f"No target selected for dataset={dataset!r}; requested={sorted(requested)}")
    return specs


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def sliding_starts(length: int, patch: int, stride: int) -> List[int]:
    if int(length) <= int(patch):
        return [0]
    starts = list(range(0, int(length) - int(patch) + 1, max(int(stride), 1)))
    last = int(length) - int(patch)
    if starts[-1] != last:
        starts.append(last)
    return starts


def student_sliding_window_prob(
    *,
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: Sequence[int],
    overlap: float,
    device: torch.device,
    amp: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    patch = tuple(int(v) for v in patch_size)
    stride = tuple(max(1, int(round(float(v) * (1.0 - float(overlap))))) for v in patch)
    starts_z = sliding_starts(int(volume.shape[0]), patch[0], stride[0])
    starts_y = sliding_starts(int(volume.shape[1]), patch[1], stride[1])
    starts_x = sliding_starts(int(volume.shape[2]), patch[2], stride[2])
    prob_sum = np.zeros_like(volume, dtype=np.float32)
    count = np.zeros_like(volume, dtype=np.float32)
    forward_seconds = 0.0
    window_count = 0
    first_input_shape: List[int] | None = None
    first_logits_shape: List[int] | None = None
    for z in starts_z:
        for y in starts_y:
            for x in starts_x:
                patch_arr = crop_or_pad_zyx(volume, (z, y, x), patch, pad_value=0.0)
                inputs = zyx_to_model_tensor(patch_arr).unsqueeze(0).to(device=device, dtype=torch.float32)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                    logits = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_seconds += float(time.perf_counter() - start_time)
                logits_zyx = model_hwd_to_zyx(logits[0, 0].detach().float().cpu().numpy())
                prob_patch = sigmoid_np(logits_zyx)
                z1, y1, x1 = min(z + patch[0], volume.shape[0]), min(y + patch[1], volume.shape[1]), min(x + patch[2], volume.shape[2])
                valid_shape = (z1 - z, y1 - y, x1 - x)
                prob_sum[z:z1, y:y1, x:x1] += prob_patch[: valid_shape[0], : valid_shape[1], : valid_shape[2]]
                count[z:z1, y:y1, x:x1] += 1.0
                window_count += 1
                if first_input_shape is None:
                    first_input_shape = [int(v) for v in inputs.shape]
                    first_logits_shape = [int(v) for v in logits.shape]
    prob = prob_sum / np.maximum(count, 1.0)
    return prob, {
        "model_forward_seconds": forward_seconds,
        "input_shape": first_input_shape,
        "logits_shape": first_logits_shape,
        "patch_shape": [int(v) for v in patch],
        "volume_shape": [int(v) for v in volume.shape],
        "sliding_overlap": float(overlap),
        "sliding_stride": [int(v) for v in stride],
        "sliding_window_count": int(window_count),
    }


def vote_prediction(probs: Sequence[np.ndarray], vote_thresh: float, min_votes: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    prob_stack = np.stack([np.asarray(prob, dtype=np.float32) for prob in probs], axis=0)
    vote_mask = prob_stack >= float(vote_thresh)
    vote_count = np.sum(vote_mask, axis=0).astype(np.int16)
    required_votes = int(min_votes) if int(min_votes) > 0 else int(np.floor(len(probs) / 2) + 1)
    pred = vote_count >= int(required_votes)
    mean_prob = np.mean(prob_stack, axis=0).astype(np.float32)
    return pred, mean_prob, {"required_votes": int(required_votes), "student_count": int(len(probs)), "vote_thresh": float(vote_thresh)}


def load_students(
    specs: Sequence[Tuple[str, Path]],
    target_label_id: int,
    patch_size: Sequence[int],
    device: torch.device,
    no_profile: bool,
) -> Tuple[List[Tuple[str, Path, torch.nn.Module]], List[Dict[str, Any]]]:
    models: List[Tuple[str, Path, torch.nn.Module]] = []
    metadata: List[Dict[str, Any]] = []
    for model_name, ckpt_path in specs:
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        model, _, loaded_name = load_single_target_student_model(
            checkpoint_path=ckpt_path,
            target_label_id=int(target_label_id),
            device=device,
            student_model=model_name,
            fallback_patch_size=tuple(int(v) for v in patch_size),
        )
        meta = checkpoint_metadata(ckpt_path)
        meta.update(parameter_count(model))
        if not bool(no_profile):
            sample = torch.zeros((1, 1, int(patch_size[1]), int(patch_size[2]), int(patch_size[0])), device=device)
            try:
                meta.update(estimate_flops(model, sample))
            finally:
                del sample
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        metadata.append(meta)
        models.append((loaded_name, ckpt_path, model))
    return models, metadata


def evaluate_target(
    *,
    args: argparse.Namespace,
    dataset: str,
    lookup: Mapping[str, Tuple[Path, Path]],
    case_ids: Sequence[str],
    target: str,
    target_label_id: int,
    source_label_ids: Sequence[int],
    student_specs: Sequence[Tuple[str, Path]],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    models, model_meta = load_students(
        specs=student_specs,
        target_label_id=int(target_label_id),
        patch_size=args.patch_size,
        device=device,
        no_profile=bool(args.no_profile),
    )
    rows: List[Dict[str, Any]] = []
    artifact_root = Path(args.artifact_root) if str(args.artifact_root).strip() else Path(args.out_dir) / "cases"
    target_start = time.perf_counter()
    target_forward_seconds: List[float] = []
    target_case_seconds: List[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for case_id in tqdm(case_ids, desc=f"{dataset}-{target}", unit="case", dynamic_ncols=True):
        case_start = time.perf_counter()
        rss_before = rss_mb()
        image_path, label_path = lookup[str(case_id)]
        volume = normalize_volume(np.asarray(load_volume_any(image_path), dtype=np.float32))
        label = np.asarray(load_volume_any(label_path), dtype=np.int32)
        gt = np.isin(label, [int(v) for v in source_label_ids])
        probs: List[np.ndarray] = []
        infer_infos: List[Dict[str, Any]] = []
        for model_name, _, model in models:
            prob, infer_info = student_sliding_window_prob(
                model=model,
                volume=volume,
                patch_size=args.patch_size,
                overlap=float(args.sliding_overlap),
                device=device,
                amp=bool(args.amp),
            )
            infer_info["student_model"] = str(model_name)
            probs.append(prob)
            infer_infos.append(infer_info)
            target_forward_seconds.append(float(infer_info["model_forward_seconds"]))
        pred, mean_prob, vote_info = vote_prediction(probs, vote_thresh=float(args.vote_thresh), min_votes=int(args.min_votes))
        metrics = binary_metric_dict(pred, gt, compute_hd95=True, compute_assd=True)
        case_seconds = float(time.perf_counter() - case_start)
        target_case_seconds.append(case_seconds)
        rss_after = rss_mb()
        case_dir = artifact_root / safe_name(dataset) / safe_name(target) / safe_name(str(case_id))
        artifact_paths: Dict[str, str] = {}
        if bool(args.save_case_artifacts):
            save_nifti_zyx(mean_prob, image_path, case_dir / "prob.nii.gz", dtype=np.float32)
            save_nifti_zyx(pred.astype(np.uint8), image_path, case_dir / "mask.nii.gz", dtype=np.uint8)
            save_nifti_zyx(gt.astype(np.uint8), image_path, case_dir / "gt.nii.gz", dtype=np.uint8)
            artifact_paths = {
                "prob_path": str(case_dir / "prob.nii.gz"),
                "mask_path": str(case_dir / "mask.nii.gz"),
                "gt_path": str(case_dir / "gt.nii.gz"),
            }
        row = {
            "dataset": dataset,
            "split": str(args.split),
            "case_id": str(case_id),
            "target": target,
            "mode": "multi_student_vote",
            "task_type": "segmentation",
            "student_count": int(len(models)),
            "student_models": ";".join(str(name) for name, _, _ in models),
            "student_checkpoints": ";".join(str(path) for _, path, _ in models),
            "required_votes": int(vote_info["required_votes"]),
            "vote_thresh": float(args.vote_thresh),
            "dice": metrics["dice"],
            "hd95": metrics["hd95"],
            "assd": metrics["assd"],
            "iou": metrics.get("iou", metrics["ji"]),
            "ji": metrics["ji"],
            "sen": metrics["sen"],
            "ppv": metrics["ppv"],
            "pred_voxels": int(pred.sum()),
            "gt_voxels": int(gt.sum()),
            "prob_mean": float(np.mean(mean_prob)),
            "prob_min": float(np.min(mean_prob)),
            "prob_max": float(np.max(mean_prob)),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "case_seconds": case_seconds,
            "model_forward_seconds": float(sum(item["model_forward_seconds"] for item in infer_infos)),
            "rss_mb_before": rss_before,
            "rss_mb_after": rss_after,
            "maxrss_mb_so_far": maxrss_mb(),
            "cuda_peak_allocated_mb_so_far": cuda_peak_mb(device),
            **artifact_paths,
        }
        rows.append(row)
        if bool(args.save_case_artifacts):
            metrics_payload = {**row, "metrics": metrics, "vote_info": vote_info, "inference": infer_infos, "artifacts": artifact_paths}
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
    for _, _, model in models:
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    profile = {
        "target_wall_seconds": float(time.perf_counter() - target_start),
        "mean_case_seconds": float(np.mean(target_case_seconds)) if target_case_seconds else None,
        "std_case_seconds": float(np.std(target_case_seconds)) if target_case_seconds else None,
        "mean_model_forward_seconds": float(np.mean(target_forward_seconds)) if target_forward_seconds else None,
        "std_model_forward_seconds": float(np.std(target_forward_seconds)) if target_forward_seconds else None,
        "peak_rss_mb": maxrss_mb(),
        "peak_cuda_allocated_mb": cuda_peak_mb(device),
        "model_metadata": model_meta,
        "artifact_root": str(artifact_root),
        "case_artifacts_saved": bool(args.save_case_artifacts),
    }
    return rows, profile


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "split", "case_id", "target", "mode", "task_type", "student_count", "required_votes", "vote_thresh",
        "dice", "hd95", "assd", "iou", "ji", "sen", "ppv", "pred_voxels", "gt_voxels",
        "prob_mean", "prob_min", "prob_max", "case_seconds", "model_forward_seconds", "rss_mb_before", "rss_mb_after",
        "maxrss_mb_so_far", "cuda_peak_allocated_mb_so_far", "student_models", "student_checkpoints", "image_path", "label_path",
        "prob_path", "mask_path", "gt_path",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    fieldnames.extend(extra)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    lookup = case_lookup(str(args.dataset), str(args.dataset_root))
    train_case_ids = [str(v) for v in load_split_case_ids(Path(args.run_config), "train")]
    case_ids = [str(v) for v in load_split_case_ids(Path(args.run_config), str(args.split))]
    if int(args.max_cases) > 0:
        case_ids = case_ids[: int(args.max_cases)]
    specs = parse_student_specs(args.student)
    out_dir = Path(args.out_dir)
    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "split": str(args.split),
        "mode": "multi_student_vote",
        "task_type": "segmentation",
        "eval_args": vars(args),
        "machine": machine_info(device),
        "data_order": {"train_case_ids": train_case_ids, "eval_case_ids": case_ids, "eval_split": str(args.split), "max_cases": int(args.max_cases)},
        "targets": {},
    }
    for target, target_label_id, source_label_ids in target_specs(str(args.dataset), args.target):
        rows, profile = evaluate_target(
            args=args,
            dataset=str(args.dataset),
            lookup=lookup,
            case_ids=case_ids,
            target=target,
            target_label_id=int(target_label_id),
            source_label_ids=source_label_ids,
            student_specs=specs,
            device=device,
        )
        all_rows.extend(rows)
        target_summary = summarize_metric_rows(rows)
        summary["targets"][target] = {
            **target_summary,
            "student_count": int(len(specs)),
            "student_checkpoints": [str(path) for _, path in specs],
            "profile": profile,
        }
        print(
            f"[Vote] {args.dataset} {args.split} {target} students={len(specs)} n={target_summary['case_count']} "
            f"dice={target_summary['mean_dice']:.4f}±{target_summary['std_dice']:.4f}",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.split}_vote"
    write_csv(out_dir / f"{stem}_rows.csv", all_rows)
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Segmentation metrics used in the DisPAM experiments.

Paper mapping:
  Experiments / Comparison with State-of-the-Art Methods. DSC quantifies
  volumetric overlap, while HD95 and ASSD use bidirectional physical-space
  surface distances. Empty-mask conventions are defined in binary_metric_dict.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def _as_bool(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape={arr.shape}.")
    return arr


def binary_metric_dict(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    compute_hd95: bool = False,
    compute_assd: bool = False,
    spacing: Sequence[float] | None = None,
) -> Dict[str, float | None]:
    pred = _as_bool(pred_mask)
    gt = _as_bool(gt_mask)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    pred_count = int(np.sum(pred))
    gt_count = int(np.sum(gt))

    if pred_count + gt_count == 0:
        dice = 1.0
    else:
        dice = float((2.0 * tp) / max(pred_count + gt_count, 1))

    union = tp + fp + fn
    ji = 1.0 if union == 0 else float(tp / max(union, 1))
    sen = 1.0 if gt_count == 0 and pred_count == 0 else (0.0 if gt_count == 0 else float(tp / max(tp + fn, 1)))
    ppv = 1.0 if pred_count == 0 and gt_count == 0 else (0.0 if pred_count == 0 else float(tp / max(tp + fp, 1)))
    hd95 = hd95_distance(pred, gt, spacing=spacing) if bool(compute_hd95) else None
    assd = assd_distance(pred, gt, spacing=spacing) if bool(compute_assd) else None

    return {
        "dice": float(dice),
        "hd95": None if hd95 is None else float(hd95),
        "assd": None if assd is None else float(assd),
        "iou": float(ji),
        "ji": float(ji),
        "sen": float(sen),
        "ppv": float(ppv),
    }


def _surface_distances(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Sequence[float] | None = None,
) -> np.ndarray | None:
    pred = _as_bool(pred_mask)
    gt = _as_bool(gt_mask)
    pred_count = int(np.sum(pred))
    gt_count = int(np.sum(gt))
    if pred_count == 0 and gt_count == 0:
        return np.zeros((1,), dtype=np.float64)
    if pred_count == 0 or gt_count == 0:
        return None

    try:
        from scipy import ndimage as ndi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy.ndimage is required for HD95/ASSD surface metrics.") from exc

    union = pred | gt
    coords = np.argwhere(union)
    lo = np.maximum(coords.min(axis=0) - 2, 0)
    hi = np.minimum(coords.max(axis=0) + 3, np.asarray(pred.shape, dtype=np.int64))
    slices = tuple(slice(int(lo[axis]), int(hi[axis])) for axis in range(3))
    pred = pred[slices]
    gt = gt[slices]

    sampling = None if spacing is None else tuple(float(v) for v in spacing)
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    pred_surface = pred & ~ndi.binary_erosion(pred, structure=structure, border_value=0)
    gt_surface = gt & ~ndi.binary_erosion(gt, structure=structure, border_value=0)
    if not bool(np.any(pred_surface)) or not bool(np.any(gt_surface)):
        return None

    dt_gt = ndi.distance_transform_edt(~gt_surface, sampling=sampling)
    dt_pred = ndi.distance_transform_edt(~pred_surface, sampling=sampling)
    pred_to_gt = dt_gt[pred_surface]
    gt_to_pred = dt_pred[gt_surface]
    return np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float64, copy=False)


def hd95_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Sequence[float] | None = None,
) -> float | None:
    distances = _surface_distances(pred_mask, gt_mask, spacing=spacing)
    if distances is None:
        return None
    return float(np.percentile(distances, 95))


def assd_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Sequence[float] | None = None,
) -> float | None:
    distances = _surface_distances(pred_mask, gt_mask, spacing=spacing)
    if distances is None:
        return None
    return float(np.mean(distances))


def summarize_metric_rows(rows: Iterable[Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
    metrics = ("dice", "hd95", "assd", "ji", "sen", "ppv")
    prepared: List[Dict[str, Any]] = [dict(row) for row in rows]
    summary: Dict[str, Any] = {"case_count": int(len(prepared))}
    for metric in metrics:
        key = f"{prefix}{metric}" if prefix else metric
        vals = [float(row[key]) for row in prepared if key in row and row[key] is not None]
        summary[f"mean_{metric}"] = float(np.mean(vals)) if vals else None
        summary[f"std_{metric}"] = float(np.std(vals)) if vals else None
    return summary

"""Prompt-Free full-volume evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .components import sliding_window_probability
from .datasets import CaseRecord, target_label_ids
from .io import (
    load_case_image,
    load_spacing_any,
    load_volume_any,
    robust_normalize_channels,
)
from .metrics import binary_metric_dict, summarize_metric_rows


def evaluate_model(
    model: torch.nn.Module,
    cases: Sequence[CaseRecord],
    dataset: str,
    target: str,
    patch_size: Sequence[int],
    overlap: float,
    sw_batch_size: int,
    device: torch.device,
    amp: bool,
) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    source_ids = target_label_ids(dataset, target)
    for case in cases:
        image = robust_normalize_channels(load_case_image(case.image_paths))
        label = load_volume_any(case.label_path)
        probability, runtime = sliding_window_probability(
            model=model,
            image_czyx=image,
            patch_size=patch_size,
            overlap=overlap,
            sw_batch_size=sw_batch_size,
            device=device,
            amp=amp,
        )
        prediction = probability >= 0.5
        reference = np.isin(label, source_ids)
        metrics = binary_metric_dict(
            prediction,
            reference,
            compute_hd95=True,
            compute_assd=True,
            spacing=load_spacing_any(case.label_path),
        )
        rows.append({"case_id": case.case_id, **metrics, **runtime})
    mean_dice = float(np.mean([row["dice"] for row in rows])) if rows else 0.0
    return mean_dice, rows


def evaluate_ensemble(
    models: Sequence[torch.nn.Module],
    cases: Sequence[CaseRecord],
    dataset: str,
    target: str,
    patch_size: Sequence[int],
    overlap: float,
    sw_batch_size: int,
    device: torch.device,
    amp: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(models) != 4:
        raise ValueError("Prompt-Free inference requires the four-model target bank.")
    source_ids = target_label_ids(dataset, target)
    rows: List[Dict[str, Any]] = []
    for case in cases:
        image = robust_normalize_channels(load_case_image(case.image_paths))
        probabilities = [
            sliding_window_probability(
                model=model,
                image_czyx=image,
                patch_size=patch_size,
                overlap=overlap,
                sw_batch_size=sw_batch_size,
                device=device,
                amp=amp,
            )[0]
            for model in models
        ]
        prediction = np.mean(np.stack(probabilities, axis=0), axis=0) >= 0.5
        reference = np.isin(load_volume_any(case.label_path), source_ids)
        metrics = binary_metric_dict(
            prediction,
            reference,
            compute_hd95=True,
            compute_assd=True,
            spacing=load_spacing_any(case.label_path),
        )
        rows.append({"case_id": case.case_id, **metrics})
    return summarize_metric_rows(rows), rows

"""Aggregate case-level metrics across independent training runs.

Paper mapping:
  Experiments / Implementation Details. Each case is averaged across five
  independently trained models before the table mean and population standard
  deviation are computed across cases.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_METRICS: Tuple[str, ...] = (
    "dice",
    "iou",
    "sen",
    "ppv",
    "hd95",
    "assd",
)


def read_case_metrics(
    path: Path,
    case_column: str,
    metrics: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {case_column, *metrics}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} is missing columns {sorted(missing)}.")
        for row in reader:
            case_id = str(row[case_column]).strip()
            if not case_id:
                raise RuntimeError(f"{path} contains an empty case identifier.")
            if case_id in output:
                raise RuntimeError(f"{path} contains duplicate case {case_id!r}.")
            values: Dict[str, float] = {}
            for metric in metrics:
                raw_value = str(row[metric]).strip()
                if not raw_value:
                    raise RuntimeError(
                        f"{path} has no {metric} value for case {case_id!r}."
                    )
                value = float(raw_value)
                if not np.isfinite(value):
                    raise RuntimeError(
                        f"{path} has non-finite {metric} for case {case_id!r}."
                    )
                values[metric] = value
            output[case_id] = values
    if not output:
        raise RuntimeError(f"{path} contains no case rows.")
    return output


def aggregate_case_runs(
    runs: Sequence[Mapping[str, Mapping[str, float]]],
    metrics: Sequence[str],
) -> List[Dict[str, float | str]]:
    if not runs:
        raise ValueError("At least one run is required.")
    reference_cases = set(runs[0])
    for run_index, run in enumerate(runs[1:], start=1):
        current_cases = set(run)
        if current_cases != reference_cases:
            missing = sorted(reference_cases.difference(current_cases))
            extra = sorted(current_cases.difference(reference_cases))
            raise RuntimeError(
                f"Run {run_index} has mismatched cases; "
                f"missing={missing[:10]}, extra={extra[:10]}."
            )
    rows: List[Dict[str, float | str]] = []
    for case_id in sorted(reference_cases):
        row: Dict[str, float | str] = {"case_id": case_id}
        for metric in metrics:
            row[metric] = float(
                np.mean([float(run[case_id][metric]) for run in runs])
            )
        rows.append(row)
    return rows


def summarize_case_averages(
    rows: Sequence[Mapping[str, float | str]],
    metrics: Sequence[str],
    run_count: int,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "run_count": int(run_count),
        "case_count": int(len(rows)),
        "run_aggregation": "arithmetic mean per case",
        "case_summary": "mean and population standard deviation",
    }
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[f"mean_{metric}"] = float(np.mean(values))
        summary[f"std_{metric}"] = float(np.std(values, ddof=0))
    return summary

#!/usr/bin/env python3
"""Average each case across independent runs and summarize across cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dispam.aggregation import (
    DEFAULT_METRICS,
    aggregate_case_runs,
    read_case_metrics,
    summarize_case_averages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average matched case metrics across independent runs, then compute "
            "the mean and population standard deviation across cases."
        )
    )
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--metric", action="append", dest="metrics")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_case_csv(
    path: Path,
    rows: Sequence[dict[str, float | str]],
    metrics: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", *metrics])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if len(args.run) != 5:
        raise ValueError(f"Reported aggregation requires five runs, got {len(args.run)}.")
    metrics = tuple(args.metrics or DEFAULT_METRICS)
    runs = [
        read_case_metrics(path, args.case_column, metrics)
        for path in args.run
    ]
    rows = aggregate_case_runs(runs, metrics)
    summary = summarize_case_averages(rows, metrics, len(runs))
    summary["run_files"] = [str(path) for path in args.run]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_case_csv(args.output_dir / "cases.csv", rows, metrics)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

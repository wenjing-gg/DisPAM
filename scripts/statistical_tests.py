#!/usr/bin/env python3
"""Run paired case-level significance tests for DisPAM comparisons.

Paper mapping:
  Experiments / Implementation Details. The reported significance analysis uses
  a two-sided Wilcoxon signed-rank test on matched case-level metric values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict

from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched per-case metrics with a Wilcoxon signed-rank test."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--metric-column", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def read_metric(path: Path, case_column: str, metric_column: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {case_column, metric_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            case_id = str(row[case_column]).strip()
            raw_value = str(row[metric_column]).strip()
            if not case_id or not raw_value:
                continue
            value = float(raw_value)
            if math.isfinite(value):
                values[case_id] = value
    return values


def main() -> None:
    args = parse_args()
    baseline = read_metric(Path(args.baseline), args.case_column, args.metric_column)
    method = read_metric(Path(args.method), args.case_column, args.metric_column)
    case_ids = sorted(set(baseline).intersection(method))
    if not case_ids:
        raise RuntimeError("No matched cases were found.")
    baseline_values = [baseline[case_id] for case_id in case_ids]
    method_values = [method[case_id] for case_id in case_ids]
    differences = [
        method_value - baseline_value
        for baseline_value, method_value in zip(baseline_values, method_values)
    ]
    if all(abs(value) <= 1e-12 for value in differences):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            method_values,
            baseline_values,
            alternative="two-sided",
            zero_method="wilcox",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    payload = {
        "test": "two-sided Wilcoxon signed-rank",
        "metric": args.metric_column,
        "matched_case_count": len(case_ids),
        "statistic": statistic,
        "p_value": p_value,
        "case_ids": case_ids,
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()

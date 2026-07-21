#!/usr/bin/env python3
"""Validate public dataset layouts and emit an executable manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dispam.datasets import cases_for_split, serialize_case, validate_shape_count
from dispam.io import load_volume_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["kits19", "brats18", "lits"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--check-volumes", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "dataset": args.dataset,
        "dataset_root": str(args.dataset_root),
        "split_file": str(args.split_file),
        "splits": {},
    }
    for split in args.splits:
        cases = cases_for_split(
            args.dataset,
            args.dataset_root,
            args.split_file,
            split,
        )
        if args.max_cases > 0:
            cases = cases[: args.max_cases]
        rows = []
        for case in cases:
            row = serialize_case(case)
            if args.check_volumes:
                shapes = [
                    load_volume_any(path).shape
                    for path in (*case.image_paths, case.label_path)
                ]
                validate_shape_count(shapes, case.case_id)
                row["shape_zyx"] = list(shapes[0])
                row["channel_count"] = len(case.image_paths)
            rows.append(row)
        manifest["splits"][split] = {"case_count": len(rows), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

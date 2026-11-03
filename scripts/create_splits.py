#!/usr/bin/env python3
"""Create a fixed validation holdout while preserving the outer test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dispam.datasets import make_three_way_split


def create_splits(payload: dict, validation_fraction: float, seed: int) -> dict:
    def ids(name):
        return payload.get(f"{name}_case_ids", payload.get(name))

    development = ids("train")
    test = ids("test")
    if test is None:
        # The old KiTS split called its outer evaluation collection "val".
        test = ids("val")
    elif ids("val") is not None:
        development = list(development) + list(ids("val"))
    return make_three_way_split(development, test, validation_fraction, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    splits = create_splits(payload, args.validation_fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(splits[name]) for name in ("train", "val", "test")}))


if __name__ == "__main__":
    main()

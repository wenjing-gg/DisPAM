#!/usr/bin/env python3
"""Evaluate Prompt-Free target-specific four-model banks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dispam.datasets import cases_for_split, target_label_ids, split_protocol
from dispam.evaluation import evaluate_ensemble
from dispam.models import MODEL_NAMES, load_model_checkpoint
from dispam.reproducibility import set_seed


def _bank_specs(values: Sequence[str]) -> Dict[str, Dict[str, Path]]:
    banks: Dict[str, Dict[str, Path]] = {}
    for value in values:
        parts = value.split("=", 1)
        if len(parts) != 2 or ":" not in parts[0]:
            raise ValueError(
                "--bank must use target:model=checkpoint syntax, "
                f"got {value!r}."
            )
        target, model_name = parts[0].split(":", 1)
        target = target.strip().lower()
        model_name = model_name.strip().lower()
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model {model_name!r}; expected {MODEL_NAMES}.")
        path = Path(parts[1]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        banks.setdefault(target, {})
        if model_name in banks[target]:
            raise ValueError(f"Duplicate {target}:{model_name} checkpoint.")
        banks[target][model_name] = path
    for target, model_paths in banks.items():
        missing = set(MODEL_NAMES).difference(model_paths)
        if missing:
            raise ValueError(f"{target} is missing models {sorted(missing)}.")
    return banks


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["kits19", "brats18", "lits"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument(
        "--bank",
        action="append",
        required=True,
        help="Checkpoint in target:model=path form; provide four entries per target.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--encoder-embed-dim", type=int, default=48)
    parser.add_argument("--sw-overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = split_protocol(args.dataset, args.split_file)
    banks = _bank_specs(args.bank)
    set_seed(args.seed)
    device = torch.device(args.device)
    cases = cases_for_split(
        args.dataset,
        args.dataset_root,
        args.split_file,
        args.split,
    )
    in_channels = len(cases[0].image_paths)
    complete_summary: Dict[str, object] = {
        "dataset": args.dataset,
        "split": args.split,
        "purpose": "final_test" if args.split == "test" else "validation",
        "data_protocol": protocol,
        "seed": args.seed,
        "targets": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target, model_paths in sorted(banks.items()):
        target_id = target_label_ids(args.dataset, target)[0]
        models: List[torch.nn.Module] = []
        for model_name in MODEL_NAMES:
            model, _ = load_model_checkpoint(
                model_paths[model_name],
                model_name,
                target_id,
                in_channels,
                args.patch_size,
                device,
                args.encoder_embed_dim,
                data_protocol=protocol,
                target=target,
            )
            model.eval()
            models.append(model)
        summary, rows = evaluate_ensemble(
            models,
            cases,
            args.dataset,
            target,
            args.patch_size,
            args.sw_overlap,
            args.sw_batch_size,
            device,
            args.amp,
        )
        _write_csv(args.output_dir / f"{target}_cases.csv", rows)
        (args.output_dir / f"{target}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        complete_summary["targets"][target] = summary
    (args.output_dir / "summary.json").write_text(
        json.dumps(complete_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(complete_summary, indent=2))


if __name__ == "__main__":
    main()

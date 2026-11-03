#!/usr/bin/env python3
"""Build fixed full-volume Consensus and Divergence component labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Sequence

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dispam.cache import region_files, save_regions
from dispam.component_selection import label_components_6
from dispam.components import build_consensus_divergence, sliding_window_probability
from dispam.datasets import cases_for_split, target_label_ids, split_protocol
from dispam.io import load_case_image, robust_normalize_channels
from dispam.models import MODEL_NAMES, load_model_checkpoint
from dispam.reproducibility import set_seed
from dispam.tensor_ops import mask_zyx_to_model_tensor, model_tensor_to_zyx


def _checkpoint_specs(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must use model=path syntax, got {value!r}.")
        model_name, raw_path = value.split("=", 1)
        model_name = model_name.strip().lower()
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model {model_name!r}; expected {MODEL_NAMES}.")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        output[model_name] = path
    missing = set(MODEL_NAMES).difference(output)
    if missing:
        raise ValueError(f"Missing checkpoints for {sorted(missing)}.")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["kits19", "brats18", "lits"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train"], default=["train"])
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-size", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--encoder-embed-dim", type=int, default=48)
    parser.add_argument("--sw-overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--consensus-min-votes", type=int, default=3)
    parser.add_argument("--probability-std-threshold", type=float, default=0.12)
    parser.add_argument("--min-component-voxels", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = split_protocol(args.dataset, args.split_file)
    checkpoint_paths = _checkpoint_specs(args.checkpoint)
    set_seed(args.seed)
    device = torch.device(args.device)
    for split in args.splits:
        cases = cases_for_split(
            args.dataset,
            args.dataset_root,
            args.split_file,
            split,
        )
        in_channels = len(cases[0].image_paths)
        target_id = target_label_ids(args.dataset, args.target)[0]
        models = []
        for model_name in MODEL_NAMES:
            model, _ = load_model_checkpoint(
                checkpoint_paths[model_name],
                model_name,
                target_id,
                in_channels,
                args.patch_size,
                device,
                args.encoder_embed_dim,
                data_protocol=protocol,
                target=args.target,
            )
            model.eval().requires_grad_(False)
            models.append(model)
        for case in cases:
            files = region_files(
                args.output_root,
                split,
                args.target,
                case.case_id,
            )
            if files.npz.exists() and not args.overwrite:
                continue
            image = robust_normalize_channels(load_case_image(case.image_paths))
            probabilities = []
            for model in models:
                probability, _ = sliding_window_probability(
                    model,
                    image,
                    args.patch_size,
                    args.sw_overlap,
                    args.sw_batch_size,
                    device,
                    args.amp,
                )
                probabilities.append(
                    mask_zyx_to_model_tensor(probability).unsqueeze(0)
                )
            regions = build_consensus_divergence(
                probabilities,
                args.probability_threshold,
                args.consensus_min_votes,
                args.probability_std_threshold,
                args.min_component_voxels,
            )
            consensus_zyx = model_tensor_to_zyx(
                regions.consensus[0, 0]
            ) > 0.5
            divergence_zyx = model_tensor_to_zyx(
                regions.divergence[0, 0]
            ) > 0.5
            consensus_labels, consensus_count = label_components_6(consensus_zyx)
            divergence_labels, divergence_count = label_components_6(divergence_zyx)
            save_regions(
                files,
                consensus_labels,
                divergence_labels,
                {
                    "case_id": case.case_id,
                    "dataset": args.dataset,
                    "target": args.target,
                    "split": split,
                    "data_protocol": protocol,
                    "probability_threshold": args.probability_threshold,
                    "consensus_min_votes": args.consensus_min_votes,
                    "probability_std_type": "population",
                    "probability_std_threshold": args.probability_std_threshold,
                    "min_component_voxels": args.min_component_voxels,
                    "component_connectivity": 6,
                    "consensus_component_count": int(consensus_count),
                    "divergence_component_count": int(divergence_count),
                    "seed": args.seed,
                },
            )
            print(json.dumps({"saved": str(files.npz), "case_id": case.case_id}))


if __name__ == "__main__":
    main()

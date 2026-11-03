#!/usr/bin/env python3
"""Create target-specific Hard/Core PAM Cache entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dispam.agent import (
    OpenAICompatibleVisibilityAgent,
    ReplayVisibilityAgent,
    select_visible_slice,
)
from dispam.cache import cache_files, save_cache
from dispam.datasets import cases_for_split, target_label_ids, load_splits
from dispam.io import load_case_image, load_volume_any
from dispam.pam import (
    FrozenPAM,
    boxes_from_binary_mask,
    build_multi_perturbation_targets,
)
from dispam.reproducibility import set_seed
from dispam.slice_selector import propagation_aware_slice_estimation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["kits19", "brats18", "lits"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train"], default=["train"])
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pam-checkpoint", type=Path, required=True)
    parser.add_argument("--box2mask-checkpoint", type=Path)
    parser.add_argument("--prompt-type", choices=["mask", "box"], default="mask")
    parser.add_argument("--prompt-channel", type=int, default=0)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--agent-backend", choices=["replay", "api"], default="replay")
    parser.add_argument("--agent-decisions", type=Path)
    parser.add_argument("--agent-model", default="gpt-5.5")
    parser.add_argument("--agent-endpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--binarization-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_agent(args: argparse.Namespace):
    if args.agent_backend == "replay":
        if args.agent_decisions is None:
            raise ValueError("--agent-decisions is required for replay.")
        return ReplayVisibilityAgent(args.agent_decisions)
    if not args.agent_model or not args.agent_endpoint:
        raise ValueError("--agent-model and --agent-endpoint are required for api.")
    return OpenAICompatibleVisibilityAgent(args.agent_model, args.agent_endpoint)


def main() -> None:
    args = parse_args()
    load_splits(args.split_file)
    set_seed(args.seed)
    prompts: Dict[str, str] = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    agent = build_agent(args)
    pam = FrozenPAM(
        propagation_checkpoint=args.pam_checkpoint,
        box2mask_checkpoint=(
            args.box2mask_checkpoint if args.prompt_type == "box" else None
        ),
        device=torch.device(args.device),
        target_size=args.target_size,
    )
    for split in args.splits:
        cases = cases_for_split(
            args.dataset,
            args.dataset_root,
            args.split_file,
            split,
        )
        for case in cases:
            image = load_case_image(case.image_paths)
            label = load_volume_any(case.label_path).astype(np.int32)
            channel = int(args.prompt_channel)
            if not 0 <= channel < image.shape[0]:
                raise ValueError(f"prompt-channel={channel} is invalid for {case.case_id}.")
            ranking = propagation_aware_slice_estimation(image)
            for target in args.targets:
                files = cache_files(args.cache_root, split, target, case.case_id)
                if files.npz.exists() and not args.overwrite:
                    metadata = (
                        json.loads(files.metadata.read_text(encoding="utf-8"))
                        if files.metadata.exists() else {}
                    )
                    if metadata.get("agent_queries_per_candidate") != 1:
                        raise ValueError("Existing cache uses an older agent policy; rebuild with --overwrite.")
                    continue
                target_mask = np.isin(
                    label,
                    target_label_ids(args.dataset, target),
                )
                if not np.any(target_mask):
                    continue
                prompt_key = f"{args.dataset}.{target}"
                if prompt_key not in prompts:
                    raise KeyError(f"Missing target prompt {prompt_key!r}.")
                selected_slice = select_visible_slice(
                    agent,
                    case.case_id,
                    target,
                    ranking.order,
                    image[channel],
                    prompts[prompt_key],
                )
                if selected_slice is None:
                    raise RuntimeError(
                        f"{case.case_id}/{target}: no candidate received an affirmative "
                        "visibility decision; no teacher cache was generated."
                    )
                annotated_mask = target_mask[selected_slice]
                if not np.any(annotated_mask):
                    raise RuntimeError(
                        f"{case.case_id}/{target}: selected slice has no simulated annotation."
                    )
                if args.prompt_type == "box":
                    boxes = boxes_from_binary_mask(annotated_mask)
                    prompt_mask = pam.mask_from_boxes(
                        image[channel, selected_slice],
                        boxes,
                        args.binarization_threshold,
                    )
                else:
                    boxes = []
                    prompt_mask = annotated_mask
                targets = build_multi_perturbation_targets(
                    pam,
                    image[channel],
                    selected_slice,
                    prompt_mask,
                    args.binarization_threshold,
                    args.confidence_threshold,
                )
                save_cache(
                    files,
                    targets["hard_target"],
                    targets["core_region"],
                    targets["core_probability"],
                    {
                        "case_id": case.case_id,
                        "dataset": args.dataset,
                        "target": target,
                        "split": split,
                        "selected_slice": selected_slice,
                        "agent_queries_per_candidate": 1,
                        "agent_decoding": {"temperature": 0, "top_p": 1},
                        "agent_model": args.agent_model if args.agent_backend == "api" else "replay",
                        "prompt_type": args.prompt_type,
                        "boxes_xyxy": boxes,
                        "pase_score": float(ranking.score[selected_slice]),
                        "confidence_threshold": args.confidence_threshold,
                        "binarization_threshold": args.binarization_threshold,
                        "seed": args.seed,
                    },
                )
                print(f"Saved {files.npz}")


if __name__ == "__main__":
    main()

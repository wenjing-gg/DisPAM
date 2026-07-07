#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from StudentBank.student_model_registry_v2 import (  # noqa: E402
    canonical_student_model_name,
    load_single_target_student_model,
    parse_patch_size_arg,
)

CURRENT_CDFT_METHOD = "component_embedding_contrast_cdft"


def load_trainable_student(
    checkpoint_path: Path,
    student_model: str,
    target_label_id: int,
    device: torch.device,
    encoder_embed_dim: int,
    patch_size: Tuple[int, int, int],
) -> torch.nn.Module:
    """Load one distilled student checkpoint for component-contrast fine-tuning."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = canonical_student_model_name(student_model)
    train_args = dict(checkpoint.get("args", {}))
    ckpt_patch_size = parse_patch_size_arg(train_args.get("patch_size", patch_size))
    if tuple(int(v) for v in ckpt_patch_size) != tuple(int(v) for v in patch_size):
        raise RuntimeError(
            "CDFT patch size must match the checkpoint patch size: "
            f"checkpoint_patch_size={tuple(int(v) for v in ckpt_patch_size)} "
            f"requested_patch_size={tuple(int(v) for v in patch_size)}."
        )
    _ = int(train_args.get("encoder_embed_dim", encoder_embed_dim))
    model, _, loaded_name = load_single_target_student_model(
        checkpoint_path=checkpoint_path,
        target_label_id=int(target_label_id),
        device=device,
        student_model=model_name,
        fallback_patch_size=ckpt_patch_size,
    )
    if loaded_name != model_name:
        raise RuntimeError(f"Loaded checkpoint as {loaded_name!r}, expected {model_name!r}.")
    for param in model.parameters():
        param.requires_grad_(True)
    model.train()
    return model


def checkpoint_report(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    args = dict(payload.get("args", {}))
    return {
        "method": CURRENT_CDFT_METHOD,
        "checkpoint": str(Path(path).expanduser()),
        "epoch": payload.get("epoch"),
        "student_model": args.get("student_model"),
        "target_name": args.get("target_name"),
        "target_label_id": args.get("target_label_id"),
        "dataset": args.get("dataset"),
        "teacher_cache_root": args.get("teacher_cache_root"),
        "run_config": args.get("run_config"),
        "lr": args.get("lr"),
        "head_lr": args.get("head_lr"),
        "batch_size": args.get("batch_size"),
        "epochs": args.get("epochs"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utilities for the current DisPAM CDFT checkpoints.")
    parser.add_argument("checkpoint", nargs="?", default="", help="Optional checkpoint to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint:
        print(json.dumps({"method": CURRENT_CDFT_METHOD, "status": "ready"}, indent=2))
        return
    print(json.dumps(checkpoint_report(Path(args.checkpoint)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

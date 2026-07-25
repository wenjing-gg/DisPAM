#!/usr/bin/env python3
"""Train one heterogeneous student with Hard-Core Dual Distillation.

Paper mapping:
  Sparse-Prompt Dual Distillation / Hard-Core Dual Distillation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dispam.cache import DistillationPatchDataset
from dispam.datasets import cases_for_split, target_label_ids
from dispam.evaluation import evaluate_model
from dispam.losses import (
    core_distillation_loss,
    hard_distillation_loss,
    sum_active_losses,
)
from dispam.models import MODEL_NAMES, build_model
from dispam.models.registry import save_model_checkpoint
from dispam.reproducibility import make_generator, seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["kits19", "brats18", "lits"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--samples-per-case", type=int, default=1)
    parser.add_argument("--foreground-probability", type=float, default=0.8)
    parser.add_argument("--core-probability", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--encoder-embed-dim", type=int, default=48)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--sw-overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    train_cases = cases_for_split(
        args.dataset, args.dataset_root, args.split_file, "train"
    )
    val_cases = cases_for_split(
        args.dataset, args.dataset_root, args.split_file, "val"
    )
    dataset = DistillationPatchDataset(
        train_cases,
        args.cache_root,
        "train",
        args.target,
        args.patch_size,
        args.samples_per_case,
        args.foreground_probability,
        args.core_probability,
        args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker(args.seed),
        generator=make_generator(args.seed),
    )
    in_channels = len(train_cases[0].image_paths)
    model = build_model(
        args.model,
        target_label_ids(args.dataset, args.target)[0],
        in_channels,
        args.patch_size,
        args.encoder_embed_dim,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_dice = -1.0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(args.amp and device.type == "cuda")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            hard_target = batch["hard_target"].to(device=device, dtype=torch.float32)
            core_region = batch["core_region"].to(device=device, dtype=torch.float32)
            core_probability = batch["core_probability"].to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image)
                hard_loss = hard_distillation_loss(logits, hard_target)
                core_loss = core_distillation_loss(
                    logits,
                    core_probability,
                    core_region,
                )
                loss = sum_active_losses([hard_loss, core_loss])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().item())
            steps += 1
        scheduler.step()
        row = {"epoch": epoch, "train_loss": total / max(1, steps)}
        should_validate = epoch % args.validation_interval == 0 or epoch == args.epochs
        if should_validate:
            val_dice, _ = evaluate_model(
                model,
                val_cases,
                args.dataset,
                args.target,
                args.patch_size,
                args.sw_overlap,
                args.sw_batch_size,
                device,
                args.amp,
            )
            row["val_dice"] = val_dice
            if val_dice > best_dice:
                best_dice = val_dice
                save_model_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args.model,
                    vars(args),
                )
        save_model_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            args.model,
            vars(args),
        )
        history.append(row)
        print(json.dumps(row))
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

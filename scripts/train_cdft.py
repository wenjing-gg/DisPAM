#!/usr/bin/env python3
"""Jointly fine-tune the four target-specific students with CDFT.

Paper mapping:
  Consensus-Divergence Fine-Tune / Consensus-Divergence Contrastive Learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dispam.cache import CDFTPatchDataset
from dispam.components import (
    component_masks_from_labels,
)
from dispam.datasets import cases_for_split, target_label_ids, split_protocol
from dispam.evaluation import evaluate_ensemble
from dispam.losses import (
    FeatureAdapter,
    SharedDescriptorEncoder,
    consensus_divergence_contrastive_loss,
    consensus_foreground_loss,
    core_distillation_loss,
    hard_distillation_loss,
)
from dispam.models import (
    MODEL_NAMES,
    forward_with_decoder_features,
    load_model_checkpoint,
)
from dispam.models.registry import save_model_checkpoint
from dispam.reproducibility import make_generator, seed_worker, set_seed


def _checkpoint_specs(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must use model=path syntax, got {value!r}.")
        model_name, raw_path = value.split("=", 1)
        model_name = model_name.strip().lower()
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model {model_name!r}; expected {MODEL_NAMES}.")
        if model_name in output:
            raise ValueError(f"Duplicate checkpoint for {model_name}.")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        output[model_name] = path
    missing = set(MODEL_NAMES).difference(output)
    if missing:
        raise ValueError(f"Missing checkpoints for {sorted(missing)}.")
    return output


def _mean_active(values: Iterable[torch.Tensor | None]) -> torch.Tensor | None:
    active = [value for value in values if value is not None]
    return torch.stack(active).mean() if active else None


def _sum_active(values: Iterable[torch.Tensor | None]) -> torch.Tensor:
    active = [value for value in values if value is not None]
    if not active:
        raise RuntimeError("CDFT produced no active loss term.")
    return torch.stack(active).sum()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint Consensus-Divergence fine-tuning for one binary target."
    )
    parser.add_argument("--dataset", required=True, choices=["kits19", "brats18", "lits"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--region-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Dual-distilled checkpoint in model=path form; provide all four models.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--samples-per-case", type=int, default=1)
    parser.add_argument("--foreground-probability", type=float, default=0.8)
    parser.add_argument("--core-probability", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--descriptor-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--encoder-embed-dim", type=int, default=48)
    parser.add_argument("--descriptor-hidden-dim", type=int, default=32)
    parser.add_argument("--descriptor-dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--sw-overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _save_bank(
    directory: Path,
    models: Dict[str, torch.nn.Module],
    adapters: Dict[str, FeatureAdapter],
    descriptor_encoder: SharedDescriptorEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metadata: Dict[str, object],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        save_model_checkpoint(
            directory / f"{model_name}.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            model_name,
            metadata,
        )
    torch.save(
        {
            "epoch": int(epoch),
            "adapters": {
                model_name: adapter.state_dict()
                for model_name, adapter in adapters.items()
            },
            "descriptor_encoder": descriptor_encoder.state_dict(),
            "metadata": metadata,
        },
        directory / "descriptors.pt",
    )


def main() -> None:
    args = parse_args()
    protocol = split_protocol(args.dataset, args.split_file)
    checkpoint_paths = _checkpoint_specs(args.checkpoint)
    set_seed(args.seed)
    device = torch.device(args.device)
    train_cases = cases_for_split(
        args.dataset, args.dataset_root, args.split_file, "train"
    )
    val_cases = cases_for_split(
        args.dataset, args.dataset_root, args.split_file, "val"
    )
    train_dataset = CDFTPatchDataset(
        train_cases,
        args.cache_root,
        args.region_root,
        "train",
        args.target,
        args.patch_size,
        args.samples_per_case,
        args.foreground_probability,
        args.core_probability,
        args.seed,
        data_protocol=protocol,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker(args.seed),
        generator=make_generator(args.seed),
    )
    in_channels = len(train_cases[0].image_paths)
    target_id = target_label_ids(args.dataset, args.target)[0]
    models: Dict[str, torch.nn.Module] = {}
    for model_name in MODEL_NAMES:
        models[model_name], _ = load_model_checkpoint(
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

    first_batch = next(iter(loader))
    first_image = first_batch["image"].to(device=device, dtype=torch.float32)
    feature_channels: Dict[str, int] = {}
    with torch.no_grad():
        for model_name, model in models.items():
            _, features = forward_with_decoder_features(model, first_image)
            feature_channels[model_name] = int(features.shape[1])
    adapters = {
        model_name: FeatureAdapter(channels, args.descriptor_hidden_dim).to(device)
        for model_name, channels in feature_channels.items()
    }
    descriptor_encoder = SharedDescriptorEncoder(
        args.descriptor_hidden_dim,
        args.descriptor_dim,
    ).to(device)
    optimizer = AdamW(
        [
            {
                "params": [
                    parameter
                    for model in models.values()
                    for parameter in model.parameters()
                ],
                "lr": args.learning_rate,
            },
            {
                "params": [
                    parameter
                    for adapter in adapters.values()
                    for parameter in adapter.parameters()
                ]
                + list(descriptor_encoder.parameters()),
                "lr": args.descriptor_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    use_amp = bool(args.amp and device.type == "cuda")
    metadata = {**vars(args), "data_protocol": protocol}
    metadata["checkpoint"] = [str(value) for value in args.checkpoint]
    metadata["feature_channels"] = feature_channels
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    history: List[Dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        for model in models.values():
            model.train()
        for adapter in adapters.values():
            adapter.train()
        descriptor_encoder.train()
        epoch_total = 0.0
        steps = 0
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            hard_target = batch["hard_target"].to(device=device, dtype=torch.float32)
            core_region = batch["core_region"].to(device=device, dtype=torch.float32)
            core_probability = batch["core_probability"].to(
                device=device, dtype=torch.float32
            )
            consensus_labels = batch["consensus_labels"].to(
                device=device, dtype=torch.int64
            )
            divergence_labels = batch["divergence_labels"].to(
                device=device, dtype=torch.int64
            )
            consensus_region = (consensus_labels > 0).float()
            consensus_parts = component_masks_from_labels(consensus_labels)
            divergence_parts = component_masks_from_labels(divergence_labels)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits_by_model: List[torch.Tensor] = []
                descriptor_fields: List[torch.Tensor] = []
                for model_name in MODEL_NAMES:
                    logits, features = forward_with_decoder_features(
                        models[model_name],
                        image,
                    )
                    logits_by_model.append(logits)
                    descriptor_fields.append(
                        descriptor_encoder(adapters[model_name](features))
                    )
                hard = _mean_active(
                    hard_distillation_loss(logits, hard_target)
                    for logits in logits_by_model
                )
                core = _mean_active(
                    core_distillation_loss(logits, core_probability, core_region)
                    for logits in logits_by_model
                )
                consensus = _mean_active(
                    consensus_foreground_loss(logits, consensus_region)
                    for logits in logits_by_model
                )
                contrast = consensus_divergence_contrastive_loss(
                    descriptor_fields,
                    consensus_parts,
                    divergence_parts,
                    args.temperature,
                )
                loss = _sum_active([hard, core, consensus, contrast])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ],
                args.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            epoch_total += float(loss.detach().item())
            steps += 1
        scheduler.step()
        row: Dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": epoch_total / max(1, steps),
        }
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            summary, _ = evaluate_ensemble(
                [models[name] for name in MODEL_NAMES],
                val_cases,
                args.dataset,
                args.target,
                args.patch_size,
                args.sw_overlap,
                args.sw_batch_size,
                device,
                args.amp,
            )
            val_dice = float(summary["mean_dice"] or 0.0)
            row["val_dice"] = val_dice
            if val_dice > best_dice:
                best_dice = val_dice
                _save_bank(
                    args.output_dir / "best",
                    models,
                    adapters,
                    descriptor_encoder,
                    optimizer,
                    scheduler,
                    epoch,
                    metadata,
                )
        _save_bank(
            args.output_dir / "last",
            models,
            adapters,
            descriptor_encoder,
            optimizer,
            scheduler,
            epoch,
            metadata,
        )
        history.append(row)
        print(json.dumps(row))
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

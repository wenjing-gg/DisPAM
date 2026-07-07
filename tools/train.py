#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from StudentBank.cdft import load_trainable_student  # noqa: E402
from dataset import load_split_case_ids  # noqa: E402
from tools.profiles import get_target_profile  # noqa: E402
from tools.components import label_components_6, select_component  # noqa: E402
from StudentBank.student_model_registry_v2 import checkpoint_filename, checkpoint_last_filename  # noqa: E402
from tools.data import (  # noqa: E402
    TeacherCacheVolumeDataset,
    case_lookup,
    model_hwd_to_zyx,
    zyx_to_model_tensor,
)
from tools.common import (  # noqa: E402
    dice_loss_with_logits,
    set_seed,
    weighted_bce_with_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a prompt-free student with component embedding contrast. The student "
            "logit field is converted to a dense vector embedding map by a lightweight "
            "3D depthwise-separable head; teacher components are pooled into vectors "
            "and optimized with an InfoNCE-style objective."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--teacher-cache-root", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-label-id", type=int, default=0)
    parser.add_argument("--student-model", default="segmambav2")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--variant-name", default="component_embedding_contrast")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--samples-per-case", type=int, default=2)
    parser.add_argument("--foreground-crop-prob", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-cases", type=int, default=10)
    parser.add_argument("--train-case-ids", nargs="*", default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--encoder-embed-dim", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--embedding-hidden-dim", type=int, default=32)
    parser.add_argument("--contrast-temperature", type=float, default=0.15)
    parser.add_argument("--max-neg-components", type=int, default=4)
    parser.add_argument("--lambda-bce", type=float, default=0.30)
    parser.add_argument("--lambda-dice", type=float, default=0.55)
    parser.add_argument("--lambda-contrast", type=float, default=0.20)
    parser.add_argument("--lambda-keep", type=float, default=0.03)
    parser.add_argument("--seed-thresh", type=float, default=0.70)
    parser.add_argument("--seed-power", type=float, default=2.0)
    parser.add_argument("--fill-start-coverage", type=float, default=0.60)
    parser.add_argument("--trust-start-coverage", type=float, default=0.80)
    parser.add_argument("--min-component-voxels", type=int, default=8)
    parser.add_argument("--challenge-density-ratio", type=float, default=1.20)
    parser.add_argument("--challenge-min-selected-fraction", type=float, default=0.02)
    parser.add_argument("--challenge-min-voxels", type=int, default=512)
    parser.add_argument(
        "--component-selection-mode",
        choices=["density_challenge_top1", "multi_instance_density"],
        default="density_challenge_top1",
    )
    parser.add_argument("--multi-density-thresh", type=float, default=0.50)
    parser.add_argument("--multi-seed-fraction-thresh", type=float, default=0.05)
    parser.add_argument("--multi-min-voxels", type=int, default=512)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class DepthwiseComponentEmbeddingHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        hidden = int(max(4, hidden_dim))
        embed = int(max(4, embed_dim))
        self.depthwise_in = nn.Conv3d(2, 2, kernel_size=3, padding=1, groups=2, bias=False)
        self.pointwise_in = nn.Conv3d(2, hidden, kernel_size=1, bias=False)
        self.norm_in = nn.GroupNorm(num_groups=1, num_channels=hidden)
        self.depthwise_mid = nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)
        self.pointwise_out = nn.Conv3d(hidden, embed, kernel_size=1, bias=True)

    def forward(self, logits_hwd: torch.Tensor) -> torch.Tensor:
        logits = logits_hwd.float()
        x = torch.cat([logits, torch.sigmoid(logits)], dim=1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = self.depthwise_in(x)
        x = self.pointwise_in(x)
        x = F.gelu(self.norm_in(x))
        x = self.depthwise_mid(x)
        x = self.pointwise_out(F.gelu(x))
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return F.normalize(x, p=2, dim=1)


def parse_case_id_values(raw_values: Sequence[str] | None) -> List[str]:
    values: List[str] = []
    for raw in raw_values or []:
        values.extend(part.strip() for part in str(raw).split(",") if part.strip())
    return values


def select_case_ids(split_ids: Sequence[str], explicit_ids: Sequence[str] | None, max_cases: int) -> List[str]:
    ids = [str(v) for v in split_ids]
    explicit = parse_case_id_values(explicit_ids)
    if explicit:
        allowed = {str(v).lower() for v in ids}
        missing = [str(v) for v in explicit if str(v).lower() not in allowed]
        if missing:
            raise RuntimeError(f"Explicit case ids are not in split: {missing}")
        ids = explicit
    if int(max_cases) > 0:
        ids = ids[: int(max_cases)]
    return ids


def fill_alpha_from_coverage(coverage: float, fill_start: float, trust_start: float) -> float:
    fill = float(fill_start)
    trust = max(float(trust_start), fill + 1e-6)
    return float(np.clip((trust - float(coverage)) / (trust - fill), 0.0, 1.0))


def build_component_training_targets(
    *,
    logits: torch.Tensor,
    final: torch.Tensor,
    seed_thresh: float,
    seed_power: float,
    fill_start_coverage: float,
    trust_start_coverage: float,
    min_component_voxels: int,
    challenge_density_ratio: float,
    challenge_min_selected_fraction: float,
    challenge_min_voxels: int,
    component_selection_mode: str,
    multi_density_thresh: float,
    multi_seed_fraction_thresh: float,
    multi_min_voxels: int,
    max_neg_components: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    probs = torch.sigmoid(logits.detach().float()).cpu().numpy()
    final_np = (final.detach().float().cpu().numpy() > 0.5)
    batch_size = int(probs.shape[0])
    max_neg = int(max(1, max_neg_components))
    target = np.zeros_like(probs, dtype=np.float32)
    pos_masks = np.zeros_like(probs, dtype=np.float32)
    neg_masks = np.zeros((batch_size, max_neg, 1, *probs.shape[2:]), dtype=np.float32)
    neg_valid = np.zeros((batch_size, max_neg), dtype=np.float32)
    rows: List[Dict[str, Any]] = []

    for batch_index in range(batch_size):
        teacher_mask = np.asarray(model_hwd_to_zyx(final_np[batch_index, 0]), dtype=bool)
        prob_zyx = np.asarray(model_hwd_to_zyx(probs[batch_index, 0]), dtype=np.float32)
        selected, selection = select_component(
            teacher_mask=teacher_mask,
            prob=prob_zyx,
            seed_thresh=float(seed_thresh),
            seed_power=float(seed_power),
            min_component_voxels=int(min_component_voxels),
            challenge_density_ratio=float(challenge_density_ratio),
            challenge_min_selected_fraction=float(challenge_min_selected_fraction),
            challenge_min_voxels=int(challenge_min_voxels),
        )
        selected_rows = [dict(row) for row in selection.get("selected_components", [])]
        if not selected_rows:
            selected_info = selection.get("selected_component") or {}
            if selected_info:
                selected_rows = [dict(selected_info)]
        if str(component_selection_mode) == "multi_instance_density":
            selected_rows = [
                dict(row)
                for row in selection.get("component_rows", [])
                if int(row.get("voxels", 0)) >= int(multi_min_voxels)
                and float(row.get("density_evidence", row.get("coverage", 0.0))) >= float(multi_density_thresh)
                and float(row.get("seed_fraction", 0.0)) >= float(multi_seed_fraction_thresh)
            ]
            if not selected_rows:
                fallback = selection.get("selected_component") or {}
                selected_rows = [dict(fallback)] if fallback else []
            selected_ids = [int(row.get("component_id", 0)) for row in selected_rows if int(row.get("component_id", 0)) > 0]
            if selected_ids:
                labels, _ = label_components_6(teacher_mask)
                selected = np.isin(labels, selected_ids)
            selected_rows = sorted(selected_rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)
            selection["selected_components"] = selected_rows
            selection["selected_component"] = selected_rows[0] if selected_rows else None
        selected_info = selection.get("selected_component") or {}
        selected_component_ids = {
            int(row.get("component_id", 0))
            for row in selected_rows
            if int(row.get("component_id", 0)) > 0
        }
        selected_values = np.asarray(prob_zyx[selected], dtype=np.float32)
        coverage = float(selected_values.mean()) if selected_values.size else float(selected_info.get("coverage", 0.0))
        alpha = fill_alpha_from_coverage(
            coverage=coverage,
            fill_start=float(fill_start_coverage),
            trust_start=float(trust_start_coverage),
        )
        target_zyx = np.zeros_like(prob_zyx, dtype=np.float32)
        target_zyx[selected] = (1.0 - alpha) * prob_zyx[selected] + alpha
        target[batch_index, 0] = zyx_to_model_tensor(target_zyx)[0].numpy()
        pos_masks[batch_index, 0] = zyx_to_model_tensor(np.asarray(selected, dtype=np.float32))[0].numpy()

        labels, _ = label_components_6(teacher_mask)
        candidate_rows = [
            row
            for row in selection.get("component_rows", [])
            if int(row.get("component_id", 0)) not in selected_component_ids
            and int(row.get("voxels", 0)) >= int(min_component_voxels)
        ]
        candidate_rows = sorted(
            candidate_rows,
            key=lambda row: (
                float(row.get("student_score", row.get("score", 0.0))),
                float(row.get("density_evidence", row.get("coverage", 0.0))),
            ),
            reverse=True,
        )
        negative_component_ids: List[int] = []
        for neg_index, row in enumerate(candidate_rows[:max_neg]):
            component_id = int(row.get("component_id", 0))
            if component_id <= 0:
                continue
            component_mask = labels == component_id
            neg_masks[batch_index, neg_index, 0] = zyx_to_model_tensor(component_mask.astype(np.float32))[0].numpy()
            neg_valid[batch_index, neg_index] = 1.0
            negative_component_ids.append(component_id)
        rows.append(
            {
                "component_count": int(selection.get("component_count", 0)),
                "selected_component": selected_info,
                "selected_components": selected_rows,
                "negative_component_ids": negative_component_ids,
                "fill_alpha": float(alpha),
                "target_voxels": int(np.sum(target_zyx >= 0.5)),
            }
        )

    return (
        torch.from_numpy(target).to(device=logits.device, dtype=torch.float32),
        torch.from_numpy(pos_masks).to(device=logits.device, dtype=torch.float32),
        torch.from_numpy(neg_masks).to(device=logits.device, dtype=torch.float32),
        torch.from_numpy(neg_valid).to(device=logits.device, dtype=torch.float32),
        rows,
    )


def masked_avg_pool(embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    denom = torch.sum(weights, dim=(2, 3, 4)).clamp_min(1.0)
    pooled = torch.sum(embedding * weights, dim=(2, 3, 4)) / denom
    return F.normalize(pooled, p=2, dim=1)


def masked_max_pool(embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.float() > 0.5
    values = embedding.masked_fill(~valid, -1e4)
    pooled = torch.amax(values, dim=(2, 3, 4))
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    return F.normalize(pooled, p=2, dim=1)


def component_embedding_contrast_loss(
    embedding: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_masks: torch.Tensor,
    neg_valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    batch_size = int(embedding.shape[0])
    losses: List[torch.Tensor] = []
    temp = float(max(temperature, 1e-4))
    pos_avg = masked_avg_pool(embedding, pos_mask)
    pos_max = masked_max_pool(embedding, pos_mask)
    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(neg_valid[batch_index] > 0.5, as_tuple=False).flatten()
        if int(valid_indices.numel()) <= 0:
            continue
        neg_avg_vectors: List[torch.Tensor] = []
        neg_max_vectors: List[torch.Tensor] = []
        for neg_index in valid_indices.tolist():
            mask = neg_masks[batch_index : batch_index + 1, int(neg_index)]
            neg_avg_vectors.append(masked_avg_pool(embedding[batch_index : batch_index + 1], mask)[0])
            neg_max_vectors.append(masked_max_pool(embedding[batch_index : batch_index + 1], mask)[0])
        neg_keys = torch.stack([*neg_avg_vectors, *neg_max_vectors], dim=0)
        q_avg = pos_avg[batch_index : batch_index + 1]
        q_max = pos_max[batch_index : batch_index + 1]
        k_avg = pos_avg[batch_index : batch_index + 1]
        k_max = pos_max[batch_index : batch_index + 1]
        logits_avg = torch.cat(
            [
                torch.sum(q_avg * k_max, dim=1, keepdim=True),
                torch.matmul(q_avg, neg_keys.t()),
            ],
            dim=1,
        ) / temp
        logits_max = torch.cat(
            [
                torch.sum(q_max * k_avg, dim=1, keepdim=True),
                torch.matmul(q_max, neg_keys.t()),
            ],
            dim=1,
        ) / temp
        labels = torch.zeros(1, dtype=torch.long, device=embedding.device)
        losses.append(F.cross_entropy(logits_avg, labels))
        losses.append(F.cross_entropy(logits_max, labels))
    if not losses:
        return embedding.new_tensor(0.0)
    return torch.stack(losses).mean()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    embedding_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_head_state_dict": embedding_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": int(epoch),
            "args": vars(args),
            "protocol": {
                "method": "component_embedding_contrast",
                "sampling": "foreground_aware_random_crop",
                "embedding": "depthwise_3d_dense_map_masked_avg_max_pooling",
                "contrast": "avg_to_max_and_max_to_avg_infonce",
                "gt_usage": "not_used_in_training",
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(str(args.device))
    patch_size = tuple(int(v) for v in args.patch_size)
    profile = get_target_profile(str(args.dataset))
    if str(args.target_name) not in profile.branch_export_label_ids:
        raise RuntimeError(
            f"Unsupported target_name={args.target_name!r} for dataset={args.dataset!r}. "
            f"Expected one of {tuple(profile.branch_export_label_ids)}."
        )
    if int(args.target_label_id) <= 0:
        args.target_label_id = int(profile.branch_export_label_ids[str(args.target_name)])
    out_dir = Path(args.out_root) / str(args.variant_name) / str(args.target_name) / str(args.student_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    lookup = case_lookup(args.dataset, args.dataset_root)
    train_case_ids = select_case_ids(
        load_split_case_ids(Path(args.run_config), "train"),
        args.train_case_ids,
        int(args.max_train_cases),
    )
    dataset = TeacherCacheVolumeDataset(
        train_case_ids,
        lookup,
        teacher_cache_root=Path(args.teacher_cache_root),
        split="train",
        target_name=str(args.target_name),
        patch_size=patch_size,
        samples_per_case=int(args.samples_per_case),
        foreground_crop_prob=float(args.foreground_crop_prob),
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = load_trainable_student(
        checkpoint_path=Path(args.init_checkpoint).expanduser().resolve(),
        student_model=str(args.student_model),
        target_label_id=int(args.target_label_id),
        device=device,
        encoder_embed_dim=int(args.encoder_embed_dim),
        patch_size=patch_size,
    )
    init_model = load_trainable_student(
        checkpoint_path=Path(args.init_checkpoint).expanduser().resolve(),
        student_model=str(args.student_model),
        target_label_id=int(args.target_label_id),
        device=device,
        encoder_embed_dim=int(args.encoder_embed_dim),
        patch_size=patch_size,
    )
    init_model.eval()
    for param in init_model.parameters():
        param.requires_grad_(False)
    embedding_head = DepthwiseComponentEmbeddingHead(
        embed_dim=int(args.embedding_dim),
        hidden_dim=int(args.embedding_hidden_dim),
    ).to(device)
    optimizer = AdamW(
        [
            {"params": model.parameters(), "lr": float(args.lr)},
            {"params": embedding_head.parameters(), "lr": float(args.head_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    use_amp = bool(args.amp and device.type == "cuda")
    best_path = out_dir / checkpoint_filename(str(args.student_model), str(args.target_name))
    last_path = out_dir / checkpoint_last_filename(str(args.student_model), str(args.target_name))
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        embedding_head.train()
        totals = {
            "loss": 0.0,
            "bce": 0.0,
            "dice": 0.0,
            "keep": 0.0,
            "contrast": 0.0,
            "target_mean": 0.0,
        }
        steps = 0
        selected_rows: List[Dict[str, Any]] = []
        progress = tqdm(loader, desc=f"component-contrast-e{epoch}", dynamic_ncols=True)
        for batch in progress:
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            final = batch["final_target"].to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    init_logits = init_model(images)
                init_prob = torch.sigmoid(init_logits).detach()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                embedding = embedding_head(logits)
            latent_target, pos_mask, neg_masks, neg_valid, rows = build_component_training_targets(
                logits=logits,
                final=final,
                seed_thresh=float(args.seed_thresh),
                seed_power=float(args.seed_power),
                fill_start_coverage=float(args.fill_start_coverage),
                trust_start_coverage=float(args.trust_start_coverage),
                min_component_voxels=int(args.min_component_voxels),
                challenge_density_ratio=float(args.challenge_density_ratio),
                challenge_min_selected_fraction=float(args.challenge_min_selected_fraction),
                challenge_min_voxels=int(args.challenge_min_voxels),
                component_selection_mode=str(args.component_selection_mode),
                multi_density_thresh=float(args.multi_density_thresh),
                multi_seed_fraction_thresh=float(args.multi_seed_fraction_thresh),
                multi_min_voxels=int(args.multi_min_voxels),
                max_neg_components=int(args.max_neg_components),
            )
            selected_rows.extend(rows[:4])
            weight = torch.clamp(1.0 + 3.0 * latent_target + 1.5 * pos_mask, 1.0, 5.0)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss_bce = weighted_bce_with_logits(logits, latent_target, weight)
                loss_dice = dice_loss_with_logits(logits, latent_target, weight)
                loss_keep = torch.mean((torch.sigmoid(logits) - init_prob) ** 2)
                loss_contrast = component_embedding_contrast_loss(
                    embedding=embedding.float(),
                    pos_mask=pos_mask,
                    neg_masks=neg_masks,
                    neg_valid=neg_valid,
                    temperature=float(args.contrast_temperature),
                )
                loss = (
                    float(args.lambda_bce) * loss_bce
                    + float(args.lambda_dice) * loss_dice
                    + float(args.lambda_keep) * loss_keep
                    + float(args.lambda_contrast) * loss_contrast
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(embedding_head.parameters()),
                    max_norm=float(args.grad_clip_norm),
                )
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            totals["loss"] += float(loss.detach().item())
            totals["bce"] += float(loss_bce.detach().item())
            totals["dice"] += float(loss_dice.detach().item())
            totals["keep"] += float(loss_keep.detach().item())
            totals["contrast"] += float(loss_contrast.detach().item())
            totals["target_mean"] += float(latent_target.detach().mean().item())
            progress.set_postfix(loss=f"{totals['loss']/steps:.4f}", contrast=f"{totals['contrast']/steps:.4f}")
        scheduler.step()
        row = {
            "epoch": int(epoch),
            "train_loss": float(totals["loss"] / max(steps, 1)),
            "train_bce": float(totals["bce"] / max(steps, 1)),
            "train_dice": float(totals["dice"] / max(steps, 1)),
            "train_keep": float(totals["keep"] / max(steps, 1)),
            "train_contrast": float(totals["contrast"] / max(steps, 1)),
            "target_mean": float(totals["target_mean"] / max(steps, 1)),
            "selected_rows_sample": selected_rows[:16],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
        }
        save_checkpoint(last_path, model, embedding_head, optimizer, scheduler, scaler, epoch, args)
        save_checkpoint(best_path, model, embedding_head, optimizer, scheduler, scaler, epoch, args)
        print(
            f"[Epoch {epoch}] loss={row['train_loss']:.4f} "
            f"contrast={row['train_contrast']:.4f} target_mean={row['target_mean']:.6f}",
            flush=True,
        )
    print(f"[Done] checkpoint={best_path}", flush=True)


if __name__ == "__main__":
    main()

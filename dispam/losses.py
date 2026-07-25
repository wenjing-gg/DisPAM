"""Hard-Core distillation and Consensus-Divergence objectives.

Paper mapping:
  Hard-Core Dual Distillation uses the Normal PAM mask for Hard supervision and
  the confidence-filtered Dilation/Erosion intersection for Core supervision.
  Consensus-Divergence Contrastive Learning aligns cross-model Consensus
  descriptors and separates Divergence descriptors from the stopped Consensus
  reference.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-6


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    target = target.float()
    intersection = (probability * target).sum(dim=(1, 2, 3, 4))
    denominator = probability.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    return 1.0 - ((2.0 * intersection + EPS) / (denominator + EPS)).mean()


def hard_distillation_loss(logits: torch.Tensor, hard_target: torch.Tensor) -> torch.Tensor:
    """Unweighted BCE-Dice supervision from the Normal PAM branch."""
    # HCDD correspondence: this is the Hard term L_hard.
    bce = F.binary_cross_entropy_with_logits(logits, hard_target.float())
    return bce + dice_loss(logits, hard_target)


def core_distillation_loss(
    logits: torch.Tensor,
    core_probability: torch.Tensor,
    core_region: torch.Tensor,
) -> torch.Tensor | None:
    """Bernoulli KL on voxels jointly accepted by Dilation and Erosion."""
    # HCDD correspondence: this is the Core term L_core.
    active = core_region.float()
    if float(active.sum().item()) <= 0:
        return None
    target = core_probability.float().clamp(1e-5, 1.0 - 1e-5)
    log_foreground = F.logsigmoid(logits)
    log_background = F.logsigmoid(-logits)
    kl = target * (torch.log(target) - log_foreground)
    kl += (1.0 - target) * (torch.log(1.0 - target) - log_background)
    return (kl * active).sum() / active.sum().clamp_min(EPS)


def consensus_foreground_loss(
    logits: torch.Tensor,
    consensus_region: torch.Tensor,
) -> torch.Tensor | None:
    """Supervise only voxels accepted as Consensus foreground."""
    if float(consensus_region.sum().item()) <= 0:
        return None
    positive_bce = F.softplus(-logits)
    active = consensus_region.float()
    return (positive_bce * active).sum() / active.sum().clamp_min(EPS)


class FeatureAdapter(nn.Module):
    """Map one architecture's decoder features to a shared channel width."""

    def __init__(self, input_channels: int, hidden_dim: int) -> None:
        super().__init__()
        hidden = max(4, int(hidden_dim))
        self.network = nn.Sequential(
            nn.Conv3d(int(input_channels), hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
        )

    def forward(self, features_hwd: torch.Tensor) -> torch.Tensor:
        features = features_hwd.permute(0, 1, 4, 2, 3).contiguous()
        adapted = self.network(features)
        return adapted.permute(0, 1, 3, 4, 2).contiguous()


class SharedDescriptorEncoder(nn.Module):
    """Produce normalized dense descriptors from adapted decoder features."""

    def __init__(self, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(4, int(hidden_dim))
        embedding = max(4, int(embedding_dim))
        self.network = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv3d(hidden, embedding, kernel_size=1),
        )

    def forward(self, adapted_hwd: torch.Tensor) -> torch.Tensor:
        adapted = adapted_hwd.permute(0, 1, 4, 2, 3).contiguous()
        descriptors = self.network(adapted)
        descriptors = descriptors.permute(0, 1, 3, 4, 2).contiguous()
        return F.normalize(descriptors, p=2, dim=1)


def _pool_descriptor(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    denominator = weights.sum(dim=(2, 3, 4)).clamp_min(1.0)
    pooled = (field * weights).sum(dim=(2, 3, 4)) / denominator
    return F.normalize(pooled, p=2, dim=1)


def consensus_divergence_contrastive_loss(
    descriptor_fields: Sequence[torch.Tensor],
    consensus_components: Sequence[Sequence[torch.Tensor]],
    divergence_components: Sequence[Sequence[torch.Tensor]],
    temperature: float,
) -> torch.Tensor | None:
    """Align Consensus descriptors and separate all Divergence descriptors.

    For each model and Consensus component, the stopped mean descriptor from the
    other three models is the positive reference. Every Divergence component
    from the same model contributes to the global relative-negative mean.
    """
    # CDFT correspondence: Consensus components are cross-model positives,
    # whereas Divergence components provide relative negatives.
    if len(descriptor_fields) != 4:
        raise ValueError("CDFT contrast requires four descriptor fields.")
    losses: List[torch.Tensor] = []
    scale = max(float(temperature), 1e-4)
    for batch_index in range(descriptor_fields[0].shape[0]):
        positive_masks = list(consensus_components[batch_index])
        negative_masks = list(divergence_components[batch_index])
        if not positive_masks or not negative_masks:
            continue
        for positive_mask in positive_masks:
            model_descriptors = [
                _pool_descriptor(
                    field[batch_index : batch_index + 1],
                    positive_mask.unsqueeze(0),
                )[0]
                for field in descriptor_fields
            ]
            for model_index, descriptor in enumerate(model_descriptors):
                reference = torch.stack(
                    [
                        value.detach()
                        for index, value in enumerate(model_descriptors)
                        if index != model_index
                    ],
                    dim=0,
                ).mean(dim=0)
                reference = F.normalize(reference, p=2, dim=0).detach()
                positive_affinity = torch.exp(
                    F.cosine_similarity(descriptor, reference, dim=0) / scale
                )
                negative_affinities = []
                for negative_mask in negative_masks:
                    negative_descriptor = _pool_descriptor(
                        descriptor_fields[model_index][batch_index : batch_index + 1],
                        negative_mask.unsqueeze(0),
                    )[0]
                    negative_affinities.append(
                        torch.exp(
                            F.cosine_similarity(
                                negative_descriptor,
                                reference,
                                dim=0,
                            )
                            / scale
                        )
                    )
                negative_mean = torch.stack(negative_affinities).mean()
                losses.append(
                    -torch.log(
                        positive_affinity
                        / (positive_affinity + negative_mean + EPS)
                    )
                )
    if not losses:
        return None
    return torch.stack(losses).mean()


def sum_active_losses(losses: Sequence[torch.Tensor | None]) -> torch.Tensor:
    active = [loss for loss in losses if loss is not None]
    if not active:
        raise RuntimeError("No active optimization term was produced.")
    return torch.stack(active).sum()

"""Supervised loss for target-sector companies with known outcomes."""

import torch
from torch.nn import functional as F

from .model import ModelOutput


def bankruptcy_loss(
    output: ModelOutput,
    target_labels: torch.Tensor,
    *,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    """Binary loss; NaN labels are excluded rather than treated as non-events.

    Estimate ``positive_weight`` on the chronological training split only.
    Labels must describe outcomes *after* the snapshot's prediction date.
    """

    labels = target_labels.to(device=output.logits.device, dtype=output.logits.dtype)
    if labels.shape != output.logits.shape:
        raise ValueError("target_labels must match the number of target-sector companies")
    known = torch.isfinite(labels)
    if not known.any():
        raise ValueError("no known target outcomes")
    if not ((labels[known] == 0) | (labels[known] == 1)).all():
        raise ValueError("known labels must be zero or one")
    if positive_weight <= 0:
        raise ValueError("positive_weight must be positive")
    weight = labels.new_tensor(positive_weight)
    return F.binary_cross_entropy_with_logits(output.logits[known], labels[known], pos_weight=weight)

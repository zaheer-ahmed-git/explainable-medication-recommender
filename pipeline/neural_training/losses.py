"""Ranking losses for the neural branch.

The primary objective is a multi-positive listwise softmax cross-entropy over
each ranking group's valid candidates: every observed prescription is treated as
a positive competing against the shared candidate set. An auxiliary per-candidate
binary cross-entropy term stabilizes optimization and yields better-calibrated
probabilities, which the scoring step later temperature-calibrates. The
auxiliary term uses a per-batch ``pos_weight`` so the heavy negative majority
does not dominate the gradient.

PyTorch is imported directly; this module is only loaded when the neural branch
runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class LossOutputs:
    """Container for the combined loss and its components (for logging)."""

    total: torch.Tensor
    listwise: torch.Tensor
    auxiliary: torch.Tensor


def listwise_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the mean multi-positive listwise softmax loss over groups.

    ``logits`` padded slots are expected to be ``-inf`` (as produced by the
    model), so ``log_softmax`` assigns them zero probability. Groups without a
    positive candidate are ignored.
    """

    log_probs = functional.log_softmax(logits, dim=1)
    positive = (labels > 0.5) & candidate_mask
    positive_counts = positive.sum(dim=1)
    has_positive = positive_counts > 0

    if not bool(has_positive.any()):
        # No positive group in this batch: return a finite, grad-carrying zero.
        # ``logits`` can hold ``-inf`` padding, so a masked sum avoids NaN.
        return logits.masked_fill(~candidate_mask, 0.0).sum() * 0.0

    # ``log_probs`` at padded slots is ``-inf``; zero it out where not positive
    # before summing so masked entries never contribute.
    selected = torch.where(positive, log_probs, torch.zeros_like(log_probs))
    per_group = selected.sum(dim=1) / positive_counts.clamp(min=1)
    return -(per_group[has_positive]).mean()


def _batch_positive_weight(
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ``#neg / #pos`` over valid candidates (floored at 1.0)."""

    valid_labels = labels[candidate_mask]
    positive_count = (valid_labels > 0.5).sum().to(dtype=torch.float32)
    negative_count = (valid_labels <= 0.5).sum().to(dtype=torch.float32)
    if float(positive_count) <= 0.0:
        return valid_labels.new_tensor(1.0)
    return (negative_count / positive_count).clamp(min=1.0)


def auxiliary_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the mean binary cross-entropy over valid candidates only.

    Uses a per-batch positive class weight ``#neg / #pos`` so the majority
    negative class does not drown the gradient under full candidate lists.
    """

    if not bool(candidate_mask.any()):
        return logits.masked_fill(~candidate_mask, 0.0).sum() * 0.0
    valid_logits = logits[candidate_mask]
    valid_labels = labels[candidate_mask]
    pos_weight = _batch_positive_weight(labels, candidate_mask)
    return functional.binary_cross_entropy_with_logits(
        valid_logits,
        valid_labels,
        pos_weight=pos_weight,
    )


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    auxiliary_weight: float,
) -> LossOutputs:
    """Return the combined listwise + weighted auxiliary BCE loss."""

    listwise = listwise_softmax_loss(logits, labels, candidate_mask)
    auxiliary = auxiliary_bce_loss(logits, labels, candidate_mask)
    total = listwise + auxiliary_weight * auxiliary
    return LossOutputs(total=total, listwise=listwise, auxiliary=auxiliary)

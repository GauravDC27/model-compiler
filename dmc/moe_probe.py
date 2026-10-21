"""MoE router discovery probes (discovery-program stages S1/S2).

Measures, for any Hugging Face MoE model that can return router logits:
  * routing margins  — gap between the k-th and (k+1)-th router logit per (layer, token);
  * routing flips    — expert-set changes under a perturbation (dtype, batch-mates, ...);
  * margin -> flip   — how well a small margin predicts a flip (ROC AUC);
  * task specificity — how much expert usage depends on the task family, per layer
                       (normalized mutual information), the evidence for/against
                       per-profile expert masks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


def router_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return router logits as a (layers, batch, seq, experts) FP32 tensor."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
    if not getattr(out, "router_logits", None):
        raise ValueError("model did not return router logits; it may not be an MoE model")
    b, t = input_ids.shape
    return torch.stack([r.float().reshape(b, t, -1) for r in out.router_logits])


def topk_sets(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Sorted top-k expert indices along the last dim (ties broken by lowest index)."""
    return stable_topk(logits, k).sort(dim=-1).values


def stable_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k indices with a deterministic tie-break: equal logits prefer the lower index.

    ``torch.topk`` does not specify tie order; sorting by (-logit, index) does.
    """
    # A stable sort keeps the original (index) order among equal keys.
    return torch.sort(-logits, dim=-1, stable=True).indices[..., :k]


def margins(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Gap between the k-th and (k+1)-th largest router logit."""
    s = logits.sort(dim=-1, descending=True).values
    return s[..., k - 1] - s[..., k]


def flips(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean tensor: True where the top-k expert set differs between two logit tensors."""
    return (topk_sets(a, k) != topk_sets(b, k)).any(dim=-1)


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUC of `scores` predicting boolean `labels` (rank formulation, ties averaged)."""
    scores = scores.flatten().double()
    labels = labels.flatten().bool()
    n_pos = int(labels.sum())
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.double)
    # Average ranks over tied scores.
    uniq, inverse = torch.unique(scores, return_inverse=True)
    sums = torch.zeros(uniq.numel(), dtype=torch.double).index_add_(0, inverse, ranks)
    counts = torch.zeros(uniq.numel(), dtype=torch.double).index_add_(
        0, inverse, torch.ones_like(ranks)
    )
    ranks = (sums / counts)[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expert_usage(logits: torch.Tensor, k: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-layer expert selection counts, shape (layers, experts). `mask` is (batch, seq)."""
    layers, _, _, n_exp = logits.shape
    sel = torch.zeros(logits.shape, dtype=torch.float64)
    sel.scatter_(-1, stable_topk(logits, k), 1.0)
    if mask is not None:
        sel = sel * mask[None, :, :, None].double()
    return sel.sum(dim=(1, 2)).reshape(layers, n_exp)


def task_specificity(usage_by_task: dict[str, torch.Tensor]) -> torch.Tensor:
    """Normalized mutual information I(expert; task) / H(task), per layer.

    `usage_by_task[t]` is a (layers, experts) count matrix. 0 means routing ignores the task
    family; 1 means the experts used identify the task family exactly.
    """
    tasks = sorted(usage_by_task)
    joint = torch.stack([usage_by_task[t] for t in tasks], dim=1)  # (layers, tasks, experts)
    joint = joint / joint.sum(dim=(1, 2), keepdim=True)
    p_task = joint.sum(dim=2, keepdim=True)
    p_exp = joint.sum(dim=1, keepdim=True)
    ratio = torch.where(joint > 0, joint / (p_task * p_exp), torch.ones_like(joint))
    mi = (joint * ratio.log()).sum(dim=(1, 2))
    h_task = -(p_task.squeeze(-1) * p_task.squeeze(-1).clamp_min(1e-30).log()).sum(dim=1)
    return mi / h_task


@dataclass(frozen=True)
class MarginSummary:
    quantiles: dict[float, float]
    share_below: dict[float, float]

    @staticmethod
    def of(m: torch.Tensor, qs: Sequence[float] = (0.01, 0.05, 0.1, 0.5),
           thresholds: Sequence[float] = (1e-3, 1e-2, 1e-1)) -> "MarginSummary":
        flat = m.flatten().double()
        return MarginSummary(
            quantiles={q: float(flat.quantile(q)) for q in qs},
            share_below={t: float((flat < t).double().mean()) for t in thresholds},
        )

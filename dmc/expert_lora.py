"""LoRA on selected routed experts of a granite-style fused-expert MoE (plan item i baseline).

Mirrors the A3B recipe on a small model: each layer gets rank-r adapters only on its "hot"
experts (selected by routing counts); the router is untouched. Used as the ordinary
fine-tuning baseline against which new goal-gated capacity must show a gain (review M8).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HotExpertLoRA(nn.Module):
    """Replaces a fused ``experts`` module's forward with base + LoRA on ``hot`` experts."""

    def __init__(self, experts: nn.Module, hot: list[int], rank: int, alpha: float):
        super().__init__()
        self.base = experts
        self.hot = {e: j for j, e in enumerate(hot)}
        self.scale = alpha / rank
        n = len(hot)
        inter2, hidden = experts.gate_up_proj.shape[1], experts.gate_up_proj.shape[2]
        inter = experts.down_proj.shape[2]
        dt = experts.gate_up_proj.dtype
        self.a_gu = nn.Parameter(torch.randn(n, rank, hidden, dtype=dt) / hidden**0.5)
        self.b_gu = nn.Parameter(torch.zeros(n, inter2, rank, dtype=dt))
        self.a_d = nn.Parameter(torch.randn(n, rank, inter, dtype=dt) / inter**0.5)
        self.b_d = nn.Parameter(torch.zeros(n, hidden, rank, dtype=dt))
        # Switched per request by the compiler's goal; inactive means exactly the base experts.
        self.active = True

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if not self.active:
            return self.base(hidden_states, top_k_index, top_k_weights)
        ex = self.base
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(top_k_index).tolist():
            pos, tok = torch.where((top_k_index == e).T)
            if tok.numel() == 0:
                continue
            x = hidden_states[tok]
            gu = F.linear(x, ex.gate_up_proj[e])
            j = self.hot.get(e)
            if j is not None:
                gu = gu + self.scale * F.linear(F.linear(x, self.a_gu[j]), self.b_gu[j])
            gate, up = gu.chunk(2, dim=-1)
            h = ex.act_fn(gate) * up
            y = F.linear(h, ex.down_proj[e])
            if j is not None:
                y = y + self.scale * F.linear(F.linear(h, self.a_d[j]), self.b_d[j])
            out.index_add_(0, tok, y * top_k_weights[tok, pos, None])
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in (self.a_gu, self.b_gu, self.a_d, self.b_d))


def params_per_expert(experts: nn.Module, rank: int) -> int:
    inter2, hidden = experts.gate_up_proj.shape[1], experts.gate_up_proj.shape[2]
    inter = experts.down_proj.shape[2]
    return rank * (hidden + inter2) + rank * (inter + hidden)


def attach(model, hot: dict[int, list[int]], rank: int, alpha: float) -> list[HotExpertLoRA]:
    mods = []
    for i, experts in hot.items():
        moe = model.model.layers[i].block_sparse_moe
        m = HotExpertLoRA(moe.experts, experts, rank, alpha)
        moe.experts = m
        mods.append(m)
    return mods


def detach(model, mods: list[HotExpertLoRA], layers: list[int]) -> None:
    for i, m in zip(layers, mods):
        model.model.layers[i].block_sparse_moe.experts = m.base

"""Goal-gated expert expansion for a pretrained MoE (plan item ii: new capacity + router).

Each expanded MoE layer gets ``n_new`` new experts. Their router logits are
``w_new · h + b_goal``, where ``b_goal`` is a per-goal bias set at compile time from the
request's IR (the goal is known before execution, as in a static schedule); for a goal the
new experts are not trained for, ``b_goal = -inf``. Two variants:

* ``routed``   — the new experts join the top-k competition, displacing base experts for
                 tokens where their logit wins (the router sends goal traffic to them);
* ``additive`` — the base top-k is untouched and the new experts are added on top with a
                 sigmoid gate (like Qwen3.6's gated shared expert).

With the goal off, both variants are bit-identical to the base model: new capacity cannot
cause forgetting on other goals by construction. Only the new experts, their router rows and
the goal biases are trainable.

Implemented for granite-style MoE blocks (``router`` returning ``(top_k_index, top_k_weights,
router_logits)`` with fused ``experts(hidden, top_k_index, top_k_weights)``).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Variant = Literal["routed", "additive"]


class NewExperts(nn.Module):
    """``n`` gated-MLP experts with the base expert shapes; down-projection starts at zero."""

    def __init__(self, n: int, hidden: int, inter: int, act, init_from: torch.Tensor | None = None,
                 init_up: torch.Tensor | None = None):
        super().__init__()
        self.gate_up = nn.Parameter(torch.empty(n, 2 * inter, hidden))
        self.down = nn.Parameter(torch.zeros(n, hidden, inter))
        if init_up is not None:
            self.gate_up.data.copy_(init_up)
        else:
            nn.init.normal_(self.gate_up, std=0.02)
        if init_from is not None:
            self.down.data.copy_(init_from)
        self.act = act

    def forward(self, h: torch.Tensor, j: int) -> torch.Tensor:
        gate, up = F.linear(h, self.gate_up[j]).chunk(2, dim=-1)
        return F.linear(self.act(gate) * up, self.down[j])


class GoalGatedMoE(nn.Module):
    """Wraps a base MoE block; adds goal-gated experts with their own router rows."""

    def __init__(self, base: nn.Module, n_new: int, goals: list[str], variant: Variant,
                 warm_start: list[int] | None = None):
        super().__init__()
        self.base = base
        self.variant = variant
        self.n_new = n_new
        hidden = base.router.weight.shape[1]
        ex = base.experts
        inter = ex.down_proj.shape[2]
        init_up = init_down = None
        if warm_start is not None:  # copy the given base experts (down-projection too)
            init_up = ex.gate_up_proj.data[warm_start].clone().float()
            init_down = ex.down_proj.data[warm_start].clone().float()
        self.new = NewExperts(n_new, hidden, inter, ex.act_fn, init_down, init_up).to(ex.down_proj.dtype)
        self.router_new = nn.Parameter(torch.zeros(n_new, hidden, dtype=torch.float32))
        self.goals = {g: i for i, g in enumerate(goals)}
        self.goal_bias = nn.Parameter(torch.zeros(len(goals), n_new))
        self.active_goal: str | None = None

    def _bias(self) -> torch.Tensor | None:
        if self.active_goal is None or self.active_goal not in self.goals:
            return None
        return self.goal_bias[self.goals[self.active_goal]]

    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        bias = self._bias()
        if bias is None:
            return self.base(layer_input)  # goal off: exactly the base block
        bsz, length, emb = layer_input.shape
        h = layer_input.reshape(-1, emb)
        k = self.base.router.top_k
        base_logits = F.linear(h.float(), self.base.router.weight.float())  # (T, E)
        new_logits = F.linear(h.float(), self.router_new) + bias  # (T, n_new)
        E = base_logits.shape[-1]
        if self.variant == "routed":
            logits = torch.cat([base_logits, new_logits], dim=-1)
            top_logits, top_idx = torch.sort(-logits, dim=-1, stable=True)
            top_idx = top_idx[:, :k]
            top_w = torch.softmax(logits.gather(-1, top_idx), dim=-1).to(h.dtype)
            # New-expert slots go to base expert 0 with weight exactly 0, which adds an exact
            # zero whichever fused-experts implementation the model uses.
            is_base = top_idx < E
            base_idx = torch.where(is_base, top_idx, torch.zeros_like(top_idx))
            out = self.base.experts(h, base_idx, torch.where(is_base, top_w, torch.zeros_like(top_w)))
            for j in range(self.n_new):
                sel = top_idx == E + j  # (T, k)
                rows = sel.any(-1)
                if rows.any():
                    w = (top_w * sel).sum(-1, keepdim=True)[rows]
                    out[rows] = out[rows] + w * self.new(h[rows], j)
        else:  # additive
            top_logits, top_idx = torch.sort(-base_logits, dim=-1, stable=True)
            top_idx = top_idx[:, :k]
            top_w = torch.softmax(base_logits.gather(-1, top_idx), dim=-1).to(h.dtype)
            out = self.base.experts(h, top_idx, top_w)
            gate = torch.sigmoid(new_logits).to(h.dtype)  # (T, n_new)
            for j in range(self.n_new):
                out = out + gate[:, j:j + 1] * self.new(h, j)
        return out.view(bsz, length, emb)

    def new_expert_share(self, layer_input: torch.Tensor) -> float:
        """Share of tokens whose top-k includes a new expert (routed) or mean gate (additive)."""
        bias = self._bias()
        if bias is None:
            return 0.0
        h = layer_input.reshape(-1, layer_input.shape[-1]).float()
        base_logits = F.linear(h, self.base.router.weight.float())
        new_logits = F.linear(h, self.router_new) + bias
        if self.variant == "additive":
            return float(torch.sigmoid(new_logits).mean())
        logits = torch.cat([base_logits, new_logits], dim=-1)
        top = torch.sort(-logits, dim=-1, stable=True).indices[:, : self.base.router.top_k]
        return float((top >= base_logits.shape[-1]).any(-1).double().mean())


def expand(model, layers: list[int], n_new: int, goals: list[str], variant: Variant,
           warm_start: dict[int, list[int]] | None = None) -> list[GoalGatedMoE]:
    """Replace ``model.model.layers[i].block_sparse_moe`` with goal-gated wrappers."""
    wrapped = []
    for i in layers:
        layer = model.model.layers[i]
        base = layer.block_sparse_moe
        w = GoalGatedMoE(base, n_new, goals, variant, (warm_start or {}).get(i))
        layer.block_sparse_moe = w
        wrapped.append(w)
    return wrapped


def set_goal(wrapped: list[GoalGatedMoE], goal: str | None) -> None:
    for w in wrapped:
        w.active_goal = goal


def trainable_parameters(wrapped: list[GoalGatedMoE]) -> list[nn.Parameter]:
    return [p for w in wrapped for p in (*w.new.parameters(), w.router_new, w.goal_bias)]


def init_goal_bias(wrapped: list[GoalGatedMoE], goal: str, value: float) -> None:
    """Start a goal's bias so new experts are reachable (routed) or half-open (additive)."""
    with torch.no_grad():
        for w in wrapped:
            w.goal_bias[w.goals[goal]].fill_(value if math.isfinite(value) else -1e4)

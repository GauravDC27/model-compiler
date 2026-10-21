"""Routing traces and route pinning for Hugging Face MoE models (requirement R27).

``RouteTracer`` records, for every router call, the selected expert indices. ``RoutePinner``
replays a trace: each router call returns the recorded indices instead of its own top-k, and
recomputes the mixture weights from the current logits at those indices using the model's
own convention. With routes pinned, a different numeric stack differs from the reference
only by dense arithmetic: the discrete routing amplifier is removed (contract level D2-replay).

Two router conventions exist in transformers 5.x:
  * granite  — ``forward -> (top_k_index, top_k_weights, router_logits)``, weights are a
               softmax over the selected logits;
  * softmax  — ``forward -> (router_logits, router_scores, router_indices)``, weights are the
               full softmax gathered at the selected experts, optionally renormalized
               (Qwen2/3-MoE, OLMoE, Mixtral).
"""

from __future__ import annotations

from collections import defaultdict

import torch

_GRANITE = {"GraniteMoeTopKRouter"}
_SOFTMAX = {"Qwen3MoeTopKRouter", "Qwen2MoeTopKRouter", "OlmoeTopKRouter", "MixtralTopKRouter"}


def find_routers(model) -> list[tuple[str, torch.nn.Module, str]]:
    found = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls in _GRANITE:
            found.append((name, mod, "granite"))
        elif cls in _SOFTMAX:
            found.append((name, mod, "softmax"))
    if not found:
        raise ValueError("no supported MoE router modules found")
    return found


def _split(out, convention):
    if convention == "granite":
        idx, w, logits = out
    else:
        logits, w, idx = out
    return idx, w, logits


def _join(idx, w, logits, convention):
    return (idx, w, logits) if convention == "granite" else (logits, w, idx)


def _weights_at(mod, logits: torch.Tensor, idx: torch.Tensor, like: torch.Tensor, convention: str):
    if convention == "granite":
        return torch.softmax(logits.float().gather(-1, idx), dim=-1).type_as(like)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    w = probs.gather(-1, idx)
    if getattr(mod, "norm_topk_prob", False):
        w = w / w.sum(dim=-1, keepdim=True)
    return w.to(like.dtype)


class _Patch:
    def __init__(self, model):
        self.routers = find_routers(model)
        self._orig = {}

    def __enter__(self):
        for name, mod, conv in self.routers:
            self._orig[name] = mod.forward
            mod.forward = self._wrap(name, mod, conv, mod.forward)
        return self

    def __exit__(self, *exc):
        for name, mod, _ in self.routers:
            mod.forward = self._orig[name]
        return False


class RouteTracer(_Patch):
    """Record selected experts per router, in call order (prefill then each decode step)."""

    def __init__(self, model):
        super().__init__(model)
        self.calls: dict[str, list[torch.Tensor]] = defaultdict(list)

    def _wrap(self, name, mod, conv, fwd):
        def forward(hidden_states, *a, **kw):
            out = fwd(hidden_states, *a, **kw)
            idx, _, _ = _split(out, conv)
            self.calls[name].append(idx.detach().clone())
            return out

        return forward

    def trace(self) -> dict[str, torch.Tensor]:
        """Per-router (tokens, k) index tensor, concatenated over calls."""
        return {name: torch.cat(v) for name, v in self.calls.items()}


class RoutePinner(_Patch):
    """Force every router to the recorded experts, consuming the trace token by token."""

    def __init__(self, model, trace: dict[str, torch.Tensor]):
        super().__init__(model)
        self.trace = trace
        self.cursor = defaultdict(int)
        self.overruns = 0

    def _wrap(self, name, mod, conv, fwd):
        def forward(hidden_states, *a, **kw):
            out = fwd(hidden_states, *a, **kw)
            idx, w, logits = _split(out, conv)
            n = idx.shape[0]
            start = self.cursor[name]
            rec = self.trace[name][start:start + n]
            self.cursor[name] = start + n
            if rec.shape[0] != n:  # trace exhausted: fall back to live routing
                self.overruns += 1
                return out
            rec = rec.to(idx.device)
            return _join(rec, _weights_at(mod, logits, rec, w, conv), logits, conv)

        return forward


def trace_size_bytes(trace: dict[str, torch.Tensor]) -> int:
    """Storage cost of a trace with one byte per expert index (<=256 experts)."""
    return sum(t.numel() for t in trace.values())

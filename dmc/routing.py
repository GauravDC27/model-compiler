"""Router hardening, routing traces and route pinning for Hugging Face MoE models.

* ``harden_routers`` (requirement R7): router GEMM in FP32 regardless of the model dtype, and
  a stable top-k whose ties go to the lowest expert index. Stock BF16 checkpoints compute the
  router GEMM in BF16 and cast afterwards (granite: ``F.linear(h, W).float()``), so router
  logits carry BF16 rounding that the deterministic lane must not have.
* ``RouteTracer`` records, per router call, the selected expert indices, and the token ids
  seen by the input embedding, so a trace can be verified against the tokens it is replayed on.
* ``RoutePinner`` replays a trace (requirement R27, contract level D2-replay): every router
  returns the recorded experts and recomputes the mixture weights from its own logits with the
  model's convention. In strict mode (the default) a token mismatch or an exhausted trace
  raises instead of silently routing live.

Two router conventions exist in transformers 5.x:
  * granite  — ``forward -> (top_k_index, top_k_weights, router_logits)``; weights are a
               softmax over the selected logits;
  * softmax  — ``forward -> (router_logits, router_scores, router_indices)``; weights are the
               full softmax gathered at the selected experts, optionally renormalized
               (Qwen2/3-MoE, OLMoE, Mixtral).
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

_GRANITE = {"GraniteMoeTopKRouter"}
_SOFTMAX = {"Qwen3MoeTopKRouter", "Qwen2MoeTopKRouter", "OlmoeTopKRouter", "MixtralTopKRouter"}
TOKENS = "__tokens__"


class TraceMismatch(RuntimeError):
    """The tokens being re-executed are not the tokens the trace was recorded on."""


class TraceOverrun(RuntimeError):
    """The re-execution consumed more router calls than the trace contains."""


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


def _stable_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    return torch.sort(-logits, dim=-1, stable=True).indices[..., :k]


def _hardened_forward(mod, convention: str):
    def forward(hidden_states, *a, **kw):
        h = hidden_states.reshape(-1, mod.weight.shape[1])
        logits = F.linear(h.float(), mod.weight.float())  # FP32 GEMM, FP32 logits
        idx = _stable_topk(logits, mod.top_k)
        w = _weights_at(mod, logits, idx, hidden_states, convention)
        return _join(idx, w, logits, convention)

    return forward


class _Patch:
    def __init__(self, model):
        self.model = model
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

    def _wrap(self, name, mod, conv, fwd):
        raise NotImplementedError


class harden_routers(_Patch):
    """Context manager: FP32 router GEMM + lowest-index tie-break on every router (R7)."""

    def _wrap(self, name, mod, conv, fwd):
        return _hardened_forward(mod, conv)


class _TokenHook:
    """Forward pre-hook on the input embedding that sees every token id fed to the model."""

    def __init__(self, model, on_tokens):
        self.emb = model.get_input_embeddings()
        self.on_tokens = on_tokens
        self.handle = None

    def __enter__(self):
        def hook(_mod, args):
            self.on_tokens(args[0].detach().reshape(-1).clone())

        self.handle = self.emb.register_forward_pre_hook(hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        return False


class RouteTracer(_Patch):
    """Record selected experts per router, in call order, plus the token ids routed.

    With ``record_margins=True`` it also keeps the top-k vs (k+1) router-logit margin of every
    routing decision (for the fragile-routing share, R24).
    """

    def __init__(self, model, *, record_margins: bool = False):
        super().__init__(model)
        self.calls: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.margins: list[torch.Tensor] = []
        self.record_margins = record_margins
        self._tokens = _TokenHook(model, lambda t: self.calls[TOKENS].append(t))

    def __enter__(self):
        self._tokens.__enter__()
        return super().__enter__()

    def __exit__(self, *exc):
        self._tokens.__exit__(*exc)
        return super().__exit__(*exc)

    def _wrap(self, name, mod, conv, fwd):
        def forward(hidden_states, *a, **kw):
            out = fwd(hidden_states, *a, **kw)
            idx, _, logits = _split(out, conv)
            self.calls[name].append(idx.detach().clone())
            if self.record_margins:
                top = logits.detach().float().topk(idx.shape[-1] + 1, dim=-1).values
                self.margins.append((top[..., -2] - top[..., -1]).flatten())
            return out

        return forward

    def fragile_share(self, eps: float) -> float:
        """Share of routing decisions whose margin is below ``eps``."""
        if not self.margins:
            raise ValueError("tracer was created without record_margins=True")
        m = torch.cat(self.margins)
        return float((m < eps).double().mean())

    def trace(self) -> dict[str, torch.Tensor]:
        """Per-router (tokens, k) index tensors and the token ids, concatenated over calls."""
        return {name: torch.cat(v) for name, v in self.calls.items()}


class RoutePinner(_Patch):
    """Force every router to the recorded experts, consuming the trace token by token.

    ``strict=True`` raises ``TraceMismatch`` when the token ids differ from the recorded ones and
    ``TraceOverrun`` when the trace is exhausted. ``strict=False`` falls back to live routing and
    reports ``status`` as "partial".
    """

    def __init__(self, model, trace: dict[str, torch.Tensor], *, strict: bool = True):
        super().__init__(model)
        self.trace = trace
        self.strict = strict
        self.cursor: dict[str, int] = defaultdict(int)
        self.overruns = 0
        self.mismatches = 0
        self._tokens = _TokenHook(model, self._check_tokens)

    @property
    def status(self) -> str:
        return "full" if self.overruns == 0 and self.mismatches == 0 else "partial"

    def __enter__(self):
        self._tokens.__enter__()
        return super().__enter__()

    def __exit__(self, *exc):
        self._tokens.__exit__(*exc)
        return super().__exit__(*exc)

    def _check_tokens(self, ids: torch.Tensor) -> None:
        if TOKENS not in self.trace:
            return
        start = self.cursor[TOKENS]
        rec = self.trace[TOKENS][start:start + ids.numel()]
        self.cursor[TOKENS] = start + ids.numel()
        if rec.numel() != ids.numel() or not torch.equal(rec, ids.to(rec.device)):
            self.mismatches += 1
            if self.strict:
                raise TraceMismatch(f"token ids differ from the trace at offset {start}")

    def _wrap(self, name, mod, conv, fwd):
        def forward(hidden_states, *a, **kw):
            out = fwd(hidden_states, *a, **kw)
            idx, w, logits = _split(out, conv)
            n = idx.shape[0]
            start = self.cursor[name]
            rec = self.trace[name][start:start + n]
            self.cursor[name] = start + n
            if rec.shape[0] != n:
                self.overruns += 1
                if self.strict:
                    raise TraceOverrun(f"trace for {name} exhausted at token {start}")
                return out
            rec = rec.to(idx.device)
            return _join(rec, _weights_at(mod, logits, rec, w, conv), logits, conv)

        return forward


def trace_size_bytes(trace: dict[str, torch.Tensor]) -> int:
    """Storage cost of the routing part of a trace at one byte per expert index (<=256 experts)."""
    return sum(t.numel() for name, t in trace.items() if name != TOKENS)

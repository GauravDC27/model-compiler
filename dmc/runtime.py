"""Execution backends for the deterministic lane.

A backend exposes two primitives that profiles are built from:
  * ``score(prompt, continuations)`` — log P(continuation | prompt) for each candidate, used
    for every decision-bearing value (labels, tool names, verdicts): one forward pass per
    candidate, no sampling path (requirement R17);
  * ``generate(prompt, max_new_tokens, seed)`` — greedy when ``seed is None``; otherwise
    Gumbel-max sampling whose noise is a pure function of (seed, position), so a request's
    sample never depends on what else is in the batch (requirement R9).

``HFBackend`` runs one request at a time at a fixed shape (no padding, no batch-mates),
which is the CPU analogue of a batch-invariant lane: CPU eager kernels are not
batch-invariant (experiments/00 measured 1e-5 logit drift between batch sizes).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol


class Backend(Protocol):
    name: str

    def score(self, prompt: str, continuations: Sequence[str]) -> list[float]: ...

    def generate(self, prompt: str, max_new_tokens: int, seed: int | None = None) -> str: ...

    def render_chat(self, system: str, user: str) -> str: ...

    def lock_fields(self) -> dict[str, object]: ...


def position_seed(seed: int, position: int) -> int:
    """Counter-based seed for one sampling step: independent of batch, thread and history."""
    digest = hashlib.sha256(f"{seed}:{position}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class HFBackend:
    """Hugging Face transformers on CPU, FP32, one request per forward pass."""

    def __init__(self, model_id: str, *, dtype: str = "float32", threads: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.use_deterministic_algorithms(True)
        if threads is not None:
            torch.set_num_threads(threads)
        self._torch = torch
        self.name = model_id
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype)
        ).eval()

    def render_chat(self, system: str, user: str) -> str:
        """Render with an explicit system message so the template reads no ambient state.

        Some chat templates inject the wall-clock date when no system message is given
        (granite-3.1 does, via ``strftime_now``); an explicit system turn suppresses that.
        """
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def score(self, prompt: str, continuations: Sequence[str]) -> list[float]:
        torch = self._torch
        base = self._ids(prompt)
        out = []
        for cont in continuations:
            # Tokenize jointly so the boundary matches what generation would produce.
            ids = self._ids(prompt + cont)
            if ids[: len(base)] != base:
                ids = base + self._ids(cont)
            target = ids[len(base):]
            with torch.no_grad():
                logits = self.model(input_ids=torch.tensor([ids])).logits[0].float()
            logp = logits.log_softmax(-1)
            out.append(float(sum(logp[len(base) - 1 + i, t] for i, t in enumerate(target))))
        return out

    def generate(self, prompt: str, max_new_tokens: int, seed: int | None = None) -> str:
        torch = self._torch
        ids = self._ids(prompt)
        start = len(ids)
        past = None
        inp = torch.tensor([ids])
        eos = self.tok.eos_token_id
        for step in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            if seed is None:
                nxt = int(logits.argmax())
            else:
                g = torch.Generator().manual_seed(position_seed(seed, start + step))
                u = torch.rand(logits.shape, generator=g, dtype=torch.float64)
                gumbel = -torch.log(-torch.log(u.clamp_min(1e-300)))
                nxt = int((logits.double().log_softmax(-1) + gumbel).argmax())
            if nxt == eos:
                break
            ids.append(nxt)
            inp = torch.tensor([[nxt]])
        return self.tok.decode(ids[start:], skip_special_tokens=True)

    def lock_fields(self) -> dict[str, object]:
        import transformers

        return {
            "engine": "transformers",
            "engine_version": transformers.__version__,
            "torch_version": self._torch.__version__,
            "device": "cpu",
            "dtype": self.dtype,
            "threads": self._torch.get_num_threads(),
            "deterministic_algorithms": True,
            "batching": "none (fixed shape, one request per pass)",
        }


class TableBackend:
    """A deterministic fake backend for tests: scores and generations come from tables."""

    def __init__(self, scores: dict[tuple[str, str], float], generations: dict[str, str] | None = None):
        self.name = "table"
        self.scores = scores
        self.generations = generations or {}

    def score(self, prompt: str, continuations: Sequence[str]) -> list[float]:
        return [self.scores.get((prompt, c), -50.0) for c in continuations]

    def generate(self, prompt: str, max_new_tokens: int, seed: int | None = None) -> str:
        return self.generations.get(prompt, "")

    def render_chat(self, system: str, user: str) -> str:
        return f"[system]{system}[user]{user}[assistant]"

    def lock_fields(self) -> dict[str, object]:
        return {"engine": "table", "engine_version": "0", "device": "none"}

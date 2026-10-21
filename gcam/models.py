"""The hub and the spoke as language models, and the hub's goal-family capacity.

``LM`` wraps a causal LM on the deterministic CPU lane (``dmc.runtime.HFBackend``: R7 router
hardening, fixed shapes, one request per pass) with the three primitives executors need:
multiple-choice scoring over single-token letters, seeded generation, and hidden states.

``Capacity`` installs the hub's goal-conditioned capacity (plan §I.2.3), switched per request by
the compiler's goal:

* goal-gated new experts (item ii, ``dmc.expansion``, routed variant) — trained for the
  ``policy`` family;
* hot-expert LoRA (item i, ``dmc.expert_lora``) — trained for the ``support`` family.

With no family set, every layer is the base layer: the gated wrapper returns ``base(h)`` and the
LoRA modules delegate to the base experts, so the output is bit-identical to the base model.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from dmc import expansion
from dmc.expert_lora import HotExpertLoRA
from dmc.runtime import HFBackend, position_seed

HUB_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
SPOKE_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LETTERS = "ABCDEFGH"


class LM:
    def __init__(self, model_id: str, role: str, *, threads: int | None = None):
        self.backend = HFBackend(model_id, dtype="float32", threads=threads)
        self.model = self.backend.model
        self.tok = self.backend.tok
        self.role = role  # "hub" or "spoke": the cost-weight key
        self.model_id = model_id
        self.capacity: Capacity | None = None

    # ---------------------------------------------------------------- primitives
    def n_tokens(self, text: str) -> int:
        return len(self.backend._ids(text))

    def render(self, system: str, user: str) -> str:
        return self.backend.render_chat(system, user, None)

    def choice_logprobs(self, prompt: str, letters: str) -> tuple[np.ndarray, int]:
        """Log-probabilities of each single-token option letter as the reply's first token."""
        ids = self.backend._ids(prompt)
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([ids])).logits[0, -1].float().log_softmax(-1)
        letter_ids = [self.backend._ids(prompt + c)[len(ids)] if self.backend._ids(prompt + c)[:len(ids)] == ids
                      else self.backend._ids(c)[0] for c in letters]
        return logits[letter_ids].numpy().astype(np.float64), len(ids)

    def generate(self, prompt: str, max_new_tokens: int, *, seed: int | None = None, temperature: float = 0.7,
                 top_k: int = 40) -> tuple[str, list[int], float]:
        """Greedy (seed None) or seeded top-k sampling with counter-based Gumbel noise (R9).

        Returns the text, the generated ids and their mean log-probability under the model
        (used to rank best-of-K candidates)."""
        ids = self.backend._ids(prompt)
        start = len(ids)
        past = None
        inp = torch.tensor([ids])
        eos = {self.tok.eos_token_id} | ({self.tok.pad_token_id} if self.tok.pad_token_id is not None else set())
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in self.tok.get_vocab() else None
        if im_end is not None:
            eos.add(im_end)
        out, lps = [], []
        for step in range(max_new_tokens):
            with torch.no_grad():
                o = self.model(input_ids=inp, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logp = o.logits[0, -1].double().log_softmax(-1)
            if seed is None:
                nxt = int(torch.argmax(logp))
            else:
                # Stable truncation: probability descending, token id ascending on ties (R9).
                vals, idx = torch.sort(-logp, stable=True)
                keep = idx[:top_k]
                g = torch.Generator().manual_seed(position_seed(seed, start + step))
                u = torch.rand(len(keep), generator=g, dtype=torch.float64)
                gumbel = -torch.log(-torch.log(u.clamp_min(1e-300)))
                nxt = int(keep[int(torch.argmax(logp[keep] / temperature + gumbel))])
            if nxt in eos:
                break
            out.append(nxt)
            lps.append(float(logp[nxt]))
            inp = torch.tensor([[nxt]])
        text = self.tok.decode(out, skip_special_tokens=True).strip()
        return text, out, (sum(lps) / len(lps) if lps else -math.inf)

    def hidden_mean(self, text: str, layer: int) -> tuple[np.ndarray, int]:
        ids = self.backend._ids(text)
        with torch.no_grad():
            hs = self.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
        return hs[layer + 1][0].mean(0).numpy().astype(np.float64), len(ids)

    @contextmanager
    def family(self, name: str | None):
        """Activate one goal family's capacity for the duration of a call."""
        if self.capacity is None:
            yield
            return
        prev = self.capacity.active
        self.capacity.set(name)
        try:
            yield
        finally:
            self.capacity.set(prev)


class Capacity:
    """Goal-gated experts and hot-expert LoRA installed in one hub, switched per family."""

    def __init__(self, lm: LM):
        self.lm = lm
        self.gated: dict[str, list[expansion.GoalGatedMoE]] = {}  # family -> wrappers
        self.lora: dict[str, list[HotExpertLoRA]] = {}  # family -> modules
        self.active: str | None = None

    @property
    def layers(self):
        return self.lm.model.model.layers

    def base_moe(self, i: int):
        moe = self.layers[i].block_sparse_moe
        return moe.base if isinstance(moe, expansion.GoalGatedMoE) else moe

    def add_goal_experts(self, family: str, parents: dict[int, list[int]], *, variant: str = "additive",
                         n_new: int = 1, noise: float = 0.05, bias_init: float | None = None,
                         seed: int = 11) -> list[expansion.GoalGatedMoE]:
        """Goal-gated new experts for one family.

        ``additive`` (v1 default): every token keeps its base top-8 and also passes through the new
        expert, scaled by a learned sigmoid gate (router row + goal bias). The new expert starts as
        a copy of the layer's hottest expert with its down-projection zeroed, so at step 0 the
        model is exactly the base model. ``routed``: the new expert competes inside the top-8 and
        starts as a noisy copy of its parent (UMoE-style). On this hub the routed copies lost their
        traffic during training (share 0.74 → 0.30, 0.13 at the answer token), so v1 uses additive.
        """
        if self.gated:
            raise ValueError("v1 supports goal-gated experts for one family")
        gen = torch.Generator().manual_seed(seed)
        if variant == "routed":
            wrapped = expansion.expand(self.lm.model, sorted(parents), n_new, [family], "routed", parents,
                                       noise=noise, router_init="parent",
                                       bias_init=0.1 if bias_init is None else bias_init, generator=gen)
        else:
            wrapped = expansion.expand(self.lm.model, sorted(parents), n_new, [family], "additive", parents,
                                       router_init="zero", bias_init=0.0 if bias_init is None else bias_init)
            with torch.no_grad():
                for w in wrapped:
                    w.new.down.zero_()
        self.variant = variant
        self.gated[family] = wrapped
        return wrapped

    def add_lora(self, family: str, hot: dict[int, list[int]], *, rank: int = 4, alpha: float = 8.0) -> list[HotExpertLoRA]:
        mods = []
        for i, experts in sorted(hot.items()):
            base = self.base_moe(i)
            inner = base.experts.base if isinstance(base.experts, HotExpertLoRA) else base.experts
            m = HotExpertLoRA(inner, experts, rank, alpha)
            m.active = False
            base.experts = m
            mods.append(m)
        self.lora[family] = mods
        return mods

    def set(self, family: str | None) -> None:
        for fam, wrapped in self.gated.items():
            expansion.set_goal(wrapped, family if fam == family else None)
        for fam, mods in self.lora.items():
            for m in mods:
                m.active = fam == family
        self.active = family

    def state(self, family: str) -> dict[str, torch.Tensor]:
        if family in self.gated:
            return {f"{i}.{k}": v.detach().clone() for i, w in enumerate(self.gated[family])
                    for k, v in (("gate_up", w.new.gate_up), ("down", w.new.down), ("router", w.router_new),
                                 ("bias", w.goal_bias))}
        return {f"{i}.{k}": getattr(m, k).detach().clone() for i, m in enumerate(self.lora[family])
                for k in ("a_gu", "b_gu", "a_d", "b_d")}

    def load(self, family: str, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            if family in self.gated:
                for i, w in enumerate(self.gated[family]):
                    w.new.gate_up.copy_(state[f"{i}.gate_up"])
                    w.new.down.copy_(state[f"{i}.down"])
                    w.router_new.copy_(state[f"{i}.router"])
                    w.goal_bias.copy_(state[f"{i}.bias"])
            else:
                for i, m in enumerate(self.lora[family]):
                    for k in ("a_gu", "b_gu", "a_d", "b_d"):
                        getattr(m, k).copy_(state[f"{i}.{k}"])


def artifacts_dir() -> Path:
    return Path(os.environ.get("GCAM_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts" / "gcam"))

"""Decide profile: closed-set decisions by candidate scoring, never by sampling text.

Execution (plan section 5.1):
  1. render the IR with a versioned template and an explicit system message;
  2. score every label as a continuation (one forward pass per label and per ordering);
  3. average log-probabilities over a fixed set of label orderings, which cancels
     option-order bias and makes the result invariant to the order labels were given in;
  4. confidence = softmax(scores / T) with a calibrated temperature T;
  5. if the top-1 vs top-2 log-odds margin is below the abstain threshold, return ABSTAIN.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..ir import DecideClassify
from ..runtime import Backend

TEMPLATE_VERSION = "decide.t0"
SYSTEM = "You are a careful decision function. Reply with exactly one of the allowed labels."


class DecisionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["decided", "abstained"]
    label: str | None
    confidence: float
    margin: float  # log-odds between the best and second-best label
    scores: dict[str, float]


def render_user(op: DecideClassify, labels: tuple[str, ...]) -> str:
    lines = [f"Question: {op.question}"]
    if op.policy:
        lines.append(f"Policy: {op.policy}")
    for ev in op.evidence:
        lines.append(f"Evidence: {ev}")
    lines.append("Allowed labels: " + ", ".join(labels))
    lines.append("Answer with one label.")
    return "\n".join(lines)


def surface_forms(label: str) -> tuple[str, ...]:
    """Case variants a model may use to say the same label ("no", "No", "NO")."""
    return tuple(dict.fromkeys((label, label.capitalize(), label.upper())))


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def label_orderings(labels: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Canonical order and its reverse: cheap, fixed, and symmetric across positions."""
    canon = tuple(sorted(labels))
    rev = tuple(reversed(canon))
    return [canon] if rev == canon else [canon, rev]


class DecideProfile:
    name = "decide"
    result_type = DecisionResult

    def __init__(
        self,
        backend: Backend,
        *,
        temperature: float = 1.0,
        abstain_margin: float = 0.5,
        marginalize_case: bool = False,
    ):
        self.backend = backend
        self.temperature = temperature
        self.abstain_margin = abstain_margin
        # Score a label as the total probability of its case variants, so a model that
        # prefers "No" over "no" is not read as unsure (verbalizer sensitivity).
        self.marginalize_case = marginalize_case

    @property
    def version(self) -> str:
        return (
            f"{TEMPLATE_VERSION};T={self.temperature};abstain={self.abstain_margin}"
            f";case={int(self.marginalize_case)}"
        )

    def _label_scores(self, prompt: str, order: tuple[str, ...]) -> list[float]:
        if not self.marginalize_case:
            return self.backend.score(prompt, list(order))
        forms = [surface_forms(lab) for lab in order]
        flat = self.backend.score(prompt, [f for fs in forms for f in fs])
        out, i = [], 0
        for fs in forms:
            out.append(_logsumexp(flat[i:i + len(fs)]))
            i += len(fs)
        return out

    def run(self, op: DecideClassify) -> DecisionResult:
        totals = {lab: 0.0 for lab in op.labels}
        orderings = label_orderings(op.labels)
        for order in orderings:
            prompt = self.backend.render_chat(SYSTEM, render_user(op, order))
            for lab, s in zip(order, self._label_scores(prompt, order)):
                totals[lab] += s / len(orderings)
        # Deterministic ranking: score descending, then label text ascending on exact ties.
        ranked = sorted(totals, key=lambda lab: (-totals[lab], lab))
        margin = totals[ranked[0]] - totals[ranked[1]]
        z = [totals[lab] / self.temperature for lab in ranked]
        m = max(z)
        probs = [math.exp(v - m) for v in z]
        confidence = probs[0] / sum(probs)
        decided = margin >= self.abstain_margin
        return DecisionResult(
            status="decided" if decided else "abstained",
            label=ranked[0] if decided else None,
            confidence=confidence,
            margin=margin,
            scores=totals,
        )

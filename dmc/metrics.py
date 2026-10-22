"""Determinism and output-parity metrics.

Definitions follow the literature surveyed in research_notes/:
  * TARr@N / TARa@N  — total agreement rate over N repeated runs on raw outputs / parsed
    answers (Atil et al., "Non-Determinism of 'Deterministic' LLM Settings").
  * unique@N, first-divergence index — Thinking Machines, "Defeating Nondeterminism in
    LLM Inference" (80 unique completions in 1000 runs, first divergence at token 103).
  * paraphrase invariance — share of paraphrase groups whose answers all agree.
  * negative flips — MUSCLE-style release-to-release regressions.
  * router margin / routing flip rate — MoE-specific fragility measures introduced here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any


def unique_at_n(outputs: Sequence[Hashable]) -> int:
    """Number of distinct outputs among N runs of one input (1 == fully repeatable)."""
    return len(set(outputs))


def total_agreement_rate(runs_per_item: Sequence[Sequence[Hashable]]) -> float:
    """Fraction of items whose N runs are all identical (TAR@N)."""
    if not runs_per_item:
        return 1.0
    return sum(1 for runs in runs_per_item if len(set(runs)) <= 1) / len(runs_per_item)


def tar_raw(runs_per_item: Sequence[Sequence[str]]) -> float:
    """TARr@N: agreement on raw output strings."""
    return total_agreement_rate(runs_per_item)


def tar_answer(
    runs_per_item: Sequence[Sequence[str]], parse: Callable[[str], Hashable]
) -> float:
    """TARa@N: agreement on parsed answers (e.g. the label, the tool call, the number)."""
    return total_agreement_rate([[parse(r) for r in runs] for runs in runs_per_item])


def first_divergence(a: Sequence[Any], b: Sequence[Any]) -> int | None:
    """Index of the first differing token, or None when the sequences are identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def paraphrase_invariance(groups: Sequence[Sequence[Hashable]]) -> float:
    """Share of paraphrase groups whose answers are all equal (semantic determinism)."""
    return total_agreement_rate(groups)


def majority_share(outputs: Sequence[Hashable]) -> float:
    """Share of runs agreeing with the modal output (a softer repeatability measure)."""
    if not outputs:
        return 1.0
    return Counter(outputs).most_common(1)[0][1] / len(outputs)


@dataclass(frozen=True)
class FlipReport:
    """Release-to-release parity on a golden set."""

    negative_flips: int  # old correct -> new wrong
    positive_flips: int  # old wrong -> new correct
    changed: int  # outputs that differ at all
    total: int

    @property
    def negative_flip_rate(self) -> float:
        return self.negative_flips / self.total if self.total else 0.0


def flip_report(
    old: Sequence[Hashable], new: Sequence[Hashable], gold: Sequence[Hashable]
) -> FlipReport:
    if not (len(old) == len(new) == len(gold)):
        raise ValueError("old, new and gold must be aligned")
    neg = sum(1 for o, n, g in zip(old, new, gold) if o == g and n != g)
    pos = sum(1 for o, n, g in zip(old, new, gold) if o != g and n == g)
    changed = sum(1 for o, n in zip(old, new) if o != n)
    return FlipReport(neg, pos, changed, len(gold))


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Standard equal-width-bin ECE."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be aligned")
    if not confidences:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, ok))
    total = len(confidences)
    ece = 0.0
    for b in bins:
        if b:
            avg_conf = sum(c for c, _ in b) / len(b)
            acc = sum(1 for _, ok in b if ok) / len(b)
            ece += len(b) / total * abs(avg_conf - acc)
    return ece

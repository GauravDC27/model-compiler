"""Conformance harness: run the same requests under execution variations and report parity.

A *variation* is anything that must not change the output under the claimed contract level:
repeat runs, batch-mates, thread counts, chunk sizes (D1); dtype, engine version, hardware
SKU (D2, decision fields only); paraphrases (D3). The harness is backend-agnostic: callers
pass a function ``run(request, variant) -> output``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field

from .metrics import first_divergence, total_agreement_rate, unique_at_n


@dataclass
class ConformanceReport:
    variants: list[str]
    agreement_rate: float  # share of requests whose output is identical across all variants
    max_unique: int  # worst-case number of distinct outputs for one request
    disagreements: list[dict] = field(default_factory=list)

    def passed(self) -> bool:
        return self.agreement_rate == 1.0


def check(
    requests: Sequence[str],
    variants: Sequence[str],
    run: Callable[[str, str], Hashable],
    *,
    tokens: Callable[[Hashable], Sequence] | None = None,
) -> ConformanceReport:
    """Run every request under every variant; collect agreement and divergence details."""
    per_request = []
    disagreements = []
    for req in requests:
        outs = [run(req, v) for v in variants]
        per_request.append(outs)
        if len(set(outs)) > 1:
            detail: dict = {"request": req, "outputs": dict(zip(variants, map(str, outs)))}
            if tokens is not None:
                ref = tokens(outs[0])
                detail["first_divergence"] = {
                    v: first_divergence(ref, tokens(o)) for v, o in zip(variants[1:], outs[1:])
                }
            disagreements.append(detail)
    return ConformanceReport(
        variants=list(variants),
        agreement_rate=total_agreement_rate(per_request),
        max_unique=max((unique_at_n(o) for o in per_request), default=1),
        disagreements=disagreements,
    )

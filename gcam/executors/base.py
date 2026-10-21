"""Executor contract (plan §I.2.2): what every executor returns, and how its cost is counted.

Every executor, whether compiled code, a GBDT, an encoder head or an LLM, answers with an
``ExecResult``: a typed value, whether it answered at its calibrated threshold, a confidence
and its token usage per model. ``forced=True`` means it is the last rung of its ladder and must
answer.

Cost is counted in **hub-token equivalents**: raw tokens weighted by active parameters per
token relative to the hub (granite-3.1-1b-a400m, 0.4B active). This is the CPU analogue of the
plan's cost basis r: the spoke (Qwen2.5-1.5B, dense) costs 3.75 per token, the MiniLM encoder
(22.7M parameters) 0.057, code and classic ML zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

COST_WEIGHT = {"hub": 1.0, "spoke": 1.5 / 0.4, "encoder": 0.0227 / 0.4}


def cost_of(usage: dict[str, int]) -> int:
    return math.ceil(sum(COST_WEIGHT[m] * n for m, n in usage.items()))


@dataclass
class ExecResult:
    answered: bool
    value: Any
    confidence: float | None = None
    usage: dict[str, int] = field(default_factory=dict)  # raw tokens per model
    detail: dict = field(default_factory=dict)

    @property
    def cost(self) -> int:
        return cost_of(self.usage)


class Impl(Protocol):
    name: str
    version: str

    def run(self, op: str, payload: dict, *, forced: bool) -> ExecResult: ...


class NotSupported(KeyError):
    pass

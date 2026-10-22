"""Goal IR: what a natural-language goal compiles to before planning (plan §I.2.1).

A goal is a family (which plan template applies), typed slots, an explicit context and an
autonomy level. Context minimization (R12) applies: ``now`` and the world snapshot hash are
present only for families whose answer depends on them, so time-independent goals keep a
stable hash. Ambiguity never becomes a guess: the parser returns ``Clarify`` instead.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from dmc.canonical import content_hash
from dmc.ir import Context

Autonomy = Literal["read_only", "confirm", "auto"]

# Families whose answer depends on the clock and on the world snapshot.
TIME_DEPENDENT = frozenset({"refund_decision", "support_request", "write_reply", "refund_analytics",
                            "expense_review", "forecast_orders"})
DATA_DEPENDENT = TIME_DEPENDENT | {"compare_replies", "policy_question"}

# Capability ceiling per autonomy level (fixed at compile time; plan §I.2.1).
CEILING = {
    "read_only": frozenset({"read"}),
    "confirm": frozenset({"read", "write", "destructive"}),
    "auto": frozenset({"read", "write", "destructive", "external_send"}),
}


class Goal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str
    text: str
    slots: dict[str, Any]
    context: Context
    autonomy: Autonomy = "confirm"

    @property
    def hash(self) -> str:
        return content_hash(self.model_dump())

    @property
    def ceiling(self) -> frozenset[str]:
        return CEILING[self.autonomy]


class Clarify(BaseModel):
    """Defined failure (D⊥) for an ambiguous or unsupported goal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    candidates: tuple[str, ...] = ()
    text: str = ""

"""Parse: natural-language goal → Goal IR, or Clarify (plan §I.2.1, step 1).

v1 parses with a versioned grammar: one rule per supported goal form, each a pure function of
the text. That makes the front end D1 by construction. Anything outside the grammar, or a
supported form with a missing slot that cannot be resolved from the world, is a ``Clarify``
with candidates, never a guess. (The plan's dense parser spoke replaces this grammar in P2;
the IR it must produce is the one defined here.)
"""

from __future__ import annotations

import re
from collections.abc import Callable

from dmc.ir import Context

from .domain import NOW, World
from .goal_ir import DATA_DEPENDENT, TIME_DEPENDENT, Autonomy, Clarify, Goal

GRAMMAR_VERSION = "gcam.grammar/1"

Q = r'"(?P<{name}>[^"]+)"'


def _q(name: str) -> str:
    return Q.format(name=name)


ISSUE_TO_REASON = {"arrived damaged": "damaged", "has not arrived": "not_received", "was the wrong item": "wrong_item"}

RULES: list[tuple[str, re.Pattern]] = [
    ("refund_decision", re.compile(
        rf"^Customer (?P<customer>C-\d{{4}}) requests a refund for order (?P<order>A-\d{{4}}): {_q('claim')}$")),
    ("refund_decision", re.compile(
        rf"^Refund request from (?P<customer>C-\d{{4}}) for (?P<order>A-\d{{4}}): {_q('claim')}$")),
    ("support_request", re.compile(rf"^Handle (?:this )?request from (?P<customer>C-\d{{4}}): {_q('request')}$")),
    ("compare_replies", re.compile(
        rf"^Which reply to (?P<customer>C-\d{{4}}) about order (?P<order>A-\d{{4}}) is better\? "
        rf"A: {_q('a')} B: {_q('b')}$")),
    ("write_reply", re.compile(
        r"^Write a reply to (?P<customer>C-\d{4}) about order (?P<order>A-\d{4}) that "
        r"(?P<issue>arrived damaged|has not arrived|was the wrong item)$")),
    ("refund_analytics", re.compile(
        r"^Which (?:product )?category had the highest refund rate in the last (?P<days>\d{1,3}) days\??$")),
    ("cost_estimate", re.compile(r"^Estimate: (?P<question>.+)$")),
    ("expense_review", re.compile(r"^Review (?:the )?expense reports from the last (?P<days>\d{1,3}) days$")),
    ("forecast_orders", re.compile(r"^Forecast daily orders for the next (?P<days>\d{1,2}) days$")),
    ("policy_question", re.compile(r"^What does (?:our|the) policy say about (?P<topic>.+?)\??$")),
]
# Supported forms with a missing slot: recognised, but clarified rather than guessed.
PARTIAL = [
    ("refund_decision", re.compile(rf"^Customer (?P<customer>C-\d{{4}}) requests a refund: {_q('claim')}$")),
]

SUPPORTED = (
    'Customer C-1042 requests a refund for order A-5531: "<claim>"',
    'Handle request from C-1042: "<request>"',
    'Which reply to C-1042 about order A-5531 is better? A: "<reply>" B: "<reply>"',
    "Write a reply to C-1042 about order A-5531 that arrived damaged",
    "Which category had the highest refund rate in the last 90 days?",
    "Estimate: <cost question>",
    "Review expense reports from the last 7 days",
    "Forecast daily orders for the next 7 days",
    "What does our policy say about <topic>?",
)


def _normalize(text: str) -> str:
    """NFC-safe normalization of quotes and whitespace (deterministic code, plan step 1)."""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def _slots(family: str, m: re.Match) -> dict:
    s = {k: v for k, v in m.groupdict().items() if v is not None}
    for k in ("days",):
        if k in s:
            s[k] = int(s[k])
    if family == "write_reply":
        s["reason"] = ISSUE_TO_REASON[s.pop("issue")]
    return s


def _check(family: str, slots: dict, world: World) -> Clarify | None:
    if "customer" in slots and world.customer(slots["customer"]) is None:
        return Clarify(question=f"No customer {slots['customer']} exists. Which customer?")
    if "order" in slots:
        order = world.order(slots["order"])
        if order is None:
            return Clarify(question=f"No order {slots['order']} exists. Which order?")
        if "customer" in slots and order["customer_id"] != slots["customer"]:
            return Clarify(question=f"Order {slots['order']} belongs to another customer. Which order?",
                           candidates=_recent_orders(world, slots["customer"]))
    if "days" in slots and not 1 <= slots["days"] <= (30 if family == "forecast_orders" else 180):
        return Clarify(question=f"{slots['days']} days is outside the supported range. How many days?")
    return None


def _recent_orders(world: World, customer: str, k: int = 3) -> tuple[str, ...]:
    rows = world.orders[world.orders.customer_id == customer].sort_values("order_date", ascending=False)
    return tuple(rows.order_id.head(k))


def parse(text: str, world: World, *, now: str = NOW, autonomy: Autonomy = "confirm") -> Goal | Clarify:
    text = _normalize(text)
    for family, rx in RULES:
        m = rx.match(text)
        if not m:
            continue
        slots = _slots(family, m)
        problem = _check(family, slots, world)
        if problem is not None:
            return problem.model_copy(update={"text": text})
        context = Context(
            now=now if family in TIME_DEPENDENT else None, locale="en-US", timezone="UTC",
            session_state_hash=world.snapshot_hash() if family in DATA_DEPENDENT else None,
        )
        return Goal(family=family, text=text, slots=slots, context=context, autonomy=autonomy)
    for family, rx in PARTIAL:
        m = rx.match(text)
        if m and family == "refund_decision":
            return Clarify(question="Which order is the refund for?", text=text,
                           candidates=_recent_orders(world, m["customer"]))
    return Clarify(question="This goal is not in the supported grammar. Rephrase it as one of these forms.",
                   candidates=SUPPORTED, text=text)


Parser = Callable[[str, World], Goal | Clarify]

"""Plan: Goal IR → task graph of typed nodes, guards and pre-compiled variants (plan §I.2.1, step 2).

v1 plans with versioned templates, one per goal family, cached by template hash and reused with
slot substitution (plan caching). A template fixes, for every node: its op, accuracy bar,
required determinism contract, input cap, capabilities and how its payload is built from the
goal's slots and upstream outputs. The scheduler, not the template, picks executors.

The expense-review template is expanded at compile time over the snapshotted rows it covers:
the row set is data the compiler can see, so the whole program is known before it runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from dmc.canonical import content_hash
from dmc.scheduler import Guard, Node, Program

from .domain import NOW_DT, World
from .goal_ir import Goal

TEMPLATE_VERSION = "gcam.plans/1"

Builder = Callable[[Goal, dict[str, Any], World], dict]


@dataclass(frozen=True)
class NodePlan:
    node: Node
    build: Builder


@dataclass(frozen=True)
class GoalPlan:
    family: str
    nodes: dict[str, NodePlan]  # main and variant nodes by id
    program: Program
    result: tuple[str, ...]  # node ids whose outputs form the goal's result
    template_hash: str
    notes: dict = field(default_factory=dict)


def expense_rows(goal: Goal, world: World) -> list[dict]:
    since = (NOW_DT - timedelta(days=goal.slots["days"])).date().isoformat()
    rows = world.expenses[world.expenses.date >= since].sort_values("expense_id")
    return [_plain(r) for r in rows.to_dict(orient="records")]


def _plain(row: dict) -> dict:
    """JSON-safe copy of a pandas row (numpy scalars → Python)."""
    out = {}
    for k, v in row.items():
        out[k] = v.item() if hasattr(v, "item") else v
    return out


def _node(nid: str, op: str, deps: tuple[str, ...] = (), *, bar: float = 0.99, cap: int = 0,
          contract: str = "D0", caps: frozenset[str] = frozenset({"read"}), group: str | None = None) -> Node:
    return Node(nid, op, deps, required_accuracy=bar, prefix_group=group, requires_contract=contract,
                input_cap=cap, capabilities=caps)


def _plan_refund(goal: Goal, world: World) -> GoalPlan:
    nodes = {
        "order": NodePlan(_node("order", "tool.get_order", contract="D1"),
                          lambda g, o, w: {"order_id": g.slots["order"]}),
        "reason": NodePlan(_node("reason", "decide.claim_reason", bar=0.92, cap=96),
                           lambda g, o, w: {"text": g.slots["claim"]}),
        "decision": NodePlan(_node("decision", "decide.refund", ("order", "reason"), contract="D1"),
                             lambda g, o, w: {"order": o["order"], "reason": o["reason"], "now": g.context.now}),
    }
    guards = (Guard("order_found", "order", "order_found", "abstain"),)
    return _assemble(goal, nodes, guards, {}, ("reason", "decision"), frozenset({"read"}))


def _plan_support(goal: Goal, world: World) -> GoalPlan:
    effect = frozenset({"read", "write", "destructive"})  # the most any allowed tool needs
    nodes = {
        "tool": NodePlan(_node("tool", "tool.select", bar=0.92, cap=64),
                         lambda g, o, w: {"text": g.slots["request"]}),
        "args": NodePlan(_node("args", "tool.args", ("tool",), contract="D1"),
                         lambda g, o, w: {"tool": o["tool"], "text": g.slots["request"],
                                          "customer": g.slots["customer"], "now": g.context.now}),
        "confirm": NodePlan(_node("confirm", "meta.confirm", ("args",), contract="D1"),
                            lambda g, o, w: {"tool": o["tool"], "args": o["args"], "autonomy": g.autonomy,
                                             "approvals": list(g.slots.get("approvals", ()))}),
        "execute": NodePlan(_node("execute", "tool.execute", ("confirm",), contract="D1", caps=effect),
                            lambda g, o, w: {"tool": o["tool"], "args": o["args"], "approved": o["confirm"]}),
    }
    guards = (Guard("args_complete", "args", "args_complete", "abstain"),
              Guard("approved", "confirm", "approved", "abstain"))
    return _assemble(goal, nodes, guards, {}, ("tool", "args", "execute"), effect)


def _plan_compare(goal: Goal, world: World) -> GoalPlan:
    def build(g, o, w):
        cust = w.customer(g.slots["customer"])
        return {"a": g.slots["a"], "b": g.slots["b"], "customer_name": cust["name"], "order_id": g.slots["order"],
                "eligible": None}
    nodes = {"verdict": NodePlan(_node("verdict", "evaluate.pairwise", bar=0.9, cap=256), build)}
    return _assemble(goal, nodes, (), {}, ("verdict",), frozenset({"read"}))


def _plan_write(goal: Goal, world: World) -> GoalPlan:
    def reply_payload(g, o, w):
        cust = w.customer(g.slots["customer"])
        order = o["order"]
        return {"customer_name": cust["name"], "order_id": g.slots["order"], "product": order["product"],
                "reason": g.slots["reason"], "decision": o["decision"]["decision"], "rule": o["decision"]["rule"],
                "explanation": o["decision"]["explanation"]}
    nodes = {
        "order": NodePlan(_node("order", "tool.get_order", contract="D1"),
                          lambda g, o, w: {"order_id": g.slots["order"]}),
        "decision": NodePlan(_node("decision", "decide.refund", ("order",), contract="D1"),
                             lambda g, o, w: {"order": o["order"], "reason": g.slots["reason"], "now": g.context.now}),
        "reply": NodePlan(_node("reply", "create.reply", ("decision",), bar=0.8, cap=160), reply_payload),
        # Variant: the specialist spoke writes the reply when no hub candidate passes the verifier.
        "reply_spoke": NodePlan(_node("reply_spoke", "create.reply_fallback", bar=0.5, cap=160), reply_payload),
    }
    guards = (Guard("order_found", "order", "order_found", "abstain"),
              Guard("reply_verified", "reply", "reply_verified", "spoke_reply"),
              Guard("reply_verified_2", "reply_spoke", "reply_verified", "abstain"))
    variants = {"spoke_reply": ("reply_spoke",)}
    return _assemble(goal, nodes, guards, variants, ("decision", "reply"), frozenset({"read"}),
                     replaces={"reply_spoke": "reply"})


def _plan_analytics(goal: Goal, world: World) -> GoalPlan:
    nodes = {
        "rates": NodePlan(_node("rates", "analyze.refund_rates", contract="D1"),
                          lambda g, o, w: {"days": g.slots["days"], "now": g.context.now}),
        "summary": NodePlan(_node("summary", "create.summary", ("rates",), bar=0.8, cap=200),
                            lambda g, o, w: {"facts": o["rates"]}),
        "summary_template": NodePlan(_node("summary_template", "create.summary_template", contract="D1"),
                                     lambda g, o, w: {"facts": o["rates"]}),
    }
    guards = (Guard("summary_verified", "summary", "summary_verified", "template"),)
    return _assemble(goal, nodes, guards, {"template": ("summary_template",)}, ("rates", "summary"),
                     frozenset({"read"}), replaces={"summary_template": "summary"})


def _plan_estimate(goal: Goal, world: World) -> GoalPlan:
    nodes = {"estimate": NodePlan(_node("estimate", "analyze.estimate", bar=0.9, cap=96),
                                  lambda g, o, w: {"text": g.slots["question"]})}
    return _assemble(goal, nodes, (), {}, ("estimate",), frozenset({"read"}))


def _plan_expenses(goal: Goal, world: World) -> GoalPlan:
    rows = expense_rows(goal, world)
    nodes: dict[str, NodePlan] = {}
    ids = []
    for i, row in enumerate(rows):
        nid = f"x{i:03d}"
        ids.append(nid)
        nodes[nid] = NodePlan(_node(nid, "decide.expense_review", bar=0.9, cap=64, group="expense"),
                              lambda g, o, w, i=i: {"row": expense_rows(g, w)[i]})
    nodes["flagged"] = NodePlan(_node("flagged", "analyze.collect", tuple(ids), contract="D1"),
                                lambda g, o, w, ids=tuple(ids): {"rows": [o[n] for n in ids]})
    return _assemble(goal, nodes, (), {}, ("flagged",), frozenset({"read"}), shape=len(rows))


def _plan_forecast(goal: Goal, world: World) -> GoalPlan:
    nodes = {"forecast": NodePlan(_node("forecast", "forecast.orders", contract="D1"),
                                  lambda g, o, w: {"days": g.slots["days"], "now": g.context.now})}
    return _assemble(goal, nodes, (), {}, ("forecast",), frozenset({"read"}))


def _plan_policy(goal: Goal, world: World) -> GoalPlan:
    nodes = {
        "section": NodePlan(_node("section", "retrieve.policy", bar=0.85, cap=48),
                            lambda g, o, w: {"query": g.slots["topic"]}),
        "answer": NodePlan(_node("answer", "create.policy_answer", ("section",), contract="D1"),
                           lambda g, o, w: {"section": o["section"], "query": g.slots["topic"]}),
    }
    return _assemble(goal, nodes, (), {}, ("section", "answer"), frozenset({"read"}))


TEMPLATES: dict[str, Callable[[Goal, World], GoalPlan]] = {
    "refund_decision": _plan_refund, "support_request": _plan_support, "compare_replies": _plan_compare,
    "write_reply": _plan_write, "refund_analytics": _plan_analytics, "cost_estimate": _plan_estimate,
    "expense_review": _plan_expenses, "forecast_orders": _plan_forecast, "policy_question": _plan_policy,
}


def _assemble(goal: Goal, nodes: dict[str, NodePlan], guards: tuple[Guard, ...], variants: dict[str, tuple[str, ...]],
              result: tuple[str, ...], ceiling: frozenset[str], *, replaces: dict[str, str] | None = None,
              shape: int | None = None) -> GoalPlan:
    in_variant = {nid for v in variants.values() for nid in v}
    main = tuple(p.node for nid, p in nodes.items() if nid not in in_variant)
    program = Program(
        main=main, variants={vid: tuple(nodes[n].node for n in ns) for vid, ns in variants.items()},
        guards=guards, capability_ceiling=goal.ceiling,
    )
    template = {"version": TEMPLATE_VERSION, "family": goal.family, "shape": shape, "autonomy": goal.autonomy,
                "nodes": [[n.id, n.op, list(n.deps), n.required_accuracy, n.input_cap, sorted(n.capabilities)]
                          for n in (p.node for p in nodes.values())],
                "guards": [g.__dict__ for g in guards], "variants": {k: list(v) for k, v in variants.items()}}
    return GoalPlan(family=goal.family, nodes=nodes, program=program, result=result,
                    template_hash=content_hash(template), notes={"replaces": replaces or {}, "needs": sorted(ceiling)})


class Planner:
    """Template planner with a plan cache keyed by template hash."""

    def __init__(self) -> None:
        self.cache: dict[str, GoalPlan] = {}
        self.hits = 0

    def plan(self, goal: Goal, world: World) -> GoalPlan:
        fresh = TEMPLATES[goal.family](goal, world)
        cached = self.cache.get(fresh.template_hash)
        if cached is not None:
            self.hits += 1
            return cached
        self.cache[fresh.template_hash] = fresh
        return fresh

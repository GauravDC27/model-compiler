"""Compiled-code executors: exact where they apply, zero LLM tokens, D1 by construction.

Each executor is a set of pure functions over typed payloads and the snapshotted world. Rule-
based classifiers (``code.keywords``, ``code.arith``) cover the canonical phrasings they were
written for and abstain elsewhere, so the scheduler escalates instead of trusting a guess.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from datetime import date, datetime, timedelta

from scipy.stats import norm

from dmc.canonical import content_hash

from ..domain import FIRST, POLICY, TOOL_CAPABILITY, World
from .base import ExecResult, NotSupported

Handler = Callable[[dict, bool], ExecResult]


class CodeExecutor:
    def __init__(self, name: str, handlers: dict[str, Handler], version: str = "1"):
        self.name = name
        self.handlers = handlers
        self.version = f"{name}/{version}"

    def run(self, op: str, payload: dict, *, forced: bool) -> ExecResult:
        if op not in self.handlers:
            raise NotSupported(f"{self.name} cannot run {op}")
        return self.handlers[op](payload, forced)


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


# --------------------------------------------------------------------------- lookups and policy

def get_order(world: World) -> Handler:
    def run(p: dict, forced: bool) -> ExecResult:
        order = world.order(p["order_id"])
        if order is not None:
            order = {k: (v.item() if hasattr(v, "item") else v) for k, v in order.items()}
        return ExecResult(True, order, 1.0)
    return run


def refund_rule(order: dict, reason: str, now: str) -> dict:
    """The refund policy (sections P1–P4) as code."""
    delivered = order["status"] == "delivered"
    since_delivery = _days(order["delivered_date"], now) if delivered else None
    since_order = _days(order["order_date"], now)
    if reason == "not_received":
        if delivered:
            return {"decision": "deny", "rule": "P3", "explanation": "Tracking shows the order was delivered; "
                    "open a carrier investigation before refunding."}
        if since_order >= 10:
            return {"decision": "approve", "rule": "P3", "explanation": f"Not delivered {since_order} days after ordering."}
        return {"decision": "deny", "rule": "P3", "explanation": f"Only {since_order} days since ordering; ask the customer to wait."}
    if not delivered:
        return {"decision": "deny", "rule": "P3", "explanation": "The order has not been delivered yet."}
    if reason == "damaged":
        ok = since_delivery <= 90
        return {"decision": "approve" if ok else "deny", "rule": "P2",
                "explanation": f"Damaged item reported {since_delivery} days after delivery (limit 90)."}
    if reason == "wrong_item":
        ok = since_delivery <= 60
        return {"decision": "approve" if ok else "deny", "rule": "P4",
                "explanation": f"Wrong item reported {since_delivery} days after delivery (limit 60)."}
    if order["category"] == "grocery":
        return {"decision": "deny", "rule": "P1", "explanation": "Grocery items cannot be returned for a change of mind."}
    ok = since_delivery <= 30
    return {"decision": "approve" if ok else "deny", "rule": "P1",
            "explanation": f"Change of mind {since_delivery} days after delivery (limit 30)."}


def decide_refund(p: dict, forced: bool) -> ExecResult:
    return ExecResult(True, refund_rule(p["order"], p["reason"], p["now"]), 1.0)


# --------------------------------------------------------------------------- tool arguments, confirmation, effects

REQUIRED_ARGS = {
    "get_order_status": ("order_id",), "cancel_order": ("order_id",), "open_refund_case": ("order_id",),
    "schedule_callback": ("customer_id", "date", "time"), "update_address": ("customer_id", "address"),
    "search_policy": ("query",),
}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _resolve_day(text: str, now: str) -> str | None:
    t = text.lower()
    today = datetime.fromisoformat(now).date()
    if "tomorrow" in t:
        return (today + timedelta(days=1)).isoformat()
    if "today" in t:
        return today.isoformat()
    for i, name in enumerate(WEEKDAYS):
        if name in t:
            ahead = (i - today.weekday()) % 7 or 7  # "on Friday": the next Friday after today
            if f"next {name}" in t:  # "next Monday": the one in the following week
                ahead += 7 if ahead < 7 else 0
            return (today + timedelta(days=ahead)).isoformat()
    return None


def _resolve_time(text: str) -> str | None:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text.lower())
    if not m:
        return None
    h, mins = int(m[1]) % 12 + (12 if m[3] == "pm" else 0), int(m[2] or 0)
    return f"{h:02d}:{mins:02d}"


def tool_args(p: dict, forced: bool) -> ExecResult:
    tool, text = p["tool"], p["text"]
    args: dict = {}
    if m := re.search(r"\bA-\d{4}\b", text):
        args["order_id"] = m[0]
    if m := re.search(r"\b\d+ [A-Z][a-z]+ (?:Street|Road|Avenue|Lane|Court)\b", text):
        args["address"] = m[0]
    if tool in ("schedule_callback", "update_address"):
        args["customer_id"] = p["customer"]
    if tool == "schedule_callback" and p.get("now"):
        if d := _resolve_day(text, p["now"]):
            args["date"] = d
        if t := _resolve_time(text):
            args["time"] = t
    if tool == "search_policy":
        args["query"] = text
    missing = [a for a in REQUIRED_ARGS[tool] if a not in args]
    invalid = []
    if tool == "schedule_callback" and not missing:  # P6: weekdays, 9am–6pm
        dt = date.fromisoformat(args["date"])
        if dt.weekday() >= 5 or not "09:00" <= args["time"] <= "18:00":
            invalid.append("callbacks are weekdays between 9am and 6pm")
    args = {a: args[a] for a in REQUIRED_ARGS[tool] if a in args}
    return ExecResult(True, {"tool": tool, "args": args, "missing": missing, "invalid": invalid}, 1.0)


def confirm(p: dict, forced: bool) -> ExecResult:
    """Confirmation policy: reads and reversible writes pass; destructive actions need an explicit
    approval token supplied with the goal (a human in the loop). Refusal stops the effect."""
    cap = TOOL_CAPABILITY[p["tool"]]
    token = f"{p['tool']}:{p['args']['args'].get('order_id', '')}"
    if cap == "read":
        ok, why = True, "read-only"
    elif cap == "write":
        ok, why = p["autonomy"] in ("confirm", "auto"), "reversible write under confirm autonomy"
    else:
        ok = token in p["approvals"]
        why = "approved by a human" if ok else f"needs explicit approval '{token}'"
    return ExecResult(True, {"approved": ok, "why": why, "capability": cap}, 1.0)


class EffectLog:
    """Recorded side effects with idempotency keys (R22): replaying a goal never re-applies them."""

    def __init__(self) -> None:
        self.effects: dict[str, dict] = {}

    def apply(self, tool: str, args: dict, result: dict) -> dict:
        key = content_hash({"tool": tool, "args": args})
        if key not in self.effects:
            self.effects[key] = {"tool": tool, "args": args, "result": result}
        return {**self.effects[key]["result"], "idempotency_key": key}


def execute_tool(world: World, log: EffectLog) -> Handler:
    def run(p: dict, forced: bool) -> ExecResult:
        tool, args = p["tool"], p["args"]["args"]
        if tool == "get_order_status":
            o = world.order(args["order_id"])
            res = {"status": o["status"], "order_date": o["order_date"], "delivered_date": o["delivered_date"]}
            return ExecResult(True, res, 1.0)
        if tool == "search_policy":
            return ExecResult(True, {"sections": [s.id for s in POLICY]}, 1.0)
        if tool == "cancel_order":
            o = world.order(args["order_id"])
            ok = o["status"] in ("processing",)
            res = {"cancelled": ok, "reason": "cancelled before shipping" if ok else f"cannot cancel: order is {o['status']} (P8)"}
        elif tool == "schedule_callback":
            res = {"scheduled": f"{args['date']} {args['time']}", "customer_id": args["customer_id"]}
        elif tool == "update_address":
            res = {"address": args["address"], "customer_id": args["customer_id"]}
        else:  # open_refund_case
            res = {"case": "R-" + content_hash(args)[7:13], "order_id": args["order_id"]}
        return ExecResult(True, log.apply(tool, args, res), 1.0)
    return run


# --------------------------------------------------------------------------- verifiers (contrastive and creative)

# Offers no refund decision includes: a reply that makes one promises something the policy did not grant.
PROMISES = ("twice the amount", "free gift", "refunded you twice", "guarantee", "lifetime", "replace", "store credit",
            "credit note", "a credit", "complimentary", "voucher", "discount")
NEXT_STEP = ("opened a case", "you will hear from us", "within 24 hours", "next step", "we will", "i will", "we'll",
             "i'll", "follow up", "keep you updated", "let us know", "contact us")


def reply_checks(text: str, customer_name: str, order_id: str, decision: str | None = None) -> dict:
    low = text.lower()
    first = customer_name.split()[0]
    others = [f for f in FIRST if f != first and re.search(rf"\b{f}\b", text)]
    checks = {
        "addresses_customer": bool(re.search(rf"\b{first}\b", text)) and not others,
        "mentions_order": order_id in text,
        "no_false_promise": not any(p in low for p in PROMISES),
        "has_next_step": any(s in low for s in NEXT_STEP),
        "length_ok": 12 <= len(text.split()) <= 90,
    }
    if decision == "deny":
        # Affirmative refund statements only: "we are unable to issue a refund" is a correct denial.
        checks["no_refund_promise"] = not re.search(
            r"\b(we have refunded|we've refunded|(has|have) been refunded|we will refund|we'll refund|"
            r"refund (has been|will be) (approved|issued|processed)|your refund (has been|will be|is on))\b", low)
    if decision == "approve":
        checks["states_refund"] = "refund" in low
    return checks


def pairwise_precheck(p: dict, forced: bool) -> ExecResult:
    ca = reply_checks(p["a"], p["customer_name"], p["order_id"])
    cb = reply_checks(p["b"], p["customer_name"], p["order_id"])
    fa, fb = sum(not v for v in ca.values()), sum(not v for v in cb.values())
    detail = {"checks_a": ca, "checks_b": cb}
    if fa != fb and min(fa, fb) == 0:
        return ExecResult(True, "A" if fa < fb else "B", 1.0, detail=detail)
    # Both pass (or both fail) every executable check: a judge must decide.
    return ExecResult(forced, ("A" if fa <= fb else "B") if forced else None, 0.5, detail=detail)


# --------------------------------------------------------------------------- analytics

def refund_rates(world: World) -> Handler:
    def run(p: dict, forced: bool) -> ExecResult:
        since = (datetime.fromisoformat(p["now"]) - timedelta(days=p["days"])).date().isoformat()
        o = world.orders[(world.orders.status == "delivered") & (world.orders.order_date >= since)]
        g = o.groupby("category").refunded.agg(["sum", "count"]).sort_index()
        rates = {c: {"refunds": int(r["sum"]), "orders": int(r["count"]), "rate": round(float(r["sum"] / r["count"]), 4)}
                 for c, r in g.iterrows()}
        top = max(rates, key=lambda c: (rates[c]["rate"], c))
        rest_n = sum(v["orders"] for c, v in rates.items() if c != top)
        rest_k = sum(v["refunds"] for c, v in rates.items() if c != top)
        k1, n1 = rates[top]["refunds"], rates[top]["orders"]
        pool = (k1 + rest_k) / (n1 + rest_n)
        se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / rest_n))
        z = (k1 / n1 - rest_k / rest_n) / se if se > 0 else 0.0
        pval = float(2 * norm.sf(abs(z)))
        facts = {"days": p["days"], "top_category": top, "top_rate": rates[top]["rate"],
                 "rest_rate": round(rest_k / rest_n, 4), "z": round(z, 2), "p_value": round(pval, 4),
                 "significant": pval < 0.05, "rates": rates}
        return ExecResult(True, facts, 1.0)
    return run


def summary_template(p: dict, forced: bool) -> ExecResult:
    f = p["facts"]
    verdict = "significantly higher than" if f["significant"] else "not significantly different from"
    text = (f"In the last {f['days']} days, {f['top_category']} had the highest refund rate at {100 * f['top_rate']:.1f}%, "
            f"{verdict} the {100 * f['rest_rate']:.1f}% of all other categories (z = {f['z']}, p = {f['p_value']}).")
    return ExecResult(True, text, 1.0)


def summary_checks(text: str, facts: dict) -> dict:
    low = text.lower()
    return {"names_top": facts["top_category"] in low,
            "states_top_rate": f"{100 * facts['top_rate']:.1f}" in text,
            "states_rest_rate": f"{100 * facts['rest_rate']:.1f}" in text,
            "significance_right": ("not significant" in low or "no significant" in low or "not statistically" in low)
            != facts["significant"],
            "length_ok": len(text.split()) <= 80}


def collect(p: dict, forced: bool) -> ExecResult:
    flagged = [r["expense_id"] for r in p["rows"] if r["needs_review"]]
    return ExecResult(True, {"flagged": flagged, "n_reviewed": len(p["rows"])}, 1.0)


def policy_answer(p: dict, forced: bool) -> ExecResult:
    sec = next(s for s in POLICY if s.id == p["section"]["id"])
    return ExecResult(True, f"Policy {sec.id}, {sec.title}: {sec.text}", 1.0)


# --------------------------------------------------------------------------- compiled grammars (System-1 code)

CLAIM_RULES = {
    "damaged": r"\b(arrived damaged|came broken|is defective|smashed|crack|torn|stopped working)\b",
    "not_received": r"(never arrived|not received|still not here|not been delivered|never showed up|still waiting)",
    "wrong_item": r"(wrong (size|colour|one)|sent me the wrong|different .+ this is|not the .+ i ordered|someone else's)",
    "changed_mind": r"(changed my mind|no longer need|do not want|cheaper|by mistake|would like to return it)",
}
TOOL_RULES = {
    "get_order_status": r"\b(status|track|shipped yet|when will order)\b",
    "schedule_callback": r"\b(call me|phone me|callback|call with me|ring me)\b",
    "cancel_order": r"\bcancel\b|\bstop order\b",
    "update_address": r"\b(address|ship to|send my orders to)\b",
    "search_policy": r"\b(policy|how long do i have|rules for|refund grocery)\b",
    "open_refund_case": r"\b(refund for|refund order|refund for order|money back for order|start a refund|open a refund)\b",
}


def keyword_classifier(rules: dict[str, str]) -> Handler:
    compiled = {k: re.compile(v, re.I) for k, v in rules.items()}

    def run(p: dict, forced: bool) -> ExecResult:
        hits = sorted(k for k, rx in compiled.items() if rx.search(p["text"]))
        if len(hits) == 1:
            return ExecResult(True, hits[0], 1.0, detail={"hits": hits})
        if forced:  # no rule or several: the cheapest deterministic fallback
            return ExecResult(True, (hits or sorted(rules))[0], 0.0, detail={"hits": hits})
        return ExecResult(False, None, 0.0, detail={"hits": hits})
    return run


ARITH_RULES = (
    (re.compile(r"^If we refund (\d+)% of (\d+) orders that average \$(\d+), what is the total refund cost\?$"),
     lambda p, n, a: p / 100 * n * a),
    (re.compile(r"^What does it cost to refund (\d+) percent of (\d+) orders at \$(\d+) each\?$"),
     lambda p, n, a: p / 100 * n * a),
)


def arith(p: dict, forced: bool) -> ExecResult:
    for rx, fn in ARITH_RULES:
        if m := rx.match(p["text"]):
            return ExecResult(True, round(fn(*map(float, m.groups())), 2), 1.0, detail={"rule": rx.pattern[:40]})
    return ExecResult(forced, None if not forced else 0.0, 0.0)


# Safe arithmetic evaluator for program-of-thought expressions written by a model.
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Pow: operator.pow}


def safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 6:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"disallowed expression element {type(node).__name__}")
    return ev(ast.parse(expr.replace("%", "/100").replace("$", "").replace(",", ""), mode="eval"))


def build_code_executors(world: World, log: EffectLog) -> dict[str, CodeExecutor]:
    return {e.name: e for e in (
        CodeExecutor("code.lookup", {"tool.get_order": get_order(world)}),
        CodeExecutor("code.policy", {"decide.refund": decide_refund}),
        CodeExecutor("code.args", {"tool.args": tool_args}),
        CodeExecutor("code.confirm", {"meta.confirm": confirm}),
        CodeExecutor("code.tools", {"tool.execute": execute_tool(world, log)}),
        CodeExecutor("code.verify", {"evaluate.pairwise": pairwise_precheck}),
        CodeExecutor("code.analytics", {"analyze.refund_rates": refund_rates(world), "create.summary_template": summary_template,
                                        "analyze.collect": collect, "create.policy_answer": policy_answer}),
        CodeExecutor("code.arith", {"analyze.estimate": arith}),
        CodeExecutor("code.keywords", {"decide.claim_reason": keyword_classifier(CLAIM_RULES),
                                       "tool.select": keyword_classifier(TOOL_RULES)}),
    )}

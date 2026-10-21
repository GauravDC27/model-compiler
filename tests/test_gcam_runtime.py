"""GCAM v1 runtime end to end, with a deterministic stub standing in for the spoke model.

The stub keeps the test fast and exact; everything else is the real system: parser, planner,
scheduler with joint calibration records, code executors, guards, variants, confirmation
gating, input caps, receipts and D0 replay.
"""

import pytest

from gcam.domain import build_world
from gcam.executors.base import ExecResult
from gcam.executors.code import summary_template
from gcam.runtime import Runtime
from gcam.system import System, rung_name


class StubSpoke:
    name = "spoke"
    version = "stub/1"
    k = 1

    def __init__(self):
        self.calls = 0

    def template_tokens(self, op, sample):
        return 50

    def run(self, op, payload, *, forced):
        self.calls += 1
        use = {"spoke": 40}
        if op == "decide.claim_reason":
            t = payload["text"].lower()
            label = "not_received" if "never" in t or "nowhere" in t else "wrong_item" if "wrong" in t else "damaged"
            return ExecResult(True, label, 0.9, use)
        if op == "tool.select":
            return ExecResult(True, "cancel_order" if "cancel" in payload["text"].lower() else "get_order_status", 0.9, use)
        if op == "create.summary":
            return ExecResult(True, "Refunds went up.", 0.5, use)  # fails the fact checks on purpose
        if op in ("create.reply", "create.reply_fallback"):
            first = payload["customer_name"].split()[0]
            if op == "create.reply":  # the main rung writes a reply that misses the order number
                return ExecResult(forced, f"Hi {first}, sorry about that. We will look into it within 24 hours.", 0.0, use)
            text = (f"Hi {first}, thank you for telling us about order {payload['order_id']}. "
                    f"We will refund you and you will hear from us within 24 hours.")
            if payload["decision"] == "deny":
                text = (f"Hi {first}, thank you for telling us about order {payload['order_id']}. Under our policy we "
                        f"cannot return the money this time, but you will hear from us within 24 hours with options.")
            return ExecResult(True, text, 1.0, use)
        raise KeyError(op)


def _records():
    rows = []
    for i in range(20):
        rows.append({rung_name("code.keywords", "decide.claim_reason"): {"answered": i < 12, "correct": i < 12,
                                                                        "correct_forced": i < 14, "tokens": 0},
                     rung_name("spoke", "decide.claim_reason"): {"answered": True, "correct": i != 19,
                                                                "correct_forced": i != 19, "tokens": 120}})
    tools = [{rung_name("code.keywords", "tool.select"): {"answered": i < 10, "correct": i < 10, "correct_forced": i < 12,
                                                          "tokens": 0},
              rung_name("spoke", "tool.select"): {"answered": True, "correct": True, "correct_forced": True, "tokens": 150}}
             for i in range(20)]
    one = lambda name, ok: [{name: {"answered": ok, "correct": ok, "correct_forced": ok, "tokens": 300}}] * 10  # noqa: E731
    return {"decide.claim_reason": rows, "tool.select": tools,
            "create.reply": one(rung_name("hub", "create.reply"), True),
            "create.reply_fallback": one(rung_name("spoke", "create.reply_fallback"), True),
            "create.summary": one(rung_name("spoke", "create.summary"), True)}


@pytest.fixture
def rt(tmp_path):
    system = System(llms=False, art=tmp_path, world=build_world())
    system.impls["spoke"] = StubSpoke()
    system.impls["hub"] = StubSpoke()  # stands in for the hub's best-of-K writer (create.reply)
    system.calibration = {"records": _records(), "thresholds": {}}
    system.rungs = {n: r for n, r in system._register().items()
                    if not n.startswith("hub[") or n == rung_name("hub", "create.reply")}
    return Runtime(system, tmp_path / "store")


def _delivered(world, k=0):
    row = world.orders[world.orders.status == "delivered"].iloc[k]
    return row.customer_id, row.order_id


def test_refund_goal_runs_the_compiled_ladder_and_replays(rt):
    cust, order = _delivered(rt.system.world)
    text = f'Customer {cust} requests a refund for order {order}: "The lamp arrived damaged."'
    first = rt.run(text)
    assert first.status == "done", first.reason
    reason = next(n for n in first.nodes if n.node == "reason")
    assert reason.ladder == (rung_name("code.keywords", "decide.claim_reason"), rung_name("spoke", "decide.claim_reason"))
    assert reason.executor.startswith("code.keywords") and reason.cost == 0  # System-1 code answered: no tokens
    assert first.result["decision"]["rule"] == "P2"
    assert first.bound_respected
    again = rt.run(text)
    assert all(n.replayed for n in again.nodes) and again.cost == 0  # D0 replay from receipts
    assert again.result == first.result and again.schedule_hash == first.schedule_hash


def test_escalation_happens_only_on_abstention(rt):
    cust, order = _delivered(rt.system.world, 1)
    res = rt.run(f'Customer {cust} requests a refund for order {order}: "Two weeks and counting, it is nowhere to be seen."')
    reason = next(n for n in res.nodes if n.node == "reason")
    assert [t["rung"] for t in reason.tried] == list(reason.ladder) and reason.executor.startswith("spoke")
    assert res.result["reason"] == "not_received" and res.cost == reason.cost > 0


def test_destructive_action_is_gated_by_confirmation(rt):
    world = rt.system.world
    row = world.orders[world.orders.status == "processing"].iloc[0]
    text = f'Handle request from {row.customer_id}: "Cancel order {row.order_id}."'
    refused = rt.run(text)
    assert refused.status == "abstained" and "approved" in refused.reason
    assert not rt.system.effects.effects  # nothing was cancelled
    done = rt.run(text, approvals=(f"cancel_order:{row.order_id}",))
    assert done.status == "done" and done.result["execute"]["cancelled"]
    rt.run(text, approvals=(f"cancel_order:{row.order_id}",))
    assert len(rt.system.effects.effects) == 1  # idempotent effects


def test_guard_failure_runs_the_precompiled_variant(rt):
    cust, order = _delivered(rt.system.world, 2)
    res = rt.run(f"Write a reply to {cust} about order {order} that arrived damaged")
    assert res.status == "done"
    assert any("reply_verified failed" in e and "spoke_reply" in e for e in res.events)
    assert order in res.result["reply"]  # the variant's verified reply replaced the failed one
    assert res.bound_respected


def test_failed_llm_summary_falls_back_to_the_code_template(rt):
    res = rt.run("Which category had the highest refund rate in the last 90 days?")
    assert res.status == "done"
    facts = res.result["rates"]
    assert res.result["summary"] == summary_template({"facts": facts}, True).value


def test_unsupported_goal_and_read_only_autonomy_are_defined_failures(rt):
    assert rt.run("Make us rich").status == "clarify"
    cust, _ = _delivered(rt.system.world)
    out = rt.run(f'Handle request from {cust}: "Where is my order?"', autonomy="read_only")
    assert out.status == "compile_error" and "ceiling" in out.reason

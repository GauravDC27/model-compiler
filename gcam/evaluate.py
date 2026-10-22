"""Evaluate GCAM v1: per-op ladders vs single-model baselines, predictability, and end-to-end goals.

1. **Per op, on the held-out test records** (``artifacts/gcam/test_records.json``): the ladder the
   scheduler compiles (from calibration records only) is replayed item by item on test items it
   never saw. It is compared with every single executor forced to answer alone, in particular
   "always spoke" and "always hub". Reported: accuracy, mean cost (hub-token equivalents), share
   answered by each rung, and predicted (calibration) vs realized (test) accuracy and cost.
2. **Ablations**: the same per-op compilation with only LLM rungs (no code, classic ML or System-1),
   and with only the base hub and the spoke (no goal-family layers either).
3. **Specialized MoE layers**: the capacity manifest (base vs goal-family capacity, wrong-family
   accuracy, goal-off identity), and refund and support goals run end to end with LLM rungs only,
   where the compiler must choose among the base hub, the hub with its family layers, and the spoke.
4. **End-to-end goals** through the runtime, twice: the second pass must replay every node from
   receipts (D0) and return identical results.

    python -m gcam.evaluate
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

from dmc.scheduler import CompileError, Node, compile_schedule

from .domain import build_world, claims, estimates, reply_pairs, tool_requests
from .planner import BARS as PLAN_BARS
from .runtime import Runtime
from .system import ALIAS, System

ROOT = Path(__file__).resolve().parents[1]
BARS = {op: v for op, v in PLAN_BARS.items() if op != "create.reply_fallback"}  # the plans' bars (gcam/planner.py)
LADDER = set(ALIAS.values())  # ladder rungs are named impl[alias]; exact rungs (code, forecaster) always stay


def replay(ladder: tuple[str, ...], items: list[dict]) -> dict:
    correct, cost, answered_by = 0, 0.0, {n: 0 for n in ladder}
    for rec in items:
        for i, name in enumerate(ladder):
            o = rec[name]
            cost += o["tokens"]
            if i == len(ladder) - 1:
                correct += o["correct_forced"]
                answered_by[name] += 1
                break
            if o["answered"]:
                correct += o["correct"]
                answered_by[name] += 1
                break
    n = len(items)
    return {"accuracy": round(correct / n, 4), "cost": round(cost / n, 1),
            "answered_by": {k: round(v / n, 3) for k, v in answered_by.items()}}


# Executor sets for the ablation: which rungs the compiler may use.
ABLATIONS = {
    "llm_only": lambda name: name.startswith(("hub", "spoke")),  # no code, classic ML or System-1 rungs
    "llm_only_no_goal_capacity": lambda name: name.startswith(("hub[", "spoke[")),  # and no goal-family hub layers
}
ABLATION_OPS = ("decide.claim_reason", "tool.select", "evaluate.pairwise", "analyze.estimate", "decide.expense_review",
                "retrieve.policy")


@contextmanager
def only_rungs(system: System, keep):
    full = system.rungs
    system.rungs = {n: r for n, r in full.items() if keep(n) or n.split("[")[1].rstrip("]") not in LADDER}
    try:
        yield
    finally:
        system.rungs = full


def per_op(system: System, lock_hash: str, ops=None) -> dict:
    test = json.loads((system.art / "test_records.json").read_text())["records"]
    out = {}
    for op, (bar, cap) in BARS.items():
        if ops is not None and op not in ops:
            continue
        node = Node("n", op, required_accuracy=bar, input_cap=cap, requires_contract="D1")
        try:
            sch = compile_schedule([node], system.executors, token_budget=10**9, calibration=system.outcomes(),
                                   lock_hash=lock_hash)
            a = sch.assignments[0]
            ladder, predicted = a.ladder, {"accuracy": a.expected_accuracy, "cost": a.expected_tokens}
        except CompileError as e:
            out[op] = {"compile_error": str(e)}
            continue
        items = test[op]
        realized = replay(ladder, items)
        singles = {n: {"accuracy": round(sum(r[n]["correct_forced"] for r in items) / len(items), 4),
                       "cost": round(sum(r[n]["tokens"] for r in items) / len(items), 1)} for n in items[0]}
        spoke = next((v for k, v in singles.items() if k.startswith("spoke[")), None)
        out[op] = {"bar": bar, "n_test": len(items), "ladder": list(ladder), "predicted": predicted,
                   "realized": realized, "single_executor": singles,
                   "vs_always_spoke": None if spoke is None else {
                       "accuracy_delta": round(realized["accuracy"] - spoke["accuracy"], 4),
                       "cost_ratio": round(realized["cost"] / spoke["cost"], 3) if spoke["cost"] else None}}
    return out


def demo_goals(world) -> list[tuple[str, dict]]:
    d = world.orders[world.orders.status == "delivered"].reset_index(drop=True)
    proc = world.orders[world.orders.status == "processing"].reset_index(drop=True)
    c_train = claims(4, split="train", seed=77)
    c_test = claims(4, split="test", seed=77)
    oids = list(world.orders.order_id[:400])
    t_test = tool_requests(6, split="test", seed=77, order_ids=oids)
    pairs = reply_pairs(5, seed=77, world=world)
    e_train, e_test = estimates(2, split="train", seed=77), estimates(2, split="test", seed=77)
    row = lambda i: (d.customer_id[i], d.order_id[i])  # noqa: E731
    goals = [
        (f'Customer {row(10)[0]} requests a refund for order {row(10)[1]}: "{c_train[0].text}"', {}),
        (f'Customer {row(11)[0]} requests a refund for order {row(11)[1]}: "{c_test[1].text}"', {}),
        (f'Customer {row(12)[0]} requests a refund for order {row(12)[1]}: "{c_test[2].text}"', {}),
        (f'Handle request from {row(13)[0]}: "Can someone phone me tomorrow at 3pm?"', {}),
        (f'Handle request from {row(14)[0]}: "{t_test[0].text}"', {}),
        (f'Handle request from {proc.customer_id[0]}: "Cancel order {proc.order_id[0]}."', {}),
        (f'Handle request from {proc.customer_id[0]}: "Cancel order {proc.order_id[0]}."',
         {"approvals": (f"cancel_order:{proc.order_id[0]}",)}),
        (f'Which reply to {pairs[0].customer} about order {pairs[0].order_id} is better? '
         f'A: "{pairs[0].reply_a}" B: "{pairs[0].reply_b}"', {}),
        (f'Which reply to {pairs[2].customer} about order {pairs[2].order_id} is better? '
         f'A: "{pairs[2].reply_a}" B: "{pairs[2].reply_b}"', {}),
        (f"Write a reply to {row(15)[0]} about order {row(15)[1]} that arrived damaged", {}),
        ("Which category had the highest refund rate in the last 90 days?", {}),
        (f"Estimate: {e_train[0].text}", {}),
        (f"Estimate: {e_test[1].text}", {}),
        ("Review expense reports from the last 7 days", {}),
        ("Forecast daily orders for the next 7 days", {}),
        ("What does our policy say about damaged items?", {}),
        ("What does our policy say about a parcel that is two weeks late?", {}),
        (f'Customer {row(10)[0]} requests a refund: "It broke."', {}),
    ]
    return goals


def summarize_goal(res) -> dict:
    return {"goal": res.text[:160], "family": res.family, "status": res.status, "reason": res.reason,
            "events": res.events, "cost": res.cost, "worst_case": res.worst_case, "bound_respected": res.bound_respected,
            "seconds": round(res.seconds, 1), "schedule_hash": res.schedule_hash,
            "nodes": [{"node": n.node, "op": n.op, "ladder": list(n.ladder), "executor": n.executor,
                       "escalations": max(0, len(n.tried) - 1), "cost": n.cost, "replayed": n.replayed,
                       "cap_hits": n.cap_hits} for n in res.nodes],
            "result": res.result}


def main() -> None:
    t0 = time.time()
    world = build_world()
    system = System(llms=True, world=world)
    rt = Runtime(system)
    report = {"lock_hash": rt.lock_hash, "rungs": sorted(system.rungs)}
    report["per_op"] = per_op(system, rt.lock_hash)
    print("per-op done", f"{time.time() - t0:.0f}s", flush=True)
    report["ablations"] = {}
    for label, keep in ABLATIONS.items():
        with only_rungs(system, keep):
            report["ablations"][label] = per_op(system, rt.lock_hash, ABLATION_OPS)
    print("ablations done", f"{time.time() - t0:.0f}s", flush=True)
    report["capacity"] = json.loads((system.art / "capacity" / "manifest.json").read_text())
    for fam in report["capacity"]["families"].values():
        fam.pop("train_log", None)
        fam.pop("parents", None)
        fam.pop("hot", None)
    goals = demo_goals(world)
    first, second, raw = [], [], []
    for text, kw in goals:
        raw.append(rt.run(text, **kw))
        first.append(summarize_goal(raw[-1]))
        print(first[-1]["status"], first[-1]["family"], first[-1]["cost"], f"{time.time() - t0:.0f}s", flush=True)
    for text, kw in goals:
        second.append(summarize_goal(rt.run(text, **kw)))
    report["goals"] = first
    report["replay"] = {
        "all_nodes_replayed": all(n["replayed"] for g in second for n in g["nodes"]),
        "identical_results": all(a["result"] == b["result"] and a["status"] == b["status"] for a, b in zip(first, second)),
        "replay_cost": sum(g["cost"] for g in second),
        "effects_applied": len(system.effects.effects),
    }
    exp = next(r for r in raw if r.family == "expense_review")
    truth = world.expenses.set_index("expense_id").needs_review
    rows = [n for n in exp.nodes if n.op == "decide.expense_review"]
    report["expense_goal"] = {
        "rows": len(rows), "flagged": len(exp.result["flagged"]["flagged"]),
        "accuracy": round(sum(n.value["needs_review"] == bool(truth[n.value["expense_id"]]) for n in rows) / len(rows), 4),
        "answered_by": {e: sum(n.executor == e for n in rows) for e in sorted({n.executor for n in rows})},
    }
    # The hub's goal-family layers in end-to-end goals: the same refund and support goals compiled
    # without code and System-1 rungs, so the compiler must choose among hub, hub + family, spoke.
    with only_rungs(system, ABLATIONS["llm_only"]):
        rt_llm = Runtime(system, system.art / "store_llm_only")
        llm_goals = [g for g in goals if g[0].startswith(("Customer", "Handle request")) and "Cancel" not in g[0]][:4]
        report["llm_only_goals"] = [summarize_goal(rt_llm.run(text, **kw)) for text, kw in llm_goals]
    report["elapsed_s"] = round(time.time() - t0, 1)
    out = ROOT / "experiments" / "results" / "gcam_v1_eval.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["replay"], indent=1), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

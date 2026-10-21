"""Fit the System-1 and classic models, set every abstention threshold, and write the joint
calibration records the scheduler prices ladders with (plan §I.2.2, stage T6 on CPU).

Splits, all disjoint and generated from different seeds:

* **fit** — training phrasings only (claims, tool requests) or the oldest expense rows: trains the
  encoder heads, the hub-state probe and the GBDT;
* **threshold** — a mixture of training and held-out phrasings: sets each executor's conformal
  threshold (answered accuracy ≥ 0.95);
* **calibration** — the same mixture, fresh items: every capable executor runs on every item and
  its ``Outcome`` (answered at threshold, correct, correct when forced, cost) is recorded;
* **test** — the same mixture again, fresh items: the same records, kept apart for evaluation.

Expense rows are split by date so that every expense-review goal (last 7–30 days) runs on rows
the GBDT never saw.

    python -m gcam.train.calibrate
"""

from __future__ import annotations

import json
import os
import pickle
import time
from datetime import timedelta

import numpy as np

from ..domain import (NOW, NOW_DT, CLAIM_REASONS, TOOLS, build_world, claims, estimates, policy_queries, reply_pairs,
                      tool_requests)
from ..executors.classic import ExpenseGBDT, PolicyTfidf, conformal_threshold
from ..executors.code import refund_rule, reply_checks, summary_checks
from ..executors.system1 import Encoder, EncoderHeads, HubProbe, LinearHead
from ..models import artifacts_dir
from ..system import CANDIDATES, System, rung_name

SMOKE = bool(os.environ.get("GCAM_SMOKE"))
TARGET = 0.95
N = {"fit": 480, "thr": 60, "cal": 120, "test": 120, "pair_thr": 30, "pair": 60, "reply": 12, "est": 40,
     "policy": 40, "exp_cal": 120}
if SMOKE:
    N = {k: max(4, v // 15) for k, v in N.items()}
PROBE_LAYER = 12


def _rows(df) -> list[dict]:
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in df.to_dict("records")]


def mix(fn, n: int, seed: int, **kw) -> list:
    return fn(n // 2, split="train", seed=seed, **kw) + fn(n - n // 2, split="test", seed=seed, **kw)


def build_items(world) -> dict[str, dict[str, list[tuple[dict, object]]]]:
    """op -> split -> [(payload, truth)]."""
    oids = list(world.orders.order_id[:400])
    items: dict[str, dict] = {}
    for split, seed, n in (("thr", 21, N["thr"]), ("cal", 22, N["cal"]), ("test", 23, N["test"])):
        items.setdefault("decide.claim_reason", {})[split] = [({"text": c.text}, c.reason) for c in mix(claims, n, seed)]
        items.setdefault("tool.select", {})[split] = [({"text": r.text}, r.tool)
                                                      for r in mix(tool_requests, n, seed, order_ids=oids)]
    for split, seed, n in (("thr", 31, N["pair_thr"]), ("cal", 32, N["pair"]), ("test", 33, N["pair"])):
        items.setdefault("evaluate.pairwise", {})[split] = [
            ({"a": p.reply_a, "b": p.reply_b, "customer_name": world.customer(p.customer)["name"], "order_id": p.order_id,
              "eligible": None}, p.better) for p in reply_pairs(n, seed=seed, world=world)]
    delivered = world.orders[world.orders.status == "delivered"].reset_index(drop=True)
    rng = np.random.default_rng(41)
    for split, n in (("cal", N["reply"]), ("test", N["reply"])):
        rows = []
        for i in rng.choice(len(delivered), size=n, replace=False):
            o = _rows(delivered.iloc[[int(i)]])[0]
            reason = ("damaged", "not_received", "wrong_item")[len(rows) % 3]
            d = refund_rule(o, reason, NOW)
            rows.append(({"customer_name": world.customer(o["customer_id"])["name"], "order_id": o["order_id"],
                          "product": o["product"], "reason": reason, "decision": d["decision"], "rule": d["rule"]}, True))
        items.setdefault("create.reply", {})[split] = rows
        items.setdefault("create.reply_fallback", {})[split] = rows
    code = System(llms=False, world=world).impls["code.analytics"]
    windows = {"cal": range(30, 181, 10), "test": range(35, 176, 10)}
    for split, days in windows.items():
        days = list(days)[: (3 if SMOKE else None)]
        items.setdefault("create.summary", {})[split] = [
            ({"facts": code.run("analyze.refund_rates", {"days": d, "now": NOW}, forced=True).value}, True) for d in days]
    for split, seed in (("cal", 52), ("test", 53)):
        items.setdefault("analyze.estimate", {})[split] = [({"text": e.text}, e.answer)
                                                           for e in mix(estimates, N["est"], seed)]
    for split, seed, n in (("thr", 61, N["thr"] // 2), ("cal", 62, N["policy"]), ("test", 63, N["policy"])):
        items.setdefault("retrieve.policy", {})[split] = [({"query": q.text}, q.section) for q in mix(policy_queries, n, seed)]
    ex = world.expenses.sort_values("expense_id")
    day = lambda k: (NOW_DT - timedelta(days=k)).date().isoformat()  # noqa: E731
    test_rows = _rows(ex[ex.date >= day(30)])
    cal_rows = _rows(ex[(ex.date < day(30)) & (ex.date >= day(60))])[: N["exp_cal"]]
    thr_rows = _rows(ex[(ex.date < day(60)) & (ex.date >= day(90))])[: N["exp_cal"]]
    items["decide.expense_review"] = {
        "fit": _rows(ex[ex.date < day(90)]),
        "thr": [({"row": r}, r["needs_review"]) for r in thr_rows],
        "cal": [({"row": r}, r["needs_review"]) for r in cal_rows],
        "test": [({"row": r}, r["needs_review"]) for r in test_rows[: (N["exp_cal"] if SMOKE else None)]],
    }
    return items


def correct(op: str, value, truth, payload) -> bool:
    if value is None:
        return False
    if op == "decide.expense_review":
        return value["needs_review"] == truth
    if op == "retrieve.policy":
        return value["id"] == truth
    if op in ("create.reply", "create.reply_fallback"):
        return all(reply_checks(value, payload["customer_name"], payload["order_id"], payload["decision"]).values())
    if op == "create.summary":
        return all(summary_checks(value, payload["facts"]).values())
    if op == "analyze.estimate":
        return abs(float(value) - truth) <= 0.005 * abs(truth) + 0.01
    return value == truth


def fit_system1(world, items) -> dict:
    enc = Encoder()
    heads = {}
    fit_claims = [(c.text, c.reason) for c in claims(N["fit"], split="train", seed=1)]
    oids = list(world.orders.order_id[:400])
    fit_tools = [(r.text, r.tool) for r in tool_requests(N["fit"], split="train", seed=1, order_ids=oids)]
    for op, data, labels in (("decide.claim_reason", fit_claims, CLAIM_REASONS), ("tool.select", fit_tools, TOOLS)):
        X = np.stack([enc.embed(t)[0] for t, _ in data])
        head = LinearHead(tuple(labels)).fit(X, [y for _, y in data])
        thr = items[op]["thr"]
        head.calibrate(np.stack([enc.embed(p["text"])[0] for p, _ in thr]), [y for _, y in thr], TARGET)
        heads[op] = head
    # dense retrieval margin threshold
    s1 = EncoderHeads(enc, heads)
    thr = items["retrieve.policy"]["thr"]
    res = [s1.run("retrieve.policy", p, forced=True) for p, _ in thr]
    retrieval_tau = conformal_threshold([r.confidence for r in res], [r.value["id"] == y for r, (_, y) in zip(res, thr)],
                                        TARGET)
    tfidf = PolicyTfidf()
    res = [tfidf.run("retrieve.policy", p, forced=True) for p, _ in thr]
    tfidf.tau = conformal_threshold([r.confidence for r in res], [r.value["id"] == y for r, (_, y) in zip(res, thr)],
                                    TARGET)
    gbdt = ExpenseGBDT().fit(items["decide.expense_review"]["fit"])
    gbdt.calibrate([p["row"] for p, _ in items["decide.expense_review"]["thr"]], TARGET)
    return {"encoder_heads": heads, "retrieval_tau": retrieval_tau, "tfidf": tfidf, "gbdt": gbdt, "enc": enc}


def fit_probe(hub, items) -> dict[str, LinearHead]:
    probe = HubProbe(hub, PROBE_LAYER, {})
    data = [(c.text, c.reason) for c in claims(N["fit"], split="train", seed=1)]
    X = np.stack([probe.features(t)[0] for t, _ in data])
    head = LinearHead(tuple(CLAIM_REASONS), C=0.5).fit(X, [y for _, y in data])
    thr = items["decide.claim_reason"]["thr"]
    head.calibrate(np.stack([probe.features(p["text"])[0] for p, _ in thr]), [y for _, y in thr], TARGET)
    return {"decide.claim_reason": head}


def main() -> None:
    t0 = time.time()
    art = artifacts_dir()
    world = build_world()
    items = build_items(world)
    s1 = fit_system1(world, items)
    print("system-1 fitted", f"{time.time() - t0:.0f}s", flush=True)

    from ..models import HUB_ID, LM  # noqa: F401  (System loads the hub and spoke)
    # Write the sklearn artifacts first (System reads them), with a placeholder probe.
    saved = {"encoder_heads": s1["encoder_heads"], "retrieval_tau": s1["retrieval_tau"], "tfidf": s1["tfidf"],
             "gbdt": s1["gbdt"], "probe_layer": PROBE_LAYER, "probe_heads": {}}
    with open(art / "system1.pkl", "wb") as f:
        pickle.dump(saved, f)
    system = System(llms=True, art=art, world=world)
    saved["probe_heads"] = fit_probe(system.models["hub"], items)
    with open(art / "system1.pkl", "wb") as f:
        pickle.dump(saved, f)
    system.impls["s1.hubprobe"].heads = saved["probe_heads"]
    print("probe fitted", f"{time.time() - t0:.0f}s", flush=True)

    # LLM thresholds on the threshold split (spoke is always the last rung: no threshold needed).
    thresholds = {}
    for op in ("decide.claim_reason", "tool.select", "evaluate.pairwise", "decide.expense_review", "retrieve.policy"):
        for impl in CANDIDATES[op]:
            if not impl.startswith("hub"):
                continue
            ex = system.impls[impl]
            out = [ex.run(op, p, forced=True) for p, _ in items[op]["thr"]]
            tau = conformal_threshold([o.confidence for o in out],
                                      [correct(op, o.value, y, p) for o, (p, y) in zip(out, items[op]["thr"])], TARGET)
            thresholds[f"{impl}|{op}"] = tau
            ex.tau[op] = tau
        print("thresholds", op, {k: v for k, v in thresholds.items() if k.endswith(op)}, f"{time.time() - t0:.0f}s",
              flush=True)

    records: dict[str, dict[str, list]] = {"cal": {}, "test": {}}
    for op, impls in CANDIDATES.items():
        for split in ("cal", "test"):
            rows = []
            for payload, truth in items[op][split]:
                rec = {}
                for impl in impls:
                    if impl not in system.impls:
                        continue
                    r = system.impls[impl].run(op, payload, forced=False)
                    value, answered, cost = r.value, r.answered, r.cost
                    if not answered and value is None:  # cheap code rungs: ask for the forced answer
                        f = system.impls[impl].run(op, payload, forced=True)
                        value, cost = f.value, cost + f.cost
                    ok = correct(op, value, truth, payload)
                    rec[rung_name(impl, op)] = {"answered": bool(answered), "correct": bool(answered and ok),
                                                "correct_forced": bool(ok), "tokens": int(r.cost)}
                rows.append(rec)
            records[split][op] = rows
            acc = {n: round(sum(r[n]["correct_forced"] for r in rows) / len(rows), 3) for n in rows[0]}
            cov = {n: round(sum(r[n]["answered"] for r in rows) / len(rows), 3) for n in rows[0]}
            print(split, op, "forced acc", acc, "coverage", cov, f"{time.time() - t0:.0f}s", flush=True)
    meta = {"target": TARGET, "sizes": {op: {s: len(v) for s, v in d.items() if s != "fit"} for op, d in items.items()},
            "probe_layer": PROBE_LAYER, "encoder_thresholds": {op: h.tau for op, h in s1["encoder_heads"].items()},
            "retrieval_tau": s1["retrieval_tau"], "tfidf_tau": s1["tfidf"].tau, "gbdt_tau": s1["gbdt"].tau,
            "probe_tau": saved["probe_heads"]["decide.claim_reason"].tau, "elapsed_s": round(time.time() - t0, 1)}
    (art / "calibration.json").write_text(json.dumps({"thresholds": thresholds, "records": records["cal"],
                                                      "meta": meta}, indent=1))
    (art / "test_records.json").write_text(json.dumps({"records": records["test"], "meta": meta}, indent=1))
    print(json.dumps(meta, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()

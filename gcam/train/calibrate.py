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
    GCAM_OPS=decide.expense_review,retrieve.policy python -m gcam.train.calibrate   # redo these ops only

With ``GCAM_OPS`` the System-1 and classic models are refit (seconds, deterministic), the hub-state
probe is kept, and only the listed ops' thresholds and records are recomputed and merged into the
existing calibration and test records.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from ..domain import (MEMO_CLEAR, MEMO_RISK_DOWN, MEMO_RISK_UP, NOW, NOW_DT, CLAIM_REASONS, FIRST, POLICY,
                      POLICY_QUERIES, TOOLS, build_world, claims, estimates, policy_queries, reply_pairs,
                      tool_requests)
from ..executors.classic import ExpenseGBDT, PolicyTfidf, conformal_threshold
from ..executors.code import refund_rule, reply_checks, summary_checks
from ..executors.system1 import Encoder, EncoderHeads, HubProbe, LinearHead
from ..models import artifacts_dir
from ..system import CANDIDATES, System, rung_name

SMOKE = bool(os.environ.get("GCAM_SMOKE"))
OPS = tuple(o for o in os.environ.get("GCAM_OPS", "").split(",") if o)
TARGET = 0.99  # answered accuracy every abstention threshold must reach (frontier models score ~1.0 here)
TEACHER = Path(__file__).resolve().parents[1] / "data" / "teacher.json"
# Sized for a 4-core CPU (~0.9 s per hub forward, ~1.5 s per spoke forward).
N = {"fit": 480, "thr": 200, "cal": 80, "test": 80, "pair_thr": 20, "pair": 40, "reply": 9, "est": 30,
     "policy": 24, "exp_cal": 80, "exp_test": 120}
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
                          "product": o["product"], "reason": reason, "decision": d["decision"], "rule": d["rule"],
                          "explanation": d["explanation"]}, True))
        items.setdefault("create.reply", {})[split] = rows
        items.setdefault("create.reply_fallback", {})[split] = rows
    code = System(llms=False, world=world).impls["code.analytics"]
    windows = {"cal": range(30, 181, 20), "test": range(40, 171, 20)}
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
        "test": [({"row": r}, r["needs_review"]) for r in test_rows[: N["exp_test"]]],
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


def teacher() -> dict:
    """Training texts a frontier teacher wrote from the label definitions and training-split examples
    only (capability transfer; it never saw calibration, dev-test or benchmark items)."""
    if not TEACHER.exists():
        return {}
    return {k: v for k, v in json.loads(TEACHER.read_text()).items() if not k.startswith("_")}


def training_memos(rng: random.Random) -> list[tuple[str, str]]:
    """Training-wording memos with their signal (the domain's training templates, filled in)."""
    out = []
    for signal, memos in (("neither", MEMO_CLEAR.values()), ("review", MEMO_RISK_UP), ("approve", MEMO_RISK_DOWN)):
        for m in memos:
            out += [(m.format(who=rng.choice(FIRST), n=rng.randint(1000, 9999)), signal) for _ in range(4)]
    return out


def fit_system1(world, items) -> dict:
    enc = Encoder()
    heads = {}
    t = teacher()
    fit_claims = [(c.text, c.reason) for c in claims(N["fit"], split="train", seed=1)]
    fit_claims += [(x, y) for y, xs in t.get("claims", {}).items() for x in xs]
    oids = list(world.orders.order_id[:400])
    fit_tools = [(r.text, r.tool) for r in tool_requests(N["fit"], split="train", seed=1, order_ids=oids)]
    fit_tools += [(x, y) for y, xs in t.get("tools", {}).items() for x in xs]
    for op, data, labels in (("decide.claim_reason", fit_claims, CLAIM_REASONS), ("tool.select", fit_tools, TOOLS)):
        X = np.stack([enc.embed(x)[0] for x, _ in data])
        head = LinearHead(tuple(labels)).fit(X, [y for _, y in data])
        thr = items[op]["thr"]
        head.calibrate(np.stack([enc.embed(p["text"])[0] for p, _ in thr]), [y for _, y in thr], TARGET)
        heads[op] = head
    # policy-section head: training queries + teacher queries -> section
    pol = [(q, s.id) for s in POLICY for q in POLICY_QUERIES[s.id]["train"]]
    pol += [(x, y) for y, xs in t.get("policy", {}).items() for x in xs]
    if t.get("policy"):
        head = LinearHead(tuple(s.id for s in POLICY)).fit(np.stack([enc.embed(x)[0] for x, _ in pol]), [y for _, y in pol])
        thr = items["retrieve.policy"]["thr"]
        head.calibrate(np.stack([enc.embed(p["query"])[0] for p, _ in thr]), [y for _, y in thr], TARGET)
        heads["retrieve.policy"] = head
    # expense: the GBDT on the structured fields, and the memo-signal head for the reports it hands off
    gbdt = ExpenseGBDT().fit(items["decide.expense_review"]["fit"])
    gbdt.calibrate([p["row"] for p, _ in items["decide.expense_review"]["thr"]], TARGET)
    memos = training_memos(random.Random(5)) + [(x, y) for y, xs in t.get("memos", {}).items() for x in xs]
    heads["memo.signal"] = LinearHead(("approve", "neither", "review")).fit(
        np.stack([enc.embed(x)[0] for x, _ in memos]), [y for _, y in memos])
    s1 = EncoderHeads(enc, heads, gbdt=gbdt)
    # The memo head's threshold is set on the decision it makes (memo signal, or the GBDT on a
    # neutral memo) for the reports the GBDT abstains on.
    handed = [(p, y) for p, y in items["decide.expense_review"]["thr"] if not gbdt.run(
        "decide.expense_review", p, forced=False).answered]
    out = [s1.run("decide.expense_review", p, forced=True) for p, _ in handed]
    heads["memo.signal"].tau = conformal_threshold([o.confidence for o in out],
                                                   [o.value["needs_review"] == y for o, (_, y) in zip(out, handed)], TARGET)
    # dense retrieval margin threshold (used when no section head is trained)
    thr = items["retrieve.policy"]["thr"]
    res = [s1.run("retrieve.policy", p, forced=True) for p, _ in thr]
    retrieval_tau = conformal_threshold([r.confidence for r in res], [r.value["id"] == y for r, (_, y) in zip(res, thr)],
                                        TARGET)
    tfidf = PolicyTfidf()
    res = [tfidf.run("retrieve.policy", p, forced=True) for p, _ in thr]
    tfidf.tau = conformal_threshold([r.confidence for r in res], [r.value["id"] == y for r, (_, y) in zip(res, thr)],
                                    TARGET)
    return {"encoder_heads": heads, "retrieval_tau": retrieval_tau, "tfidf": tfidf, "gbdt": gbdt, "enc": enc}


def fit_probe(hub, items) -> dict[str, LinearHead]:
    probe = HubProbe(hub, PROBE_LAYER, {})
    data = [(c.text, c.reason) for c in claims(N["fit"], split="train", seed=1)]
    data += [(x, y) for y, xs in teacher().get("claims", {}).items() for x in xs]
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

    old_cal = json.loads((art / "calibration.json").read_text()) if OPS else None
    old_test = json.loads((art / "test_records.json").read_text()) if OPS else None
    probe = {}
    if OPS:  # the probe reads the hub's claims features only: keep it when there is one
        with open(art / "system1.pkl", "rb") as f:
            probe = pickle.load(f).get("probe_heads", {})
    # Write the sklearn artifacts first (System reads them), with the kept probe or a placeholder.
    saved = {"encoder_heads": s1["encoder_heads"], "retrieval_tau": s1["retrieval_tau"], "tfidf": s1["tfidf"],
             "gbdt": s1["gbdt"], "probe_layer": PROBE_LAYER, "probe_heads": probe}
    with open(art / "system1.pkl", "wb") as f:
        pickle.dump(saved, f)
    system = System(llms=True, art=art, world=world)
    if not probe:
        saved["probe_heads"] = fit_probe(system.models["hub"], items)
        with open(art / "system1.pkl", "wb") as f:
            pickle.dump(saved, f)
    system.impls["s1.hubprobe"].heads = saved["probe_heads"]
    print("probe kept" if probe else "probe fitted", f"recalibrating {OPS}" if OPS else "", f"{time.time() - t0:.0f}s",
          flush=True)

    # LLM thresholds on the threshold split (spoke is always the last rung: no threshold needed).
    thresholds = {k: v for k, v in old_cal["thresholds"].items() if k.split("|")[1] not in OPS} if OPS else {}
    for k, v in thresholds.items():
        impl, op = k.split("|")
        system.impls[impl].tau[op] = v
    for op in ("decide.claim_reason", "tool.select", "evaluate.pairwise", "decide.expense_review", "retrieve.policy"):
        if OPS and op not in OPS:
            continue
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
    if OPS:
        records = {"cal": {k: v for k, v in old_cal["records"].items() if k not in OPS},
                   "test": {k: v for k, v in old_test["records"].items() if k not in OPS}}
    for op, impls in CANDIDATES.items():
        if OPS and op not in OPS:
            continue
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
    if OPS:
        meta["recalibrated"] = old_cal["meta"].get("recalibrated", []) + [{"ops": list(OPS),
                                                                           "elapsed_s": meta["elapsed_s"]}]
        meta["elapsed_s"] = old_cal["meta"]["elapsed_s"]
    (art / "calibration.json").write_text(json.dumps({"thresholds": thresholds, "records": records["cal"],
                                                      "meta": meta}, indent=1))
    (art / "test_records.json").write_text(json.dumps({"records": records["test"], "meta": meta}, indent=1))
    print(json.dumps(meta, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()

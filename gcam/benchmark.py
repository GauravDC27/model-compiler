"""Benchmark GCAM against reference models on fresh, held-out items: accuracy, cost, latency.

The benchmark items come from seeds that neither training, calibration nor the dev test split
used; half of every text op uses held-out phrasings, and expense rows are the held-out period's
rows the dev split did not use. GCAM's ladders are compiled from calibration records only.

Reference models answer the same items blind: they get the inputs only, in shuffled order with
opaque ids, through the prompts GCAM's own LLM executors use (the canonical option order), and
their answers are scored locally with GCAM's checks (exact labels, the reply verifier, the
summary fact checks, a numeric tolerance for estimates).

    python -m gcam.benchmark export          # blinded item files for reference models
    python -m gcam.benchmark gcam            # run GCAM's compiled ladders, per item
    python -m gcam.benchmark score           # compare: accuracy deviation, $ cost, latency

Cost is reported in dollars on a stated basis: reference models at list price per token (token
counts estimated with a BPE tokenizer, thinking tokens excluded, so a lower bound); GCAM at the
measured wall-clock time of each item on a 4-vCPU / 16 GiB machine at on-demand price.
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from dmc.scheduler import CompileError, Node, compile_schedule

from .domain import NOW, NOW_DT, build_world, claims, estimates, policy_queries, reply_pairs, tool_requests
from .executors.code import refund_rule
from .executors.llm import CHOICE, LETTERS, SYSTEMS, pair_user, prompts, reply_user
from .planner import BARS  # the plans' bars: frontier parity on decisions, verifier-gated text
from .train.calibrate import correct, mix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "benchmark"
RESULTS = ROOT / "experiments" / "results"
SIZES = {"decide.claim_reason": 200, "tool.select": 200, "evaluate.pairwise": 80, "create.reply": 30,
         "analyze.estimate": 60, "retrieve.policy": 48}
SEEDS = {"decide.claim_reason": 101, "tool.select": 102, "evaluate.pairwise": 103, "create.reply": 141,
         "analyze.estimate": 153, "retrieve.policy": 163}
DEV_EXPENSE_ROWS = 120  # the dev test split uses the first 120 held-out-period rows

# Dollar basis. Reference models: list price per million tokens (input, output), 2026-10.
PRICE = {"claude-fable-5-1": (10.0, 50.0), "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
         "claude-haiku-5-5": (0.10, 0.50)}
# GCAM: a 4-vCPU / 16 GiB machine (this container's shape), AWS m7i.xlarge on-demand, us-east-1.
CPU_DOLLARS_PER_HOUR = 0.2016


# --------------------------------------------------------------------------- items
def build(world) -> dict[str, list[tuple[dict, object]]]:
    """op -> [(payload, truth)] for the benchmark split."""
    oids = list(world.orders.order_id[:400])
    items: dict[str, list] = {
        "decide.claim_reason": [({"text": c.text}, c.reason) for c in mix(claims, SIZES["decide.claim_reason"],
                                                                         SEEDS["decide.claim_reason"])],
        "tool.select": [({"text": r.text}, r.tool) for r in mix(tool_requests, SIZES["tool.select"],
                                                                SEEDS["tool.select"], order_ids=oids)],
        "evaluate.pairwise": [({"a": p.reply_a, "b": p.reply_b, "customer_name": world.customer(p.customer)["name"],
                                "order_id": p.order_id, "eligible": None}, p.better)
                              for p in reply_pairs(SIZES["evaluate.pairwise"], seed=SEEDS["evaluate.pairwise"],
                                                   world=world)],
        "analyze.estimate": [({"text": e.text}, e.answer) for e in mix(estimates, SIZES["analyze.estimate"],
                                                                       SEEDS["analyze.estimate"])],
        "retrieve.policy": [({"query": q.text}, q.section) for q in mix(policy_queries, SIZES["retrieve.policy"],
                                                                        SEEDS["retrieve.policy"])],
    }
    delivered = world.orders[world.orders.status == "delivered"].reset_index(drop=True)
    rng = np.random.default_rng(SEEDS["create.reply"])
    rows = []
    for i in rng.choice(len(delivered), size=SIZES["create.reply"], replace=False):
        o = {k: (v.item() if hasattr(v, "item") else v) for k, v in delivered.iloc[int(i)].to_dict().items()}
        reason = ("damaged", "not_received", "wrong_item")[len(rows) % 3]
        d = refund_rule(o, reason, NOW)
        rows.append(({"customer_name": world.customer(o["customer_id"])["name"], "order_id": o["order_id"],
                      "product": o["product"], "reason": reason, "decision": d["decision"], "rule": d["rule"],
                      "explanation": d["explanation"]}, True))
    items["create.reply"] = rows
    from .system import System
    code = System(llms=False, world=world).impls["code.analytics"]
    used = set(range(30, 181, 20)) | set(range(40, 171, 20))  # calibration and dev-test windows
    items["create.summary"] = [({"facts": code.run("analyze.refund_rates", {"days": d, "now": NOW}, forced=True).value},
                                True) for d in range(35, 181, 10) if d not in used]
    ex = world.expenses.sort_values("expense_id")
    held = ex[ex.date >= (NOW_DT - timedelta(days=30)).date().isoformat()]
    exp_rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
                for r in held.to_dict("records")][DEV_EXPENSE_ROWS:]
    items["decide.expense_review"] = [({"row": r}, r["needs_review"]) for r in exp_rows]
    return items


def blinded_ids(items) -> dict[str, list[str]]:
    """Opaque ids per item, assigned after a fixed shuffle (op -> ids in item order)."""
    out = {}
    for op, its in items.items():
        order = list(range(len(its)))
        random.Random(f"bench:{op}").shuffle(order)
        ids = [""] * len(its)
        for rank, i in enumerate(order):
            ids[i] = f"{op.split('.')[1][:4]}-{rank:03d}"
        out[op] = ids
    return out


ESTIMATE_SYSTEM = "You answer business arithmetic questions. Reply with the final number only."


def reference_prompt(op: str, payload: dict) -> tuple[str, str]:
    """The (system, user) prompt a reference model gets for one item: GCAM's own prompt in the
    canonical option order (estimates: asked for the number, not an expression)."""
    if op == "analyze.estimate":
        return ESTIMATE_SYSTEM, payload["text"]
    if op == "evaluate.pairwise":
        return SYSTEMS[op], pair_user(payload, "A", "B")
    if op in ("create.reply",):
        return SYSTEMS["create.reply"], reply_user(payload)
    return prompts(op, payload, 1)[0]


def export() -> None:
    world = build_world()
    items = build(world)
    ids = blinded_ids(items)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for op, its in items.items():
        rows = sorted(((ids[op][i], *reference_prompt(op, p)) for i, (p, _) in enumerate(its)), key=lambda r: r[0])
        system = rows[0][1]
        fmt = answer_format(op)
        lines = [f"# {op}: {len(rows)} items", "", f"System instruction for every item: {system}", "",
                 f"Answer format: {fmt}", ""]
        for iid, _, user in rows:
            lines += [f"## {iid}", user, ""]
        path = OUT / f"{op}.md"
        path.write_text("\n".join(lines))
        manifest[op] = {"file": str(path.relative_to(ROOT)), "n": len(rows)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


def answer_format(op: str) -> str:
    if op in CHOICE:
        return "the option letter (A, B, ...) for each item id"
    if op == "evaluate.pairwise":
        return "1 or 2 for each item id"
    if op == "analyze.estimate":
        return "the final number for each item id (no units, no thousands separators)"
    if op == "create.reply":
        return "the reply text for each item id"
    if op == "create.summary":
        return "the two-sentence summary for each item id"
    raise KeyError(op)


def decode(op: str, payload: dict, answer):
    """A reference model's raw answer -> the value GCAM's correctness check takes."""
    if answer is None:
        return None
    a = str(answer).strip()
    if op in CHOICE:
        canon = tuple(sorted(CHOICE[op][0]))
        letter = a[:1].upper()
        if letter not in LETTERS[: len(canon)]:
            return None
        lab = canon[LETTERS.index(letter)]
        if op == "decide.expense_review":
            return {"expense_id": payload["row"]["expense_id"], "needs_review": lab == "review"}
        if op == "retrieve.policy":
            return {"id": lab}
        return lab
    if op == "evaluate.pairwise":
        return {"1": "A", "2": "B"}.get(a[:1])
    if op == "analyze.estimate":
        try:
            return float(a.replace(",", "").replace("$", "").split()[0])
        except (ValueError, IndexError):
            return None
    return a  # replies and summaries: text


# --------------------------------------------------------------------------- GCAM
def run_gcam() -> None:
    from .runtime import Runtime
    from .system import System

    world = build_world()
    items = build(world)
    ids = blinded_ids(items)
    system = System(llms=True, world=world)
    rt = Runtime(system)
    report = {"lock_hash": rt.lock_hash, "ops": {}}

    def ladder_for(op: str, bar: float, cap: int) -> tuple[str, ...]:
        node = Node("n", op, required_accuracy=bar, input_cap=cap, requires_contract="D1")
        return compile_schedule([node], system.executors, token_budget=10**9, calibration=system.outcomes(),
                                lock_hash=rt.lock_hash).assignments[0].ladder

    def run_ladder(op: str, ladder, payload: dict, cap: int):
        capped, _ = rt._cap(op, payload, cap)
        tried, res = [], None
        for j, name in enumerate(ladder):
            rung = system.rungs[name]
            res = rung.impl.run(rung.op, capped, forced=j == len(ladder) - 1)
            tried.append((name, res.cost))
            if res.answered:
                break
        return res, tried

    for op, its in items.items():
        bar, cap = BARS[op]
        try:
            ladder = ladder_for(op, bar, cap)
            # The plans' pre-compiled variants for verifier-gated generation (gcam/planner.py).
            fallback = ladder_for("create.reply_fallback", *BARS["create.reply_fallback"]) if op == "create.reply" else None
        except CompileError as e:
            report["ops"][op] = {"compile_error": str(e)}
            continue
        rows = []
        for i, (payload, truth) in enumerate(its):
            t0 = time.perf_counter()
            res, tried = run_ladder(op, ladder, payload, cap)
            main_ok = bool(correct(op, res.value, truth, payload))
            value, ok, variant = res.value, main_ok, None
            if not main_ok and op == "create.reply":  # guard reply_verified -> spoke variant -> else abstain
                res2, tried2 = run_ladder("create.reply_fallback", fallback, payload, cap)
                tried += tried2
                variant = "spoke_reply"
                ok = bool(correct("create.reply_fallback", res2.value, truth, payload))
                value = res2.value if ok else None  # unverified text never ships: the goal abstains
            elif not main_ok and op == "create.summary":  # guard summary_verified -> code template
                tmpl = system.impls["code.analytics"].run("create.summary_template", payload, forced=True)
                tried.append(("code.analytics[create.summary_template]", tmpl.cost))
                variant, value = "template", tmpl.value
                ok = bool(correct(op, value, truth, payload))
            seconds = time.perf_counter() - t0
            rows.append({"id": ids[op][i], "value": value, "correct": ok, "main_path_correct": main_ok,
                         "variant": variant, "by": tried[-1][0], "escalations": len(tried) - 1,
                         "tok_eq": sum(c for _, c in tried), "seconds": round(seconds, 4)})
        report["ops"][op] = {"ladder": list(ladder), "fallback": list(fallback) if fallback else None, "items": rows}
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(op, ladder, f"acc {acc:.3f}", f"mean {np.mean([r['seconds'] for r in rows]):.3f}s", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "benchmark_gcam.json").write_text(json.dumps(report, indent=1, default=str))


# --------------------------------------------------------------------------- scoring
def _tokens():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return lambda s: len(tok(s, add_special_tokens=False).input_ids)


def paired_ci(a: list[bool], b: list[bool], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a paired item-level bootstrap 95% CI."""
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a_), size=(n, len(a_)))
    d = (a_[idx] - b_[idx]).mean(1)
    return float(a_.mean() - b_.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def score(models: list[str]) -> dict:
    world = build_world()
    items = build(world)
    ids = blinded_ids(items)
    count = _tokens()
    gcam = json.loads((RESULTS / "benchmark_gcam.json").read_text())
    report = {"cpu_dollars_per_hour": CPU_DOLLARS_PER_HOUR, "prices_per_mtok": PRICE, "ops": {}}
    for op, its in items.items():
        g = gcam["ops"].get(op, {})
        if "items" not in g:
            report["ops"][op] = {"gcam": g}
            continue
        g_ok = [r["correct"] for r in g["items"]]
        g_sec = [r["seconds"] for r in g["items"]]
        entry = {"n": len(its), "gcam": {
            "ladder": g["ladder"], "accuracy": round(float(np.mean(g_ok)), 4),
            "main_path_accuracy": round(float(np.mean([r["main_path_correct"] for r in g["items"]])), 4),
            "dollars_per_1k": round(1000 * float(np.mean(g_sec)) * CPU_DOLLARS_PER_HOUR / 3600, 5),
            "latency_p50_s": round(float(np.median(g_sec)), 4), "latency_p90_s": round(float(np.quantile(g_sec, 0.9)), 4),
            "tok_eq_mean": round(float(np.mean([r["tok_eq"] for r in g["items"]])), 1),
            "answered_by": {k: round(sum(r["by"] == k for r in g["items"]) / len(its), 3) for k in g["ladder"]}},
            "reference": {}}
        prompts_ = [reference_prompt(op, p) for p, _ in its]
        in_tok = [count(s) + count(u) for s, u in prompts_]
        for model in models:
            path = RESULTS / "benchmark_reference" / f"{model}.json"
            if not path.exists():
                continue
            answers = json.loads(path.read_text()).get(op, {})
            vals = [decode(op, p, answers.get(ids[op][i])) for i, (p, _) in enumerate(its)]
            ok = [bool(correct(op, v, t, p)) for v, (p, t) in zip(vals, its)]
            out_tok = [count(str(answers.get(ids[op][i], ""))) for i in range(len(its))]
            pin, pout = PRICE[model]
            dollars = [(a * pin + b * pout) / 1e6 for a, b in zip(in_tok, out_tok)]
            diff, lo, hi = paired_ci(g_ok, ok)
            entry["reference"][model] = {
                "accuracy": round(float(np.mean(ok)), 4), "answered": sum(v is not None for v in vals),
                "dollars_per_1k_lower_bound": round(1000 * float(np.mean(dollars)), 4),
                "gcam_minus_reference": {"diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)]},
                "cost_ratio_gcam_over_reference": round(entry["gcam"]["dollars_per_1k"] / (1000 * float(np.mean(dollars))), 5)
                if np.mean(dollars) else None}
        report["ops"][op] = entry
    (RESULTS / "benchmark_report.json").write_text(json.dumps(report, indent=1))
    return report


SHORT = {"claude-opus-5-5": "Opus 5.5", "claude-fable-5-1": "Fable 5.1", "claude-haiku-5-5": "Haiku 5.5"}
NAMES = {"decide.claim_reason": "Refund-claim reason", "tool.select": "Tool selection", "evaluate.pairwise": "Reply judging",
         "analyze.estimate": "Cost estimate", "retrieve.policy": "Policy retrieval", "create.reply": "Customer reply",
         "create.summary": "Analytics summary", "decide.expense_review": "Expense review"}


def render(report: dict) -> str:
    """Markdown tables from benchmark_report.json: accuracy and deviation, then cost and latency."""
    models = [m for m in SHORT if any(m in e.get("reference", {}) for e in report["ops"].values())]
    acc = ["| Use case (n) | GCAM | " + " | ".join(SHORT[m] for m in models) + " | GCAM − best reference (95% CI) |",
           "|---|---|" + "---|" * len(models) + "---|"]
    cost = ["| Use case | GCAM $/1k | " + " | ".join(f"{SHORT[m]} $/1k" for m in models)
            + " | GCAM p50 / p90 latency | GCAM answered by |", "|---|---|" + "---|" * len(models) + "---|---|"]
    for op, e in report["ops"].items():
        if "gcam" not in e or "accuracy" not in e["gcam"]:
            acc.append(f"| {NAMES[op]} | compile error | " + " | ".join("" for _ in models) + " | |")
            continue
        g, refs = e["gcam"], e["reference"]
        best = max(models, key=lambda m: refs[m]["accuracy"])
        d = refs[best]["gcam_minus_reference"]
        acc.append(f"| {NAMES[op]} ({e['n']}) | {g['accuracy']:.3f} | " + " | ".join(f"{refs[m]['accuracy']:.3f}" for m in models)
                   + f" | {d['diff']:+.3f} ({d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}) |")
        by = ", ".join(f"{k.split('[')[0]} {v:.0%}" for k, v in g["answered_by"].items() if v)
        cost.append(f"| {NAMES[op]} | {g['dollars_per_1k']:.4f} | "
                    + " | ".join(f"{refs[m]['dollars_per_1k_lower_bound']:.3f}" for m in models)
                    + f" | {g['latency_p50_s'] * 1000:.0f} ms / {g['latency_p90_s'] * 1000:.0f} ms | {by} |")
    return "\n".join(acc) + "\n\n" + "\n".join(cost) + "\n"


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "export":
        export()
    elif cmd == "gcam":
        run_gcam()
    elif cmd == "render":
        print(render(json.loads((RESULTS / "benchmark_report.json").read_text())))
    else:
        print(json.dumps(score(sys.argv[2:] or list(PRICE)), indent=1))


if __name__ == "__main__":
    main()

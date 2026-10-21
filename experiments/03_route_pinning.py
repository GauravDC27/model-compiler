"""Route-pinning pilot v2. Pre-registration: 03_route_pinning_prereg.md (with amendment 1).

Requires experiments/results/05_proxy_calibration.json (eps for H4, proxy regime).
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dmc import moe_probe as mp  # noqa: E402
from dmc.metrics import first_divergence, paraphrase_invariance  # noqa: E402
from dmc.oracle import generate  # noqa: E402
from dmc.profiles.decide import SYSTEM as DECIDE_SYSTEM  # noqa: E402
from dmc.profiles.decide import DecideProfile, label_orderings, render_user  # noqa: E402
from dmc.routing import (  # noqa: E402
    TOKENS,
    RoutePinner,
    RouteTracer,
    TraceMismatch,
    TraceOverrun,
    harden_routers,
)
from dmc.runtime import HFBackend  # noqa: E402
from prompts_v0 import FAMILIES, decision_ops  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
SYSTEM = "You are a helpful assistant."
C0 = 0.85
STACKS = {"REF": ("bf16", 4), "ALT": ("bf16", 3), "STRESS": ("fp32", 4)}
COVERAGES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)
SMOKE = bool(os.environ.get("DMC_SMOKE"))  # tiny end-to-end run to catch crashes
OUT = ROOT / f"experiments/results/03_route_pinning{'_smoke' if SMOKE else ''}.json"


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def rate(flags: list[bool]) -> dict:
    k, n = sum(flags), len(flags)
    return {"rate": round(k / n, 4) if n else None, "k": k, "n": n, "ci95": wilson(k, n)}


def at_coverage(rows: list[dict], key: str, coverage: float, signal: str) -> dict:
    """Parity on the `coverage` share of rows with the highest `signal` (ties: stable order)."""
    ranked = sorted(rows, key=lambda r: -r[signal])
    keep = ranked[: max(1, round(coverage * len(rows)))]
    return rate([r[key] for r in keep])


def binom_tail(k: int, n: int) -> float:
    """One-sided exact P(X >= k) for X ~ Binomial(n, 0.5) (paired sign / McNemar test)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n if n else 1.0


def save(result: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, default=str))


def main() -> None:
    t0 = time.time()
    random.seed(0)
    torch.manual_seed(0)
    calib = json.loads((ROOT / "experiments/results/05_proxy_calibration.json").read_text())
    eps = calib["eps_H4"]
    # This pilot manages R7 explicitly (Part 4 needs the stock router), so backends start stock.
    backends = {"bf16": HFBackend(MODEL, dtype="bfloat16", harden=False),
                "fp32": HFBackend(MODEL, dtype="float32", harden=False)}
    result: dict = {"model": MODEL, "stacks": STACKS, "C0": C0, "eps_H4": eps,
                    "eps_secondary": 0.01, "proxy_regime": calib["regime_ALT"]}

    def log(msg: str) -> None:
        print(f"{msg}  {time.time() - t0:.0f}s", flush=True)

    def stack(name, *, hardened=True):
        dtype, threads = STACKS[name]
        torch.set_num_threads(threads)
        b = backends[dtype]
        es = ExitStack()
        if hardened:
            es.enter_context(harden_routers(b.model))
        return b, es

    # ---- Part 4: value of R7 at the routing level (REF vs ALT, stock vs hardened routers)
    probe_texts = [t for fam in FAMILIES.values() for g in fam for t in g][: 2 if SMOKE else None]
    r7 = {}
    for hardened in (False, True):
        flips = []
        for text in probe_texts:
            per = {}
            for name in ("REF", "ALT"):
                b, es = stack(name, hardened=hardened)
                ids = torch.tensor([b._ids(b.render_chat(SYSTEM, text))])
                with es, torch.no_grad(), RouteTracer(b.model) as tr:
                    b.model(input_ids=ids)
                per[name] = tr.trace()
            for key in per["REF"]:
                if key != TOKENS:
                    a = per["REF"][key].sort(-1).values
                    c = per["ALT"][key].sort(-1).values
                    flips.extend((a != c).any(-1).tolist())
        r7["hardened" if hardened else "stock"] = rate(flips)
    result["R7_routing_flip_REF_vs_ALT"] = r7
    save(result)
    log(f"part 4 {r7}")

    # ---- Parts 1, 2, 5, 6: decisions (Decide profile, case/space-marginalized scorer)
    oracle = generate(1 if SMOKE else 15, seed=0)
    items = [(o.op(), o.answer, f"{o.family}:{o.group}") for o in oracle]
    items += [(op, None, f"probe:{i // 3}") for i, op in enumerate(decision_ops())][: 3 if SMOKE else None]
    rows = []
    for n, (op, answer, group) in enumerate(items):
        b, es = stack("REF")
        prompt0 = b.render_chat(DECIDE_SYSTEM, render_user(op, label_orderings(op.labels)[0]))
        content_start = len(b._ids(prompt0.split(op.question, 1)[0]))
        prof = DecideProfile(b, abstain_margin=0.0)
        with es, RouteTracer(b.model, record_margins=True) as tr:
            ref = prof.run(op)
        trace = tr.trace()
        row = {"group": group, "answer": answer, "ref": ref.label, "REF_margin": ref.margin,
               "REF_fragile": tr.fragile_share(eps),
               "REF_fragile_content": tr.fragile_share(eps, content_start)}
        for name in ("ALT", "STRESS"):
            b, es = stack(name)
            prof = DecideProfile(b, abstain_margin=0.0)
            with es, RouteTracer(b.model, record_margins=True) as tr_fresh:
                fresh = prof.run(op)
            row[f"{name}_margin"] = fresh.margin
            row[f"{name}_fragile"] = tr_fresh.fragile_share(eps)
            row[f"{name}_fragile_content"] = tr_fresh.fragile_share(eps, content_start)
            row[f"{name}_fragile_001"] = tr_fresh.fragile_share(0.01)
            row[f"{name}_fresh"] = fresh.label == ref.label
            b, es = stack(name)
            try:
                with es, RoutePinner(b.model, trace) as pin:
                    pinned = prof.run(op)
                row[f"{name}_pinned"] = pinned.label == ref.label
                row[f"{name}_pinned_status"] = pin.status
            except (TraceMismatch, TraceOverrun) as e:  # recorded as a pinned failure (K6)
                row[f"{name}_pinned"] = False
                row[f"{name}_pinned_status"] = f"failed:{type(e).__name__}"
        row["correct"] = None if answer is None else ref.label == answer
        rows.append(row)
        if n % 25 == 0:
            log(f"decision {n}/{len(items)}")
            result["decision_rows_partial"] = rows
            save(result)

    decisions = {}
    for name in ("ALT", "STRESS"):
        fresh_wrong = [r for r in rows if not r[f"{name}_fresh"]]
        repaired = sum(r[f"{name}_pinned"] for r in fresh_wrong)
        introduced = sum(1 for r in rows if r[f"{name}_fresh"] and not r[f"{name}_pinned"])
        p = binom_tail(repaired, repaired + introduced)
        decisions[name] = {
            "fresh": rate([r[f"{name}_fresh"] for r in rows]),
            "pinned": rate([r[f"{name}_pinned"] for r in rows]),
            "pinned_failures": sum(r[f"{name}_pinned_status"] != "full" for r in rows),
            "H1_repaired": repaired, "H1_introduced": introduced, "H1_p_one_sided": p,
            "H1": ("uninformative (fresh flips < 5)" if len(fresh_wrong) < 5
                   else "supported" if repaired > introduced and p < 0.05 else "not supported"),
            # Primary D2-fresh curve: ranked by the executing stack's own margin (K1).
            "fresh_at_C0_own_margin": at_coverage(rows, f"{name}_fresh", C0, f"{name}_margin"),
            "fresh_at_C0_ref_margin": at_coverage(rows, f"{name}_fresh", C0, "REF_margin"),
            "pinned_at_C0_ref_margin": at_coverage(rows, f"{name}_pinned", C0, "REF_margin"),
            "curve_fresh_own_margin": {c: at_coverage(rows, f"{name}_fresh", c, f"{name}_margin")["rate"]
                                       for c in COVERAGES},
            "curve_pinned_ref_margin": {c: at_coverage(rows, f"{name}_pinned", c, "REF_margin")["rate"]
                                        for c in COVERAGES},
        }
    result["decision_parity"] = decisions

    orows = [r for r in rows if r["answer"] is not None]
    groups: dict[str, list] = {}
    for r in orows:
        groups.setdefault(r["group"], []).append(r)
    result["quality_REF"] = {
        "oracle_accuracy": rate([r["correct"] for r in orows]),
        "oracle_accuracy_at_C0": at_coverage(orows, "correct", C0, "REF_margin"),
        "PIR": paraphrase_invariance([[r["ref"] for r in g] for g in groups.values()]),
        "CCR": sum(all(r["correct"] for r in g) for g in groups.values()) / len(groups),
        "share_yes_predicted": sum(r["ref"] == "yes" for r in orows) / len(orows),
    }

    # ---- Part 5: ND2 incremental test on the executing stack (ALT), split-half by group.
    gkeys = sorted({r["group"] for r in rows})
    random.Random(1).shuffle(gkeys)
    half_a = set(gkeys[: len(gkeys) // 2])
    A = [r for r in rows if r["group"] in half_a]
    B = [r for r in rows if r["group"] not in half_a]

    def nd2(fragile_key: str) -> dict:
        def combo(rs, alpha):
            for r in rs:
                r["combo"] = r["ALT_margin"] - alpha * r[fragile_key]
            return rs

        alphas = [0, 1, 2, 5, 10, 20, 50, 100]
        best = max(alphas, key=lambda a: (at_coverage(combo(A, a), "ALT_fresh", C0, "combo")["rate"], -a))
        combo(B, best)
        base_b = at_coverage(B, "ALT_fresh", C0, "ALT_margin")["rate"]
        comb_b = at_coverage(B, "ALT_fresh", C0, "combo")["rate"]
        rng = random.Random(2)
        bgroups = sorted({r["group"] for r in B})
        by_group = {g: [r for r in B if r["group"] == g] for g in bgroups}
        boot = []
        for _ in range(50 if SMOKE else 1000):
            sample = [r for g in (rng.choice(bgroups) for _ in bgroups) for r in by_group[g]]
            boot.append(at_coverage(sample, "ALT_fresh", C0, "combo")["rate"]
                        - at_coverage(sample, "ALT_fresh", C0, "ALT_margin")["rate"])
        boot.sort()
        lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
        verdict = ("support (ND2 stays a MUST)" if lo > 0
                   else "demote (ND2 becomes a diagnostic)" if hi < 0.005
                   else "inconclusive at this n (confirmatory test at hub scale, n >= 2000)")
        return {"alpha": best, "parity_margin_only_B": base_b, "parity_combined_B": comb_b,
                "delta": comb_b - base_b, "delta_ci95": (lo, hi), "H4": verdict}

    result["ND2_incremental"] = {
        "primary_all_positions": nd2("ALT_fragile"),
        "secondary_content_only": nd2("ALT_fragile_content"),
        "secondary_eps_0.01": nd2("ALT_fragile_001"),
        "auc_ALT_fragile_predicts_flip": mp.roc_auc(
            torch.tensor([r["ALT_fragile"] for r in rows]),
            torch.tensor([not r["ALT_fresh"] for r in rows])),
        "auc_ALT_neg_margin_predicts_flip": mp.roc_auc(
            torch.tensor([-r["ALT_margin"] for r in rows]),
            torch.tensor([not r["ALT_fresh"] for r in rows])),
    }
    result["n_decisions"] = len(rows)
    result["decision_rows"] = rows
    result.pop("decision_rows_partial", None)
    save(result)
    log("decisions done")

    # ---- Part 3: generation, teacher-forced and free-running
    gen = {name: {"tf_fresh": [], "tf_pinned": [], "div_fresh": [], "div_pinned": []}
           for name in ("ALT", "STRESS")}
    ref_single_pass = []  # K11: REF single-pass vs its own incremental decode (CPU prefill/decode leak)
    gen_prompts = [paras[0] for fam in FAMILIES.values() for paras in fam][: 2 if SMOKE else None]
    for text in gen_prompts:
        b, es = stack("REF")
        prompt = b.render_chat(SYSTEM, text)
        with es, RouteTracer(b.model) as tr:
            ref_ids = b.generate_ids(prompt, 32)
        trace = tr.trace()
        n_prompt = len(b._ids(prompt))
        fed = trace[TOKENS][None]
        b, es = stack("REF")
        with es, torch.no_grad():
            pred = b.model(input_ids=fed).logits[0][n_prompt - 1: n_prompt - 1 + len(ref_ids)].argmax(-1)
        ref_single_pass.extend(p == t for p, t in zip(pred.tolist(), ref_ids))
        for name in ("ALT", "STRESS"):
            for mode in ("fresh", "pinned"):
                b, es = stack(name)
                with es, ExitStack() as pins, torch.no_grad():
                    if mode == "pinned":
                        pins.enter_context(RoutePinner(b.model, trace))
                    logits = b.model(input_ids=fed).logits[0]
                pred = logits[n_prompt - 1: n_prompt - 1 + len(ref_ids)].argmax(-1).tolist()
                gen[name][f"tf_{mode}"].extend(p == t for p, t in zip(pred, ref_ids))
                if name == "STRESS" and mode == "pinned":
                    continue  # accepted cut (round-3 review §3): no free-running pinned STRESS
                b, es = stack(name)
                with es:
                    if mode == "fresh":
                        out = b.generate_ids(prompt, 32)
                    else:
                        # Non-strict: routes are pinned exactly up to the first divergent token;
                        # only the divergence index is reported (review G1).
                        with RoutePinner(b.model, trace, strict=False):
                            out = b.generate_ids(prompt, 32)
                gen[name][f"div_{mode}"].append(first_divergence(ref_ids, out))
    generation = {"REF_single_pass_vs_incremental": rate(ref_single_pass)}
    for name, g in gen.items():
        generation[name] = {
            "teacher_forced_agreement_fresh": rate(g["tf_fresh"]),
            "teacher_forced_agreement_pinned": rate(g["tf_pinned"]),
            "free_running_identical_fresh": sum(d is None for d in g["div_fresh"]) / len(g["div_fresh"]),
            "first_divergence_fresh": g["div_fresh"],
        }
        if g["div_pinned"]:
            generation[name]["free_running_identical_pinned"] = (
                sum(d is None for d in g["div_pinned"]) / len(g["div_pinned"]))
            generation[name]["first_divergence_pinned"] = g["div_pinned"]
    result["generation"] = generation
    result["elapsed_s"] = round(time.time() - t0, 1)
    save(result)
    print(json.dumps({k: v for k, v in result.items() if k != "decision_rows"}, indent=1, default=str))


if __name__ == "__main__":
    main()

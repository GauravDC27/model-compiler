"""Route-pinning pilot v2. Pre-registration: 03_route_pinning_prereg.md."""

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
from dmc.profiles.decide import DecideProfile  # noqa: E402
from dmc.routing import TOKENS, RoutePinner, RouteTracer, harden_routers  # noqa: E402
from dmc.runtime import HFBackend  # noqa: E402
from prompts_v0 import FAMILIES, decision_ops  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
SYSTEM = "You are a helpful assistant."
C0 = 0.85
EPS = 0.01
STACKS = {"REF": ("bf16", 4), "ALT": ("bf16", 3), "STRESS": ("fp32", 4)}
SMOKE = bool(os.environ.get("DMC_SMOKE"))  # tiny end-to-end run to catch crashes


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


def at_coverage(rows: list[dict], key: str, coverage: float, signal: str = "margin") -> dict:
    ranked = sorted(rows, key=lambda r: -r[signal])
    keep = ranked[: max(1, round(coverage * len(rows)))]
    return rate([r[key] for r in keep])


def main() -> None:
    t0 = time.time()
    random.seed(0)
    torch.manual_seed(0)
    backends = {"bf16": HFBackend(MODEL, dtype="bfloat16"), "fp32": HFBackend(MODEL, dtype="float32")}

    def stack(name, *, hardened=True):
        dtype, threads = STACKS[name]
        torch.set_num_threads(threads)
        b = backends[dtype]
        es = ExitStack()
        if hardened:
            es.enter_context(harden_routers(b.model))
        return b, es

    # ---- Part 4: value of R7 at the routing level (REF vs ALT, stock vs hardened routers)
    k = backends["bf16"].model.config.num_experts_per_tok
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
    print("R7", r7, f"{time.time() - t0:.0f}s", flush=True)

    # ---- Parts 1, 2, 5, 6: decisions
    oracle = generate(1 if SMOKE else 20, seed=0)
    items = [(o.op(), o.answer, f"{o.family}:{o.group}") for o in oracle]
    items += [(op, None, f"probe:{i // 3}") for i, op in enumerate(decision_ops())][: 3 if SMOKE else None]
    rows = []
    for n, (op, answer, group) in enumerate(items):
        b, es = stack("REF")
        prof = DecideProfile(b, abstain_margin=0.0)
        with es, RouteTracer(b.model, record_margins=True) as tr:
            ref = prof.run(op)
        trace, fragile = tr.trace(), tr.fragile_share(EPS)
        row = {"group": group, "answer": answer, "ref": ref.label, "margin": ref.margin,
               "fragile": fragile}
        for name in ("ALT", "STRESS"):
            b, es = stack(name)
            prof = DecideProfile(b, abstain_margin=0.0)
            with es:
                fresh = prof.run(op)
            b, es = stack(name)
            with es, RoutePinner(b.model, trace) as pin:
                pinned = prof.run(op)
            assert pin.status == "full"
            row[f"{name}_fresh"] = fresh.label == ref.label
            row[f"{name}_pinned"] = pinned.label == ref.label
        row["correct"] = None if answer is None else ref.label == answer
        rows.append(row)
        if n % 30 == 0:
            print(f"decision {n}/{len(items)} {time.time() - t0:.0f}s", flush=True)

    decisions = {}
    for name in ("ALT", "STRESS"):
        decisions[name] = {
            "fresh": rate([r[f"{name}_fresh"] for r in rows]),
            "pinned": rate([r[f"{name}_pinned"] for r in rows]),
            "fresh_at_C0": at_coverage(rows, f"{name}_fresh", C0),
            "pinned_at_C0": at_coverage(rows, f"{name}_pinned", C0),
            "curve_fresh": {c: at_coverage(rows, f"{name}_fresh", c)["rate"]
                            for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)},
            "curve_pinned": {c: at_coverage(rows, f"{name}_pinned", c)["rate"]
                             for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)},
        }
    orows = [r for r in rows if r["answer"] is not None]
    groups: dict[str, list] = {}
    for r in orows:
        groups.setdefault(r["group"], []).append(r)
    quality = {
        "oracle_accuracy": rate([r["correct"] for r in orows]),
        "oracle_accuracy_at_C0": at_coverage(orows, "correct", C0),
        "PIR": paraphrase_invariance([[r["ref"] for r in g] for g in groups.values()]),
        "CCR": sum(all(r["correct"] for r in g) for g in groups.values()) / len(groups),
        "share_yes_predicted": sum(r["ref"] == "yes" for r in orows) / len(orows),
    }

    # ND2 incremental test (split-half by group), ALT parity at C0.
    gkeys = sorted({r["group"] for r in rows})
    random.Random(1).shuffle(gkeys)
    half_a = set(gkeys[: len(gkeys) // 2])
    A = [r for r in rows if r["group"] in half_a]
    B = [r for r in rows if r["group"] not in half_a]

    def combined(rs, alpha):
        for r in rs:
            r["combo"] = r["margin"] - alpha * r["fragile"]
        return rs

    alphas = [0, 1, 2, 5, 10, 20, 50, 100]
    best = max(alphas, key=lambda a: at_coverage(combined(A, a), "ALT_fresh", C0, "combo")["rate"])
    combined(B, best)
    base_b = at_coverage(B, "ALT_fresh", C0)["rate"]
    combo_b = at_coverage(B, "ALT_fresh", C0, "combo")["rate"]
    boot = []
    rng = random.Random(2)
    bgroups = sorted({r["group"] for r in B})
    for _ in range(50 if SMOKE else 1000):
        pick = [rng.choice(bgroups) for _ in bgroups]
        sample = [r for g in pick for r in B if r["group"] == g]
        boot.append(at_coverage(sample, "ALT_fresh", C0, "combo")["rate"]
                    - at_coverage(sample, "ALT_fresh", C0)["rate"])
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
    nd2 = {"alpha": best, "parity_margin_only_B": base_b, "parity_combined_B": combo_b,
           "delta": combo_b - base_b, "delta_ci95": (lo, hi),
           "auc_fragile_predicts_alt_flip": mp.roc_auc(
               torch.tensor([r["fragile"] for r in rows]),
               torch.tensor([not r["ALT_fresh"] for r in rows])),
           "auc_neg_margin_predicts_alt_flip": mp.roc_auc(
               torch.tensor([-r["margin"] for r in rows]),
               torch.tensor([not r["ALT_fresh"] for r in rows]))}
    print("decisions done", f"{time.time() - t0:.0f}s", flush=True)

    # ---- Part 3: generation, teacher-forced and free-running
    gen = {name: {"tf_fresh": [], "tf_pinned": [], "div_fresh": [], "div_pinned": []}
           for name in ("ALT", "STRESS")}
    gen_prompts = [paras[0] for fam in FAMILIES.values() for paras in fam][: 2 if SMOKE else None]
    for text in gen_prompts:
        b, es = stack("REF")
        prompt = b.render_chat(SYSTEM, text)
        with es, RouteTracer(b.model) as tr:
            ref_ids = b.generate_ids(prompt, 32)
        trace = tr.trace()
        n_prompt = len(b._ids(prompt))
        fed = trace[TOKENS][None]
        for name in ("ALT", "STRESS"):
            for mode in ("fresh", "pinned"):
                b, es = stack(name)
                with es, ExitStack() as pins, torch.no_grad():
                    if mode == "pinned":
                        pins.enter_context(RoutePinner(b.model, trace))
                    logits = b.model(input_ids=fed).logits[0]
                pred = logits[n_prompt - 1: n_prompt - 1 + len(ref_ids)].argmax(-1).tolist()
                gen[name][f"tf_{mode}"].extend(p == t for p, t in zip(pred, ref_ids))
                b, es = stack(name)
                with es:
                    if mode == "fresh":
                        out = b.generate_ids(prompt, 32)
                        div = first_divergence(ref_ids, out)
                    else:
                        # Non-strict: routes are pinned exactly up to the first divergent
                        # token; only the divergence index is reported (review G1).
                        with RoutePinner(b.model, trace, strict=False):
                            out = b.generate_ids(prompt, 32)
                        div = first_divergence(ref_ids, out)
                gen[name][f"div_{mode}"].append(div)
    generation = {}
    for name, g in gen.items():
        generation[name] = {
            "teacher_forced_agreement_fresh": rate(g["tf_fresh"]),
            "teacher_forced_agreement_pinned": rate(g["tf_pinned"]),
            "free_running_identical_fresh": sum(d is None for d in g["div_fresh"]) / len(g["div_fresh"]),
            "free_running_identical_pinned": sum(d is None for d in g["div_pinned"]) / len(g["div_pinned"]),
            "first_divergence_fresh": g["div_fresh"],
            "first_divergence_pinned": g["div_pinned"],
        }

    result = {
        "model": MODEL, "stacks": STACKS, "C0": C0, "eps": EPS, "n_decisions": len(rows),
        "R7_routing_flip_REF_vs_ALT": r7,
        "decision_parity": decisions,
        "quality_REF": quality,
        "ND2_incremental": nd2,
        "generation": generation,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = ROOT / f"experiments/results/03_route_pinning{'_smoke' if SMOKE else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()

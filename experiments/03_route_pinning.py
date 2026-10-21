"""Route-pinning pilot (plan v2, R27 / D2-replay; review finding F6).

Reference stack: granite-3.1-1b-a400m in FP32. "Other toolchain": the same weights in BF16.
For every decision (decide + judge prompt families, Decide profile, order-averaged label
scoring) and every greedy continuation, compare the BF16 result with the FP32 reference:
  * fresh  — BF16 routes on its own (contract D2-fresh);
  * pinned — BF16 re-executes with the FP32 routing trace pinned (contract D2-replay).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dmc.metrics import first_divergence  # noqa: E402
from dmc.profiles.decide import DecideProfile  # noqa: E402
from dmc.routing import RoutePinner, RouteTracer, trace_size_bytes  # noqa: E402
from dmc.runtime import HFBackend  # noqa: E402
from prompts_v0 import FAMILIES, decision_ops  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
SYSTEM = "You are a helpful assistant."


def main() -> None:
    t0 = time.time()
    torch.set_num_threads(4)
    ref = HFBackend(MODEL, dtype="float32")
    alt = HFBackend(MODEL, dtype="bfloat16")
    p_ref = DecideProfile(ref, abstain_margin=0.0)
    p_alt = DecideProfile(alt, abstain_margin=0.0)

    rows = []
    for op in decision_ops():
        with RouteTracer(ref.model) as tr:
            r = p_ref.run(op)
        fresh = p_alt.run(op)
        with RoutePinner(alt.model, tr.trace()) as pin:
            pinned = p_alt.run(op)
        assert pin.overruns == 0
        rows.append({
            "q": op.question[:60],
            "ref": r.label, "fresh": fresh.label, "pinned": pinned.label,
            "ref_margin": round(r.margin, 4),
            "fresh_score_err": max(abs(r.scores[k] - fresh.scores[k]) for k in r.scores),
            "pinned_score_err": max(abs(r.scores[k] - pinned.scores[k]) for k in r.scores),
        })

    gen = []
    trace_bytes = []
    for fam in FAMILIES.values():
        for paras in fam:
            prompt = ref.render_chat(SYSTEM, paras[0])
            with RouteTracer(ref.model) as tr:
                a = ref.generate(prompt, 32)
            trace = tr.trace()
            trace_bytes.append(trace_size_bytes(trace))
            b_fresh = alt.generate(prompt, 32)
            with RoutePinner(alt.model, trace):
                b_pin = alt.generate(prompt, 32)
            ta = ref.tok(a, add_special_tokens=False).input_ids
            gen.append({
                "fresh_equal": a == b_fresh,
                "pinned_equal": a == b_pin,
                "fresh_div": first_divergence(ta, ref.tok(b_fresh, add_special_tokens=False).input_ids),
                "pinned_div": first_divergence(ta, ref.tok(b_pin, add_special_tokens=False).input_ids),
            })

    n = len(rows)
    result = {
        "model": MODEL,
        "reference": "float32", "other_toolchain": "bfloat16",
        "decisions": n,
        "D2_fresh_parity": sum(r["ref"] == r["fresh"] for r in rows) / n,
        "D2_replay_parity_pinned": sum(r["ref"] == r["pinned"] for r in rows) / n,
        "mean_max_score_err_fresh": sum(r["fresh_score_err"] for r in rows) / n,
        "mean_max_score_err_pinned": sum(r["pinned_score_err"] for r in rows) / n,
        "flipped_fresh": [r for r in rows if r["ref"] != r["fresh"]],
        "flipped_pinned": [r for r in rows if r["ref"] != r["pinned"]],
        "greedy32_text_equal_fresh": sum(g["fresh_equal"] for g in gen) / len(gen),
        "greedy32_text_equal_pinned": sum(g["pinned_equal"] for g in gen) / len(gen),
        "greedy32_first_divergence_fresh": [g["fresh_div"] for g in gen],
        "greedy32_first_divergence_pinned": [g["pinned_div"] for g in gen],
        "mean_trace_bytes_per_generation": sum(trace_bytes) / len(trace_bytes),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = ROOT / "experiments/results/03_route_pinning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

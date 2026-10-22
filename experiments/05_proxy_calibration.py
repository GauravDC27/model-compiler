"""Calibration of the cross-engine proxies used by pilots 03 and 04 (review K2/K7).

On the 75 probe prompts, with R7-hardened routers, compares REF (BF16, 4 threads) with
ALT (BF16, 3 threads; same precision, different reduction order) and with STRESS (FP32):
  * |delta router logit| distribution and the per-position max |delta output logit|
    (median, p95, p99, max), over all positions and excluding the first position;
  * per-position KL(REF || other) of the next-token distribution (mean, median, p95);
  * routing-flip rate.
Pre-registered uses (03 prereg amendment 1, 04 prereg amendment 2):
  * regime = "faithful" if mean per-position KL(REF||ALT) <= 10 x 1.5e-3 (the published
    Qwen3-30B-A3B cross-engine KL), else "stress";
  * eps for the fragile-routing share in H4 = p95 of |delta router logit| REF vs ALT,
    excluding the first position.
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

from dmc import moe_probe as mp  # noqa: E402
from dmc.runtime import HFBackend  # noqa: E402
from prompts_v0 import FAMILIES  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
SYSTEM = "You are a helpful assistant."
PUBLISHED_KL = 1.5e-3
STACKS = {"REF": ("bf16", 4), "ALT": ("bf16", 3), "STRESS": ("fp32", 4)}


def stats(x: torch.Tensor) -> dict:
    x = x.flatten().double()
    return {"median": float(x.median()), "p95": float(x.quantile(0.95)),
            "p99": float(x.quantile(0.99)), "max": float(x.max())}


def main() -> None:
    t0 = time.time()
    backends = {"bf16": HFBackend(MODEL, dtype="bfloat16"), "fp32": HFBackend(MODEL, dtype="float32")}
    k = backends["bf16"].model.config.num_experts_per_tok
    texts = [t for fam in FAMILIES.values() for g in fam for t in g]
    acc = {name: {"dr": [], "dr_nf": [], "do": [], "do_nf": [], "kl": [], "flips": []}
           for name in ("ALT", "STRESS")}
    ref_margins = []
    timing = {name: 0.0 for name in STACKS}
    for i, text in enumerate(texts):
        outs = {}
        for name, (dtype, threads) in STACKS.items():
            torch.set_num_threads(threads)
            b = backends[dtype]
            ids = torch.tensor([b._ids(b.render_chat(SYSTEM, text))])
            t = time.time()
            with torch.no_grad():
                o = b.model(input_ids=ids, output_router_logits=True)
            timing[name] += time.time() - t
            rl = torch.stack([r.float() for r in o.router_logits])  # L, T, E
            outs[name] = (rl, o.logits[0].float())
        rl_ref, lg_ref = outs["REF"]
        ref_margins.append(mp.margins(rl_ref, k).flatten())
        for name in ("ALT", "STRESS"):
            rl, lg = outs[name]
            a = acc[name]
            dr = (rl - rl_ref).abs()
            do = (lg - lg_ref).abs().amax(-1)  # per-position max |delta| over the vocabulary
            a["dr"].append(dr.flatten())
            a["dr_nf"].append(dr[:, 1:].flatten())
            a["do"].append(do.flatten())
            a["do_nf"].append(do[1:].flatten())
            p = lg_ref.log_softmax(-1)
            q = lg.log_softmax(-1)
            a["kl"].append((p.exp() * (p - q)).sum(-1))
            a["flips"].append(mp.flips(rl_ref, rl, k).flatten())
        if i % 15 == 0:
            print(f"{i}/{len(texts)} {time.time() - t0:.0f}s", flush=True)

    result = {"model": MODEL, "stacks": STACKS, "n_prompts": len(texts),
              "seconds_per_forward": {n: round(v / len(texts), 3) for n, v in timing.items()},
              "ref_router_margin": stats(torch.cat(ref_margins))}
    for name, a in acc.items():
        kl = torch.cat(a["kl"]).double()
        result[name] = {
            "router_logit_absdiff": stats(torch.cat(a["dr"])),
            "router_logit_absdiff_excl_first": stats(torch.cat(a["dr_nf"])),
            "output_logit_absdiff_per_position_max": stats(torch.cat(a["do"])),
            "output_logit_absdiff_per_position_max_excl_first": stats(torch.cat(a["do_nf"])),
            "kl_per_position": {"mean": float(kl.mean()), "median": float(kl.median()),
                                "p95": float(kl.quantile(0.95))},
            "routing_flip_rate": float(torch.cat(a["flips"]).double().mean()),
        }
    alt_kl = result["ALT"]["kl_per_position"]["mean"]
    result["regime_ALT"] = "faithful" if alt_kl <= 10 * PUBLISHED_KL else "stress"
    result["eps_H4"] = result["ALT"]["router_logit_absdiff_excl_first"]["p95"]
    result["elapsed_s"] = round(time.time() - t0, 1)
    out = ROOT / "experiments/results/05_proxy_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

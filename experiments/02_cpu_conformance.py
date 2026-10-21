"""Phase-0 conformance on CPU: which "toolchain" changes alter outputs, and does
decision extraction by label scoring survive changes that break bitwise text equality?

Variations on granite-3.1-1b-a400m-instruct, one request per forward pass (fixed shape):
  * repeat      — same settings, three runs              (must agree: D1)
  * threads     — 1 vs 2 vs 4 intra-op threads           (a reduction-order change, like TP size)
  * dtype       — FP32 vs BF16 weights                   (a numerics change, like a new SKU/kernel)

Outputs compared: greedy 32-token continuations (bitwise text, D1) and Decide-profile
labels from label scoring (decision parity, D2) on the decide + judge prompt families.
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

from dmc import conformance  # noqa: E402
from dmc.profiles.decide import DecideProfile  # noqa: E402
from dmc.runtime import HFBackend  # noqa: E402
from prompts_v0 import FAMILIES, decision_ops  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
SYSTEM = "You are a helpful assistant."


def main() -> None:
    t0 = time.time()
    b32 = HFBackend(MODEL, dtype="float32")
    b16 = HFBackend(MODEL, dtype="bfloat16")
    backends = {"float32": b32, "bfloat16": b16}

    def with_threads(n: int):
        torch.set_num_threads(n)

    gen_prompts = [paras[0] for fam in FAMILIES.values() for paras in fam]

    def run_gen(text: str, variant: str) -> str:
        dtype, threads, _rep = variant.split("|")
        with_threads(int(threads))
        b = backends[dtype]
        return b.generate(b.render_chat(SYSTEM, text), 32)

    repeat = conformance.check(gen_prompts, ["float32|4|r0", "float32|4|r1", "float32|4|r2"], run_gen)
    threads = conformance.check(gen_prompts, ["float32|4|r0", "float32|2|r0", "float32|1|r0"], run_gen)
    dtype = conformance.check(gen_prompts, ["float32|4|r0", "bfloat16|4|r0"], run_gen)

    ops = decision_ops()
    profiles = {d: DecideProfile(b, abstain_margin=0.0) for d, b in backends.items()}
    gated = {d: DecideProfile(b, abstain_margin=1.0) for d, b in backends.items()}
    by_q = {op.question: op for op in ops}

    def run_dec(q: str, variant: str) -> str:
        dtype, threads = variant.split("|")
        with_threads(int(threads))
        return str(profiles[dtype].run(by_q[q]).label)

    def run_gated(q: str, variant: str) -> str:
        dtype, threads = variant.split("|")
        with_threads(int(threads))
        r = gated[dtype].run(by_q[q])
        return str(r.label)

    qs = list(by_q)
    dec_threads = conformance.check(qs, ["float32|4", "float32|2", "float32|1"], run_dec)
    dec_dtype = conformance.check(qs, ["float32|4", "bfloat16|4"], run_dec)
    # With an abstain band, count parity only on items both variants decided.
    both = [(run_gated(q, "float32|4"), run_gated(q, "bfloat16|4")) for q in qs]
    decided_both = [(a, b) for a, b in both if a != "None" and b != "None"]
    gated_parity = sum(a == b for a, b in decided_both) / max(1, len(decided_both))

    # Margin distribution of decisions (FP32) to relate flips to margins.
    with_threads(4)
    margins = {q: profiles["float32"].run(by_q[q]).margin for q in qs}
    flipped = [d["request"] for d in dec_dtype.disagreements]

    result = {
        "model": MODEL,
        "D1_greedy_text_repeat_agreement": repeat.agreement_rate,
        "D1_greedy_text_thread_agreement": threads.agreement_rate,
        "D1_greedy_text_dtype_agreement": dtype.agreement_rate,
        "D2_decision_thread_agreement": dec_threads.agreement_rate,
        "D2_decision_dtype_agreement": dec_dtype.agreement_rate,
        "D2_decision_dtype_agreement_with_abstain_band_1.0": gated_parity,
        "abstain_band_coverage": len(decided_both) / len(qs),
        "n_generation_prompts": len(gen_prompts),
        "n_decisions": len(qs),
        "flipped_decision_margins_fp32": [round(margins[q], 4) for q in flipped],
        "median_decision_margin_fp32": sorted(margins.values())[len(margins) // 2],
        "thread_disagreements": threads.disagreements[:3],
        "dtype_disagreements": dtype.disagreements[:3],
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = ROOT / "experiments/results/02_cpu_conformance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

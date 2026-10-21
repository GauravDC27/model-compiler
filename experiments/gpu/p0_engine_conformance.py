"""GPU Phase 0: D1 conformance under concurrent load and D2-fresh decision parity across
endpoints, against OpenAI-compatible servers (vLLM / SGLang). See RUNBOOK.md.

Usage:
  python experiments/gpu/p0_engine_conformance.py --ref http://localhost:8000 \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 [--alt http://other:8000] [--n 100]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dmc.http_backend import OpenAICompatBackend  # noqa: E402
from dmc.metrics import unique_at_n  # noqa: E402
from dmc.oracle import generate  # noqa: E402
from dmc.profiles.decide import DecideProfile  # noqa: E402
from prompts_v0 import FAMILIES  # noqa: E402

SYSTEM = "You are a helpful assistant."
C0 = 0.85


def renderer(tokenizer_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)

    def render(system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    return render


def d1_under_load(b: OpenAICompatBackend, prompts: list[str], n: int, concurrency: int,
                  max_tokens: int) -> dict:
    """unique@n per prompt when its n copies are interleaved with random batch-mates."""
    rng = random.Random(0)
    fillers = [b.render_chat(SYSTEM, t) for fam in FAMILIES.values() for g in fam for t in g]
    per_prompt = []
    for text in prompts:
        target = b.render_chat(SYSTEM, text)
        jobs = [("t", target)] * n + [("f", rng.choice(fillers)) for _ in range(n)]
        rng.shuffle(jobs)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outs = list(pool.map(lambda j: (j[0], b.generate(j[1], max_tokens)), jobs))
        targets = [o for kind, o in outs if kind == "t"]
        solo = b.generate(target, max_tokens)
        per_prompt.append({"prompt": text[:60], "unique": unique_at_n(targets),
                           "solo_matches_all": all(t == solo for t in targets)})
    return {"n": n, "concurrency": concurrency,
            "tar_raw": sum(p["unique"] == 1 for p in per_prompt) / len(per_prompt),
            "max_unique": max(p["unique"] for p in per_prompt), "prompts": per_prompt}


def d2_parity(ref: OpenAICompatBackend, alt: OpenAICompatBackend) -> dict:
    rows = []
    for item in generate(15, seed=0):
        r = DecideProfile(ref, abstain_margin=0.0).run(item.op())
        a = DecideProfile(alt, abstain_margin=0.0).run(item.op())
        rows.append({"margin": r.margin, "agree": r.label == a.label, "correct": r.label == item.answer})
    rows.sort(key=lambda x: -x["margin"])
    keep = rows[: round(C0 * len(rows))]
    return {"n": len(rows), "parity": sum(x["agree"] for x in rows) / len(rows),
            "parity_at_C0": sum(x["agree"] for x in keep) / len(keep),
            "accuracy_ref": sum(x["correct"] for x in rows) / len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--alt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "experiments/results/gpu_p0_engine_conformance.json"))
    args = ap.parse_args()
    render = renderer(args.tokenizer or args.model)
    ref = OpenAICompatBackend(args.ref, args.model, render=render)
    t0 = time.time()
    golden = [paras[0] for fam in FAMILIES.values() for paras in fam]
    result = {"ref": ref.lock_fields(), "d1_under_load": d1_under_load(
        ref, golden, args.n, args.concurrency, args.max_tokens)}
    if args.alt:
        alt = OpenAICompatBackend(args.alt, args.model, render=render)
        result["alt"] = alt.lock_fields()
        result["d2_fresh_parity"] = d2_parity(ref, alt)
    result["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "d1_under_load"} | {
        "d1_tar_raw": result["d1_under_load"]["tar_raw"],
        "d1_max_unique": result["d1_under_load"]["max_unique"]}, indent=1))


if __name__ == "__main__":
    main()

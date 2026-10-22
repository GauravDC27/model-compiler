"""Router margin-regularization pilot. Pre-registration: 04_margin_regularization_prereg.md
(amendments 1 and 2)."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dmc import moe_probe as mp  # noqa: E402
from dmc.oracle import generate  # noqa: E402
from dmc.profiles.decide import DecideProfile  # noqa: E402
from dmc.routing import TOKENS, RouteTracer, find_routers, harden_routers  # noqa: E402
from dmc.runtime import HFBackend  # noqa: E402
from prompts_v0 import FAMILIES, decision_ops  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
SYSTEM = "You are a helpful assistant."
SMOKE = bool(os.environ.get("DMC_SMOKE"))  # tiny end-to-end run to catch crashes
LR, STEPS, BATCH, SEQ = 2e-4, (2 if SMOKE else 80), 4, 128
GRID = [(g, lam) for g in (0.05, 0.2) for lam in (1.0, 5.0)][: 1 if SMOKE else None]


def chunks(tok, split: str, n: int, seed: int) -> list[torch.Tensor]:
    path = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "\n".join(t for t in pd.read_parquet(path)["text"] if t.strip())
    ids = tok(text, add_special_tokens=False).input_ids
    rng = random.Random(seed)
    starts = rng.sample(range(0, len(ids) - SEQ - 1), n)
    return [torch.tensor(ids[s:s + SEQ]) for s in starts]


def routers(model):
    return [mod for _, mod, _ in find_routers(model)]


def eval_inputs(b16, held):
    """Held-out chunks + 45 probe prompts (first 3 paraphrase groups of each family)."""
    inputs = [x[None] for x in held]
    for fam in FAMILIES.values():
        for paras in fam[:3]:
            for text in paras:
                inputs.append(torch.tensor([b16._ids(b16.render_chat(SYSTEM, text))]))
    return inputs


def routes(model, ids, threads):
    torch.set_num_threads(threads)
    with torch.no_grad(), harden_routers(model), RouteTracer(model, record_margins=True) as tr:
        model(input_ids=ids)
    return tr.trace(), torch.cat(tr.margins)


def evaluate(b32, b16, held, inputs, ops, oracle_answers) -> dict:
    m32, m16 = b32.model, b16.model
    with torch.no_grad():
        for r32, r16 in zip(routers(m32), routers(m16)):
            r16.weight.copy_(r32.weight.to(torch.bfloat16))
    k = m32.config.num_experts_per_tok
    flips, margins, stress = [], [], []
    for ids in inputs:
        a, ma = routes(m16, ids, 4)
        c, _ = routes(m16, ids, 3)
        s, _ = routes(m32, ids, 4)
        margins.append(ma)
        for key in a:
            if key == TOKENS:
                continue
            x = a[key].sort(-1).values
            flips.append((x != c[key].sort(-1).values).any(-1))
            stress.append((x != s[key].sort(-1).values).any(-1))
    torch.set_num_threads(4)
    ce = []
    usage = None
    for x in held:
        with torch.no_grad(), harden_routers(m32):
            out = m32(input_ids=x[None], labels=x[None], output_router_logits=True)
        ce.append(float(out.loss))
        rl = torch.stack([r.float() for r in out.router_logits])[:, None]  # L,1,T,E
        u = mp.expert_usage(rl, k)
        usage = u if usage is None else usage + u
    p = usage / usage.sum(dim=1, keepdim=True)
    ent = -(p * p.clamp_min(1e-30).log()).sum(dim=1) / math.log(usage.shape[1])
    with harden_routers(m32):
        labels = [DecideProfile(b32, abstain_margin=0.0).run(op).label for op in ops]
    parity_ops = ops[30:60]  # first 30 oracle decisions (secondary metric; bounded for CPU time)
    torch.set_num_threads(4)
    with harden_routers(m16):
        labels16_4 = [DecideProfile(b16, abstain_margin=0.0).run(op).label for op in parity_ops]
    torch.set_num_threads(3)
    with harden_routers(m16):
        labels16_3 = [DecideProfile(b16, abstain_margin=0.0).run(op).label for op in parity_ops]
    torch.set_num_threads(4)
    m = torch.cat(margins).double()
    return {
        "primary_flip_rate_bf16_4v3": float(torch.cat(flips).double().mean()),
        "stress_flip_rate_fp32_vs_bf16": float(torch.cat(stress).double().mean()),
        "margin_below_0.01": float((m < 0.01).double().mean()),
        "heldout_ce": sum(ce) / len(ce),
        "load_entropy_mean": float(ent.mean()),
        "labels": labels,
        "oracle_accuracy": sum(l == a for l, a in zip(labels[30:], oracle_answers)) / len(oracle_answers),
        "oracle_parity_bf16_4v3": sum(x == y for x, y in zip(labels16_4, labels16_3)) / len(labels16_4),
    }


def train(b32, train_chunks, gamma, lam, seed, orig_weights) -> list[dict]:
    m32 = b32.model
    rs = routers(m32)
    with torch.no_grad():
        for r, w in zip(rs, orig_weights):
            r.weight.copy_(w)
    for p in m32.parameters():
        p.requires_grad_(False)
    for r in rs:
        r.weight.requires_grad_(True)
    norms = [w.norm(dim=1, keepdim=True) for w in orig_weights]
    opt = torch.optim.Adam([r.weight for r in rs], lr=LR)
    k = m32.config.num_experts_per_tok
    order = list(range(len(train_chunks)))
    random.Random(seed).shuffle(order)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # training only (pre-registered)
    torch.set_num_threads(4)
    m32.train()
    log = []
    for step in range(STEPS):
        batch = torch.stack([train_chunks[i] for i in order[step * BATCH:(step + 1) * BATCH]])
        out = m32(input_ids=batch, labels=batch, output_router_logits=True)
        rl = torch.stack([x.float() for x in out.router_logits])
        s = rl.topk(k + 1, dim=-1).values
        hinge = torch.relu(gamma - (s[..., k - 1] - s[..., k])).mean()
        loss = out.loss + lam * hinge
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for r, n0 in zip(rs, norms):
                r.weight.mul_(n0 / r.weight.norm(dim=1, keepdim=True))
        if step % 20 == 0 or step == STEPS - 1:
            log.append({"step": step, "ce": round(float(out.loss), 4), "hinge": round(float(hinge), 5)})
    m32.eval()
    for r in rs:
        r.weight.requires_grad_(False)
    torch.use_deterministic_algorithms(True)
    return log


def passes(arm: dict, control: dict, base: dict) -> dict:
    checks = {
        "flip_halved_vs_control": arm["primary_flip_rate_bf16_4v3"] <= 0.5 * control["primary_flip_rate_bf16_4v3"],
        "ce_within_1pct": arm["heldout_ce"] <= 1.01 * base["heldout_ce"],
        "load_entropy_within_5pct": arm["load_entropy_mean"] >= 0.95 * base["load_entropy_mean"],
        "probe_agreement_ge_90": sum(a == b for a, b in zip(arm["labels"][:30], base["labels"][:30])) / 30 >= 0.90,
        "oracle_agreement_ge_95": sum(a == b for a, b in zip(arm["labels"][30:], base["labels"][30:]))
        / len(base["labels"][30:]) >= 0.95,
        "oracle_accuracy_within_2pts": arm["oracle_accuracy"] >= base["oracle_accuracy"] - 0.02,
    }
    checks["PASS"] = all(checks.values())
    return checks


def main() -> None:
    t0 = time.time()
    # R7 is applied explicitly (harden_routers) where the pre-registration requires it.
    b32 = HFBackend(MODEL, dtype="float32", harden=False)
    b16 = HFBackend(MODEL, dtype="bfloat16", harden=False)
    tok = b32.tok
    train_chunks = chunks(tok, "train", STEPS * BATCH, seed=0)
    held = chunks(tok, "validation", 2 if SMOKE else 16, seed=1)
    inputs = eval_inputs(b16, held)[: 4 if SMOKE else None]
    oracle = generate(1 if SMOKE else 5, seed=5)
    ops = decision_ops() + [o.op() for o in oracle]
    answers = [o.answer for o in oracle]
    orig = [r.weight.detach().clone() for r in routers(b32.model)]

    def log(msg):
        print(f"{msg}  {time.time() - t0:.0f}s", flush=True)

    base = evaluate(b32, b16, held, inputs, ops, answers)
    log(f"original {({k: v for k, v in base.items() if k != 'labels'})}")
    results = {"original": base, "arms": {}}
    for seed in (0, 1):
        arms = [(0.0, 0.0)] + (GRID if seed == 0 else [
            (g, lam) for g, lam in GRID if results["arms"].get(f"s0_g{g}_l{lam}", {}).get("checks", {}).get("PASS")
        ])
        for gamma, lam in arms:
            name = f"s{seed}_g{gamma}_l{lam}"
            tlog = train(b32, train_chunks, gamma, lam, seed, orig)
            ev = evaluate(b32, b16, held, inputs, ops, answers)
            results["arms"][name] = {"gamma": gamma, "lambda": lam, "seed": seed, "train_log": tlog, **ev}
            log(f"{name} {({k: v for k, v in ev.items() if k != 'labels'})}")
        control = results["arms"][f"s{seed}_g0.0_l0.0"]
        for name, arm in results["arms"].items():
            if arm["seed"] == seed and arm["lambda"] > 0:
                arm["checks"] = passes(arm, control, base)
    survivors = [
        f"g{g}_l{lam}" for g, lam in GRID
        if results["arms"].get(f"s0_g{g}_l{lam}", {}).get("checks", {}).get("PASS")
        and results["arms"].get(f"s1_g{g}_l{lam}", {}).get("checks", {}).get("PASS")
    ]
    results["survivors"] = survivors
    results["LEVER_SURVIVES"] = bool(survivors)
    results["elapsed_s"] = round(time.time() - t0, 1)
    for arm in results["arms"].values():
        arm["labels"] = None
    base["labels"] = None
    out = ROOT / f"experiments/results/04_margin_regularization{'_smoke' if SMOKE else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "arms"}, indent=1))


if __name__ == "__main__":
    main()

"""Router margin-regularization pilot. Pre-registration: 04_margin_regularization_prereg.md."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dmc import moe_probe as mp  # noqa: E402
from dmc.profiles.decide import DecideProfile  # noqa: E402
from dmc.routing import find_routers  # noqa: E402
from dmc.runtime import HFBackend  # noqa: E402
from prompts_v0 import FAMILIES, decision_ops  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
LAMBDA, GAMMA, LR, STEPS, BATCH, SEQ = 5.0, 0.1, 2e-4, 120, 4, 128
SYSTEM = "You are a helpful assistant."


def chunks(tok, split: str, n: int, seed: int) -> list[torch.Tensor]:
    path = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "\n".join(t for t in pd.read_parquet(path)["text"] if t.strip())
    ids = tok(text, add_special_tokens=False).input_ids
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(0, len(ids) - SEQ - 1), n))
    return [torch.tensor(ids[s:s + SEQ]) for s in starts]


def routers(model):
    return [mod for _, mod, _ in find_routers(model)]


def eval_routing(m32, m16, tok, held: list[torch.Tensor]) -> dict:
    k = m32.config.num_experts_per_tok
    flips, margins, ce = [], [], []
    inputs = [x[None] for x in held]
    for fam in FAMILIES.values():
        for paras in fam:
            for text in paras:
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
                r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs.append(torch.tensor([tok(r, add_special_tokens=False).input_ids]))
    for i, ids in enumerate(inputs):
        mask = torch.ones_like(ids)
        a = mp.router_logits(m32, ids, mask)[:, 0]
        b = mp.router_logits(m16, ids, mask)[:, 0]
        flips.append(mp.flips(a, b, k).flatten())
        margins.append(mp.margins(a, k).flatten())
        if i < len(held):
            with torch.no_grad():
                out = m32(input_ids=ids, labels=ids)
            ce.append(float(out.loss))
    f = torch.cat(flips).double()
    m = torch.cat(margins).double()
    return {
        "bf16_flip_rate": float(f.mean()),
        "margin_below_0.01": float((m < 0.01).double().mean()),
        "margin_median": float(m.median()),
        "heldout_ce": sum(ce) / len(ce),
    }


def sync_bf16(m32, m16) -> None:
    with torch.no_grad():
        for r32, r16 in zip(routers(m32), routers(m16)):
            r16.weight.copy_(r32.weight.to(torch.bfloat16))


def decisions(backend_32, backend_16) -> dict[str, list]:
    p32 = DecideProfile(backend_32, abstain_margin=0.0)
    p16 = DecideProfile(backend_16, abstain_margin=0.0)
    ops = decision_ops()
    return {"fp32": [p32.run(op).label for op in ops], "bf16": [p16.run(op).label for op in ops]}


def main() -> None:
    t0 = time.time()
    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL)
    b32 = HFBackend(MODEL, dtype="float32")
    b16 = HFBackend(MODEL, dtype="bfloat16")
    m32, m16 = b32.model, b16.model
    train = chunks(tok, "train", STEPS * BATCH, seed=0)
    held = chunks(tok, "validation", 16, seed=1)

    before = eval_routing(m32, m16, tok, held)
    dec_before = decisions(b32, b16)
    print("before", before, f"{time.time() - t0:.0f}s", flush=True)

    for p in m32.parameters():
        p.requires_grad_(False)
    rs = routers(m32)
    for r in rs:
        r.weight.requires_grad_(True)
    norms = [r.weight.detach().norm(dim=1, keepdim=True).clone() for r in rs]
    opt = torch.optim.Adam([r.weight for r in rs], lr=LR)
    k = m32.config.num_experts_per_tok
    torch.use_deterministic_algorithms(False)  # training only; evaluation runs deterministic
    m32.train()
    log = []
    for step in range(STEPS):
        batch = torch.stack(train[step * BATCH:(step + 1) * BATCH])
        out = m32(input_ids=batch, labels=batch, output_router_logits=True)
        rl = torch.stack([x.float() for x in out.router_logits])  # L, tokens, E
        s = rl.topk(k + 1, dim=-1).values
        hinge = torch.relu(GAMMA - (s[..., k - 1] - s[..., k])).mean()
        loss = out.loss + LAMBDA * hinge
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for r, n0 in zip(rs, norms):
                r.weight.mul_(n0 / r.weight.norm(dim=1, keepdim=True))
        if step % 10 == 0 or step == STEPS - 1:
            log.append({"step": step, "ce": float(out.loss), "hinge": float(hinge)})
            print(log[-1], f"{time.time() - t0:.0f}s", flush=True)
    m32.eval()
    torch.use_deterministic_algorithms(True)
    sync_bf16(m32, m16)
    after = eval_routing(m32, m16, tok, held)
    dec_after = decisions(b32, b16)

    def parity(d):
        return sum(a == b for a, b in zip(d["fp32"], d["bf16"])) / len(d["fp32"])

    flip_ratio = before["bf16_flip_rate"] / max(after["bf16_flip_rate"], 1e-12)
    mass_ratio = before["margin_below_0.01"] / max(after["margin_below_0.01"], 1e-12)
    ce_rise = after["heldout_ce"] / before["heldout_ce"] - 1
    passed = (
        (flip_ratio >= 2 or (mass_ratio >= 2 and after["bf16_flip_rate"] <= before["bf16_flip_rate"]))
        and ce_rise <= 0.01
    )
    result = {
        "model": MODEL,
        "hyper": {"lambda": LAMBDA, "gamma": GAMMA, "lr": LR, "steps": STEPS, "batch": BATCH, "seq": SEQ},
        "before": before,
        "after": after,
        "flip_rate_reduction_x": flip_ratio,
        "margin_mass_reduction_x": mass_ratio,
        "heldout_ce_relative_rise": ce_rise,
        "decision_agreement_with_original_fp32": sum(
            a == b for a, b in zip(dec_before["fp32"], dec_after["fp32"])) / len(dec_before["fp32"]),
        "decision_parity_fp32_vs_bf16_before": parity(dec_before),
        "decision_parity_fp32_vs_bf16_after": parity(dec_after),
        "train_log": log,
        "PASS": passed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = ROOT / "experiments/results/04_margin_regularization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

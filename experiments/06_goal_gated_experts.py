"""Goal-gated expert expansion pilot. Pre-registration: 06_goal_gated_experts_prereg.md."""

from __future__ import annotations

import json
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
from dmc.expansion import expand, set_goal, trainable_parameters  # noqa: E402
from dmc.oracle import generate  # noqa: E402
from dmc.profiles.decide import SYSTEM as DECIDE_SYSTEM  # noqa: E402
from dmc.profiles.decide import DecideProfile, label_orderings, render_user  # noqa: E402
from dmc.runtime import HFBackend  # noqa: E402
from prompts_v0 import FAMILIES  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
GOAL = "decide.oracle"
SMOKE = bool(os.environ.get("DMC_SMOKE"))
STEPS, BATCH, LR = (3 if SMOKE else 200), 8, 1e-3
OUT = ROOT / f"experiments/results/06_goal_gated_experts{'_smoke' if SMOKE else ''}.json"


def wikitext(tok, n: int) -> list[torch.Tensor]:
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
                           repo_type="dataset")
    ids = tok("\n".join(t for t in pd.read_parquet(path)["text"] if t.strip()), add_special_tokens=False).input_ids
    rng = random.Random(1)
    return [torch.tensor(ids[s:s + 128]) for s in rng.sample(range(len(ids) - 129), n)]


def examples(b: HFBackend, items) -> list[tuple[list[int], int]]:
    out = []
    for it in items:
        op = it.op()
        prompt = b.render_chat(DECIDE_SYSTEM, render_user(op, label_orderings(op.labels)[0]))
        base = b._ids(prompt)
        target = b._ids(prompt + it.answer)[len(base)]
        out.append((base, target))
    return out


def accuracy(b: HFBackend, items) -> list[bool]:
    prof = DecideProfile(b, abstain_margin=0.0)
    return [prof.run(it.op()).label == it.answer for it in items]


def boot_diff(a: list[bool], b: list[bool], groups: list[str], n: int = 1000) -> tuple[float, float]:
    rng = random.Random(3)
    keys = sorted(set(groups))
    by = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
    diffs = []
    for _ in range(n):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in by[k]]
        diffs.append(sum(b[i] for i in idx) / len(idx) - sum(a[i] for i in idx) / len(idx))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def new_expert_shares(model, wrapped, ids: torch.Tensor) -> float:
    """Mean over expanded layers of the new-expert share, using each layer's real input."""
    caps = []
    hooks = [w.register_forward_pre_hook(lambda m, a: caps.append((m, a[0].detach()))) for w in wrapped]
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    return sum(m.new_expert_share(x) for m, x in caps) / len(caps)


def mean_ce(model, chunks) -> float:
    with torch.no_grad():
        return sum(float(model(input_ids=c[None], labels=c[None]).loss) for c in chunks) / len(chunks)


def hottest_experts(b: HFBackend, exs, layers: list[int]) -> dict[int, list[int]]:
    k = b.model.config.num_experts_per_tok
    usage = None
    for ids, _ in exs[:200]:
        rl = mp.router_logits(b.model, torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long))
        u = mp.expert_usage(rl, k)
        usage = u if usage is None else usage + u
    return {i: [int(usage[i].argmax())] for i in layers}


def train(b: HFBackend, wrapped, exs) -> list[dict]:
    model = b.model
    for p in model.parameters():
        p.requires_grad_(False)
    params = trainable_parameters(wrapped)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    rng = random.Random(0)
    order = list(range(len(exs)))
    rng.shuffle(order)
    torch.use_deterministic_algorithms(False)
    set_goal(wrapped, GOAL)
    log = []
    for step in range(STEPS):
        batch = [exs[order[(step * BATCH + j) % len(order)]] for j in range(BATCH)]
        loss = 0.0
        for ids, target in batch:  # one sequence at a time: no padding, fixed shapes
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1].float()
            loss = loss + torch.nn.functional.cross_entropy(logits[None], torch.tensor([target])) / BATCH
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == STEPS - 1:
            log.append({"step": step, "loss": round(float(loss), 4)})
            print(log[-1], flush=True)
    torch.use_deterministic_algorithms(True)
    for p in params:
        p.requires_grad_(False)
    return log


def main() -> None:
    t0 = time.time()
    torch.manual_seed(0)
    train_items = generate(2 if SMOKE else 150, seed=100)
    test_items = generate(1 if SMOKE else 20, seed=0)
    groups = [f"{it.family}:{it.group}" for it in test_items]
    probe_texts = [t for fam in FAMILIES.values() for g in fam for t in g][: 3 if SMOKE else 20]
    result: dict = {"model": MODEL, "goal": GOAL, "steps": STEPS, "batch": BATCH, "lr": LR,
                    "n_train": len(train_items), "n_test": len(test_items)}

    b = HFBackend(MODEL, dtype="float32")
    tok = b.tok
    wiki = wikitext(tok, 2 if SMOKE else 16)
    probe_ids = [torch.tensor([b._ids(b.render_chat("You are a helpful assistant.", t))]) for t in probe_texts]
    with torch.no_grad():
        probe_ref = [b.model(input_ids=x).logits for x in probe_ids]
    base_acc = accuracy(b, test_items)
    base_ce = mean_ce(b.model, wiki)
    result["base"] = {"oracle_accuracy": sum(base_acc) / len(base_acc), "wikitext_ce": base_ce}
    print("base", result["base"], f"{time.time() - t0:.0f}s", flush=True)
    exs = examples(b, train_items)
    layers = list(range(b.model.config.num_hidden_layers))
    warm = hottest_experts(b, exs, layers)
    originals = {i: b.model.model.layers[i].block_sparse_moe for i in layers}

    for variant in ("routed", "additive"):
        for i in layers:  # restore the base blocks before each arm
            b.model.model.layers[i].block_sparse_moe = originals[i]
        wrapped = expand(b.model, layers, 1, [GOAL], variant, warm if variant == "routed" else None)
        n_params = sum(p.numel() for p in trainable_parameters(wrapped))
        tlog = train(b, wrapped, exs)
        set_goal(wrapped, None)
        with torch.no_grad():
            identical = all(torch.equal(b.model(input_ids=x).logits, r) for x, r in zip(probe_ids, probe_ref))
        t = time.time()
        with torch.no_grad():
            for x in probe_ids:
                b.model(input_ids=x)
        t_off = (time.time() - t) / len(probe_ids)
        set_goal(wrapped, GOAL)
        t = time.time()
        with torch.no_grad():
            for x in probe_ids:
                b.model(input_ids=x)
        t_on = (time.time() - t) / len(probe_ids)
        acc = accuracy(b, test_items)
        ce_on = mean_ce(b.model, wiki)
        test_exs = examples(b, test_items[:30])
        share_test = sum(new_expert_shares(b.model, wrapped, torch.tensor([ids])) for ids, _ in test_exs) / len(test_exs)
        share_wiki = sum(new_expert_shares(b.model, wrapped, c[None]) for c in wiki) / len(wiki)
        lo, hi = boot_diff(base_acc, acc, groups)
        gain = sum(acc) / len(acc) - sum(base_acc) / len(base_acc)
        result[variant] = {
            "trainable_params": n_params, "train_log": tlog,
            "oracle_accuracy": sum(acc) / len(acc), "gain": gain, "gain_ci95": (lo, hi),
            "goal_off_bit_identical": identical,
            "wikitext_ce_goal_on": ce_on, "wikitext_ce_rise": ce_on / base_ce - 1,
            "new_expert_share_test": share_test,
            "new_expert_share_wikitext": share_wiki,
            "cpu_forward_s_goal_off": round(t_off, 3), "cpu_forward_s_goal_on": round(t_on, 3),
            "SUCCESS": gain >= 0.10 and lo > 0 and identical and ce_on / base_ce - 1 <= 0.05,
        }
        print(variant, {k: v for k, v in result[variant].items() if k != "train_log"},
              f"{time.time() - t0:.0f}s", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=1))
    result["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()

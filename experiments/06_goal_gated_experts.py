"""Goal-gated expert expansion pilot.
Pre-registration: 06_goal_gated_experts_prereg.md (with amendment 1)."""

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

from dmc import expert_lora  # noqa: E402
from dmc import moe_probe as mp  # noqa: E402
from dmc.expansion import expand, set_goal, trainable_parameters  # noqa: E402
from dmc.oracle import generate  # noqa: E402
from dmc.profiles.decide import SYSTEM as DECIDE_SYSTEM  # noqa: E402
from dmc.profiles.decide import DecideProfile, label_orderings, render_user  # noqa: E402
from dmc.runtime import HFBackend  # noqa: E402
from prompts_v0 import FAMILIES  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
GOAL = "numeric"
GOAL_FAMILIES = {"compare", "even", "sum", "expense"}  # trained goal
HELD_OUT_FAMILIES = {"refund", "letter"}  # wrong-goal test
SMOKE = bool(os.environ.get("DMC_SMOKE"))
STEPS, BATCH, LR = (2 if SMOKE else 150), 8, 1e-3
NOISE, BIAS0 = 0.05, 0.1
ARMS = ("routed", "lora") if SMOKE else ("routed", "additive", "router_only", "random_parent", "lora")
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
        out.append((base, b._ids(prompt + it.answer)[len(base)]))
    return out


def correct(b: HFBackend, items) -> list[bool]:
    prof = DecideProfile(b, abstain_margin=0.0)
    return [prof.run(it.op()).label == it.answer for it in items]


def boot_diff(a: list[bool], c: list[bool], groups: list[str], n: int = 1000,
              level: float = 0.95) -> tuple[float, float]:
    """Group-level bootstrap CI of mean(c) - mean(a) (paired on items)."""
    rng = random.Random(3)
    keys = sorted(set(groups))
    by = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
    diffs = []
    for _ in range(n):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in by[k]]
        diffs.append(sum(c[i] for i in idx) / len(idx) - sum(a[i] for i in idx) / len(idx))
    diffs.sort()
    tail = (1 - level) / 2
    return diffs[int(tail * n)], diffs[int((1 - tail) * n) - 1]


def new_expert_shares(model, wrapped, ids: torch.Tensor) -> float:
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


def usage_counts(b: HFBackend, exs) -> torch.Tensor:
    k = b.model.config.num_experts_per_tok
    usage = None
    for ids, _ in exs[:200]:
        rl = mp.router_logits(b.model, torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long))
        u = mp.expert_usage(rl, k)
        usage = u if usage is None else usage + u
    return usage  # (layers, experts)


def hot_set(row: torch.Tensor, mass: float = 0.8) -> list[int]:
    order = torch.argsort(row, descending=True, stable=True).tolist()
    total, acc, out = float(row.sum()), 0.0, []
    for e in order:
        out.append(e)
        acc += float(row[e])
        if acc >= mass * total:
            break
    return sorted(out)


def train(model, params, exs, set_goal_fn) -> list[dict]:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    order = list(range(len(exs)))
    random.Random(0).shuffle(order)
    torch.use_deterministic_algorithms(False)
    set_goal_fn(GOAL)
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
    gen_train = generate(2 if SMOKE else 150, seed=100)
    gen_test = generate(1 if SMOKE else 20, seed=0)
    train_items = [it for it in gen_train if it.family in GOAL_FAMILIES]
    goal_test = [it for it in gen_test if it.family in GOAL_FAMILIES]
    wrong_test = [it for it in gen_test if it.family in HELD_OUT_FAMILIES]
    g_goal = [f"{it.family}:{it.group}" for it in goal_test]
    g_wrong = [f"{it.family}:{it.group}" for it in wrong_test]
    probe_texts = [t for fam in FAMILIES.values() for g in fam for t in g][: 3 if SMOKE else 20]
    result: dict = {"model": MODEL, "goal": GOAL, "goal_families": sorted(GOAL_FAMILIES),
                    "held_out_families": sorted(HELD_OUT_FAMILIES), "steps": STEPS, "batch": BATCH,
                    "lr": LR, "noise": NOISE, "bias_init": BIAS0, "n_train": len(train_items),
                    "n_goal_test": len(goal_test), "n_wrong_test": len(wrong_test), "arms": {}}

    def save():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=1, default=str))

    b = HFBackend(MODEL, dtype="float32")
    model = b.model
    wiki = wikitext(b.tok, 2 if SMOKE else 16)
    probe_ids = [torch.tensor([b._ids(b.render_chat("You are a helpful assistant.", t))]) for t in probe_texts]
    with torch.no_grad():
        probe_ref = [model(input_ids=x).logits for x in probe_ids]
    base_goal = correct(b, goal_test)
    base_wrong = correct(b, wrong_test)
    base_ce = mean_ce(model, wiki)
    result["base"] = {"goal_accuracy": sum(base_goal) / len(base_goal),
                      "held_out_accuracy": sum(base_wrong) / len(base_wrong), "wikitext_ce": base_ce}
    print("base", result["base"], f"{time.time() - t0:.0f}s", flush=True)
    save()

    exs = examples(b, train_items)
    layers = list(range(model.config.num_hidden_layers))
    usage = usage_counts(b, exs)
    hottest = {i: [int(torch.argmax(usage[i]))] for i in layers}
    rng = random.Random(5)
    n_exp = model.config.num_local_experts
    random_parent = {i: [rng.randrange(n_exp)] for i in layers}
    hot = {i: hot_set(usage[i]) for i in layers}
    originals = {i: model.model.layers[i].block_sparse_moe for i in layers}
    lora_rank = None
    gated_params = None

    for arm in ARMS:
        for i in layers:  # restore the base blocks before each arm
            model.model.layers[i].block_sparse_moe = originals[i]
        gen = torch.Generator().manual_seed(11)
        wrapped, mods = [], []
        if arm == "lora":
            per = sum(len(hot[i]) for i in layers) * expert_lora.params_per_expert(originals[0].experts, 1)
            lora_rank = max(1, round(gated_params / per))
            mods = expert_lora.attach(model, hot, rank=lora_rank, alpha=2.0 * lora_rank)
            params = [p for m in mods for p in (m.a_gu, m.b_gu, m.a_d, m.b_d)]

            def goal_fn(g):
                return None
        else:
            variant = "additive" if arm == "additive" else "routed"
            if variant == "additive":
                wrapped = expand(model, layers, 1, [GOAL], "additive")
            else:
                parents = random_parent if arm == "random_parent" else hottest
                wrapped = expand(model, layers, 1, [GOAL], "routed", parents,
                                 noise=0.0 if arm == "router_only" else NOISE,
                                 router_init="parent", bias_init=BIAS0, generator=gen)
            if arm == "router_only":
                params = [p for w in wrapped for p in (w.router_new, w.goal_bias)]
            else:
                params = trainable_parameters(wrapped)

            def goal_fn(g, wrapped=wrapped):
                set_goal(wrapped, g)
        n_params = sum(p.numel() for p in params)
        if arm == "routed":
            gated_params = n_params
        tlog = train(model, params, exs, goal_fn)

        rec: dict = {"trainable_params": n_params, "train_log": tlog}
        if arm == "lora":
            rec["lora_rank"] = lora_rank
            rec["hot_experts_per_layer_mean"] = sum(len(hot[i]) for i in layers) / len(layers)
            rec["goal_off_bit_identical"] = None  # ordinary fine-tuning has no goal switch
        else:
            goal_fn(None)
            with torch.no_grad():
                rec["goal_off_bit_identical"] = all(
                    torch.equal(model(input_ids=x).logits, r) for x, r in zip(probe_ids, probe_ref))
            goal_fn(GOAL)
        acc_goal = correct(b, goal_test)
        acc_wrong = correct(b, wrong_test)  # goal on (wrong goal) for gated arms; adapter on for LoRA
        ce_on = mean_ce(model, wiki)
        rec.update({
            "goal_accuracy": sum(acc_goal) / len(acc_goal),
            "goal_gain_vs_base": sum(acc_goal) / len(acc_goal) - result["base"]["goal_accuracy"],
            "goal_gain_vs_base_ci95": boot_diff(base_goal, acc_goal, g_goal),
            "held_out_accuracy_goal_on": sum(acc_wrong) / len(acc_wrong),
            "held_out_change_vs_base": sum(acc_wrong) / len(acc_wrong) - result["base"]["held_out_accuracy"],
            "held_out_change_ci95": boot_diff(base_wrong, acc_wrong, g_wrong),
            "wikitext_ce_on": ce_on, "wikitext_ce_rise": ce_on / base_ce - 1,
            "_goal_correct": acc_goal,
        })
        if wrapped:
            test_exs = examples(b, goal_test[:30])
            rec["new_expert_share_goal_test"] = sum(
                new_expert_shares(model, wrapped, torch.tensor([ids])) for ids, _ in test_exs) / len(test_exs)
            rec["new_expert_share_wikitext"] = sum(
                new_expert_shares(model, wrapped, c[None]) for c in wiki) / len(wiki)
            goal_fn(None)
        if mods:
            expert_lora.detach(model, mods, layers)
        result["arms"][arm] = rec
        print(arm, {k: v for k, v in rec.items() if k not in ("train_log", "_goal_correct")},
              f"{time.time() - t0:.0f}s", flush=True)
        save()

    # Pre-registered comparisons: each new-capacity arm vs the matched LoRA arm.
    if "lora" in result["arms"]:
        lora_correct = result["arms"]["lora"]["_goal_correct"]
        # Amendment 2: two primary comparisons, so success uses Bonferroni-adjusted 97.5% CIs;
        # the controls get the same comparison so the interpretation rule can be applied.
        for arm in (a for a in ("routed", "additive", "router_only", "random_parent") if a in result["arms"]):
            rec = result["arms"][arm]
            rec["gain_vs_lora"] = rec["goal_accuracy"] - result["arms"]["lora"]["goal_accuracy"]
            rec["gain_vs_lora_ci95"] = boot_diff(lora_correct, rec["_goal_correct"], g_goal)
            lo, hi = boot_diff(lora_correct, rec["_goal_correct"], g_goal, level=0.975)
            rec["gain_vs_lora_ci975"] = (lo, hi)
            rec["CLEARS_LORA_BAR"] = (lo > 0 and rec["goal_off_bit_identical"] is True
                                      and rec["wikitext_ce_rise"] <= 0.05)
            if arm in ("routed", "additive"):
                rec["SUCCESS"] = rec["CLEARS_LORA_BAR"]
    for rec in result["arms"].values():
        rec.pop("_goal_correct", None)
    result["elapsed_s"] = round(time.time() - t0, 1)
    save()
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=1, default=str))


if __name__ == "__main__":
    main()

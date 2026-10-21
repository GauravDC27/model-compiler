"""Train the hub's goal-family capacity (plan §I.2.3, items i and ii; stages T0 + T2 on CPU).

* ``policy`` family (refund-claim reasons) → goal-gated new experts (item ii): one routed expert
  per layer, warm-started from that layer's most-selected expert on policy prompts, router row
  from the parent plus noise on the copy only, goal bias +0.1; only the new experts, their rows
  and the bias train.
* ``support`` family (tool selection) → hot-expert LoRA (item i): rank 4 on each layer's hot
  experts (80% of selection mass on support prompts, the A3B rule); the router stays frozen.

Both are trained on training-template phrasings only and evaluated on a held-out mixture. Two
identity checks bracket training: with no family set, hub logits are bit-identical to the
untouched base model. Outputs go to ``artifacts/gcam/capacity/`` with a manifest of metrics.

    python -m gcam.train.capacity
"""

from __future__ import annotations

import json
import os
import random
import time

import torch

from dmc import moe_probe as mp
from dmc.canonical import content_hash

from ..domain import build_world, claims, tool_requests
from ..executors.llm import CLAIM_OPTIONS, SYSTEMS, TOOL_OPTIONS, LLMExecutor, claim_user, tool_user
from ..models import HUB_ID, LETTERS, LM, Capacity, artifacts_dir

SMOKE = bool(os.environ.get("GCAM_SMOKE"))
STEPS, BATCH = (2 if SMOKE else 160), 8
LR = {"policy": 1e-3, "support": 3e-4}  # new experts train from a copy; LoRA from zero B needs a gentler rate
SPECS = {
    "policy": {"op": "decide.claim_reason", "labels": tuple(sorted(CLAIM_OPTIONS)), "user": claim_user},
    "support": {"op": "tool.select", "labels": tuple(sorted(TOOL_OPTIONS)), "user": tool_user},
}


def _data(world):
    oids = list(world.orders.order_id[:400])
    return {
        "policy": {
            "train": [(c.text, c.reason) for c in claims(480, split="train", seed=1)],
            "eval": [(c.text, c.reason) for c in claims(60, split="train", seed=9) + claims(60, split="test", seed=9)],
        },
        "support": {
            "train": [(r.text, r.tool) for r in tool_requests(480, split="train", seed=1, order_ids=oids)],
            "eval": [(r.text, r.tool) for r in tool_requests(60, split="train", seed=9, order_ids=oids)
                     + tool_requests(60, split="test", seed=9, order_ids=oids)],
        },
    }


def examples(lm: LM, family: str, items) -> list[tuple[list[int], int]]:
    spec = SPECS[family]
    canon = spec["labels"]
    out = []
    for i, (text, label) in enumerate(items):
        order = canon if i % 2 == 0 else tuple(reversed(canon))  # both orderings, as at inference
        prompt = lm.render(SYSTEMS[spec["op"]], spec["user"]({"text": text}, order))
        ids = lm.backend._ids(prompt)
        target = lm.backend._ids(prompt + LETTERS[order.index(label)])[len(ids)]
        out.append((ids, target))
    return out


def accuracy(ex: LLMExecutor, family: str, items) -> float:
    op = SPECS[family]["op"]
    return sum(ex.run(op, {"text": t}, forced=True).value == y for t, y in items) / len(items)


def usage(lm: LM, exs) -> torch.Tensor:
    k = lm.model.config.num_experts_per_tok
    total = None
    for ids, _ in exs[:160]:
        rl = mp.router_logits(lm.model, torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long))
        u = mp.expert_usage(rl, k)
        total = u if total is None else total + u
    return total


def hot_set(row: torch.Tensor, mass: float = 0.8) -> list[int]:
    order = torch.argsort(row, descending=True, stable=True).tolist()
    total, acc, out = float(row.sum()), 0.0, []
    for e in order:
        out.append(e)
        acc += float(row[e])
        if acc >= mass * total:
            break
    return sorted(out)


def train(lm: LM, params, exs, family: str, log_prefix: str) -> list[dict]:
    for p in lm.model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR[family], weight_decay=0.0)
    order = list(range(len(exs)))
    random.Random(0).shuffle(order)
    torch.use_deterministic_algorithms(False)
    log = []
    with lm.family(family):
        for step in range(STEPS):
            batch = [exs[order[(step * BATCH + j) % len(order)]] for j in range(BATCH)]
            loss = 0.0
            for ids, target in batch:  # one sequence at a time: no padding, fixed shapes
                logits = lm.model(input_ids=torch.tensor([ids])).logits[0, -1].float()
                loss = loss + torch.nn.functional.cross_entropy(logits[None], torch.tensor([target])) / BATCH
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == STEPS - 1:
                log.append({"step": step, "loss": round(float(loss), 4)})
                print(log_prefix, log[-1], flush=True)
    torch.use_deterministic_algorithms(True)
    for p in params:
        p.requires_grad_(False)
    return log


def identity_ok(lm: LM, probes, reference) -> bool:
    with lm.family(None), torch.no_grad():
        return all(torch.equal(lm.model(input_ids=x).logits, ref) for x, ref in zip(probes, reference))


def main() -> None:
    t0 = time.time()
    torch.manual_seed(0)
    world = build_world()
    data = _data(world)
    if SMOKE:
        data = {f: {"train": d["train"][:16], "eval": d["eval"][:6]} for f, d in data.items()}
    lm = LM(HUB_ID, "hub")
    base_ex = LLMExecutor("hub", lm)
    probes = [torch.tensor([lm.backend._ids(lm.render("You are a helpful assistant.", t))]) for t in
              ("Summarize the refund policy.", "What is 17 times 3?", "Write a haiku about parcels.")]
    with torch.no_grad():
        reference = [lm.model(input_ids=x).logits for x in probes]
    manifest: dict = {"model": HUB_ID, "steps": STEPS, "batch": BATCH, "lr": LR, "families": {}}
    manifest["base"] = {fam: {"accuracy": accuracy(base_ex, fam, data[fam]["eval"])} for fam in SPECS}
    print("base", manifest["base"], f"{time.time() - t0:.0f}s", flush=True)

    cap = Capacity(lm)
    lm.capacity = cap
    exs = {fam: examples(lm, fam, data[fam]["train"]) for fam in SPECS}
    layers = list(range(lm.model.config.num_hidden_layers))
    use = {fam: usage(lm, exs[fam]) for fam in SPECS}

    # item (ii): goal-gated experts for the policy family
    parents = {i: [int(torch.argmax(use["policy"][i]))] for i in layers}
    wrapped = cap.add_goal_experts("policy", parents)
    from dmc.expansion import trainable_parameters
    params = trainable_parameters(wrapped)
    n_ii = sum(p.numel() for p in params)
    log_ii = train(lm, params, exs["policy"], "policy", "[policy]")
    # item (i): hot-expert LoRA for the support family
    hot = {i: hot_set(use["support"][i]) for i in layers}
    mods = cap.add_lora("support", hot, rank=4, alpha=8.0)
    params = [p for m in mods for p in (m.a_gu, m.b_gu, m.a_d, m.b_d)]
    n_i = sum(p.numel() for p in params)
    log_i = train(lm, params, exs["support"], "support", "[support]")

    fam_ex = LLMExecutor("hub", lm, family={"decide.claim_reason": "policy", "tool.select": "support"})
    cross_ex = LLMExecutor("hub", lm, family={"decide.claim_reason": "support", "tool.select": "policy"})
    out_dir = artifacts_dir() / ("capacity_smoke" if SMOKE else "capacity")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fam, kind, n, log, extra in (("policy", "goal_gated_experts", n_ii, log_ii, {"parents": parents}),
                                      ("support", "hot_expert_lora", n_i, log_i,
                                       {"hot_sizes": {i: len(v) for i, v in hot.items()}, "hot": hot, "rank": 4})):
        state = cap.state(fam)
        torch.save(state, out_dir / f"{fam}.pt")
        manifest["families"][fam] = {
            "kind": kind, "trainable_parameters": n, "train_log": log,
            "accuracy_with_family": accuracy(fam_ex, fam, data[fam]["eval"]),
            "accuracy_with_other_family": accuracy(cross_ex, fam, data[fam]["eval"]),
            "state_hash": content_hash({k: v.flatten()[:64].tolist() for k, v in sorted(state.items())}),
            **extra,
        }
        print(fam, {k: v for k, v in manifest["families"][fam].items() if k.startswith("acc")}, flush=True)
    manifest["goal_off_bit_identical"] = identity_ok(lm, probes, reference)
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != "families"}, indent=1), flush=True)


if __name__ == "__main__":
    main()

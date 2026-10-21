"""Train the hub's goal-family capacity (plan §I.2.3, items i and ii; stages T0 + T2 on CPU).

* ``policy`` family (refund-claim reasons) → goal-gated new experts (item ii): one additive
  expert in each of the hub's last 8 layers, warm-started from that layer's most-selected expert on policy prompts with
  its down-projection zeroed (step 0 is the base model), sigmoid gate from a zero router row and
  a zero goal bias; only the new experts, their rows and the bias train
  (``GCAM_VARIANT=routed`` selects the routed variant).
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
# ~3 s per forward + backward on this 4-core CPU. Policy: 160 sequences; support: 360.
STEPS = {"policy": 2, "support": 2} if SMOKE else {"policy": 40, "support": 90}
BATCH = 4
LR = {"policy": 1e-4, "support": 3e-4}
GOAL_LAYERS = range(16, 24)  # the hub's last 8 of 24 layers
VARIANT = os.environ.get("GCAM_VARIANT", "additive")
ONLY = os.environ.get("GCAM_ONLY")  # "policy": retrain only the goal experts, keep the support LoRA
SPECS = {
    "policy": {"op": "decide.claim_reason", "labels": tuple(sorted(CLAIM_OPTIONS)), "user": claim_user},
    "support": {"op": "tool.select", "labels": tuple(sorted(TOOL_OPTIONS)), "user": tool_user},
}


def _data(world):
    oids = list(world.orders.order_id[:400])
    return {
        "policy": {
            "train": [(c.text, c.reason) for c in claims(480, split="train", seed=1)],
            "eval": [(c.text, c.reason) for c in claims(24, split="train", seed=9) + claims(24, split="test", seed=9)],
        },
        "support": {
            "train": [(r.text, r.tool) for r in tool_requests(480, split="train", seed=1, order_ids=oids)],
            "eval": [(r.text, r.tool) for r in tool_requests(24, split="train", seed=9, order_ids=oids)
                     + tool_requests(24, split="test", seed=9, order_ids=oids)],
        },
    }


def examples(lm: LM, family: str, items) -> list[tuple[list[int], int]]:
    """Every item in both option orderings, as at inference.

    One ordering per item is not enough: the generators cycle labels (item i has label i mod 4 or
    i mod 6), so alternating orderings by item would pin each label to one ordering and make the
    answer letter predictable without reading the text."""
    spec = SPECS[family]
    canon = spec["labels"]
    out = []
    for text, label in items:
        for order in (canon, tuple(reversed(canon))):
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
    for ids, _ in exs[:64]:
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
        for step in range(STEPS[family]):
            batch = [exs[order[(step * BATCH + j) % len(order)]] for j in range(BATCH)]
            loss = 0.0
            for ids, target in batch:  # one sequence at a time: no padding, fixed shapes
                logits = lm.model(input_ids=torch.tensor([ids])).logits[0, -1].float()
                loss = loss + torch.nn.functional.cross_entropy(logits[None], torch.tensor([target])) / BATCH
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 20 == 0 or step == STEPS[family] - 1:
                log.append({"step": step, "loss": round(float(loss), 4)})
                print(log_prefix, log[-1], flush=True)
    torch.use_deterministic_algorithms(True)
    for p in params:
        p.requires_grad_(False)
    return log


def new_share(lm: LM, wrapped, exs) -> dict:
    """New-expert traffic share with the goal on: over all tokens and at the answer token."""
    caps = []
    hooks = [w.register_forward_pre_hook(lambda m, a: caps.append((m, a[0].detach()))) for w in wrapped]
    with torch.no_grad(), lm.family("policy"):
        for ids, _ in exs:
            lm.model(input_ids=torch.tensor([ids]))
        out = {"all_tokens": round(sum(m.new_expert_share(h) for m, h in caps) / len(caps), 3),
               "answer_token": round(sum(m.new_expert_share(h[:, -1:]) for m, h in caps) / len(caps), 3)}
    for h in hooks:
        h.remove()
    return out


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

    # item (ii): goal-gated experts for the policy family (additive variant; see Capacity.add_goal_experts)
    parents = {i: [int(torch.argmax(use["policy"][i]))] for i in GOAL_LAYERS}
    wrapped = cap.add_goal_experts("policy", parents, variant=VARIANT)
    from dmc.expansion import trainable_parameters
    params = trainable_parameters(wrapped)
    n_ii = sum(p.numel() for p in params)
    hot = {i: hot_set(use["support"][i]) for i in layers}
    if ONLY == "policy":  # retrain the goal experts only; keep the trained support LoRA
        mods = cap.add_lora("support", hot, rank=4, alpha=8.0)
        cap.load("support", torch.load(artifacts_dir() / "capacity" / "support.pt"))
        old = json.loads((artifacts_dir() / "capacity" / "manifest.json").read_text())
    log_ii = train(lm, params, exs["policy"], "policy", "[policy]")
    share = new_share(lm, wrapped, exs["policy"][:6])
    # item (i): hot-expert LoRA for the support family
    if ONLY != "policy":
        mods = cap.add_lora("support", hot, rank=4, alpha=8.0)
    params = [p for m in mods for p in (m.a_gu, m.b_gu, m.a_d, m.b_d)]
    n_i = sum(p.numel() for p in params)
    log_i = old["families"]["support"]["train_log"] if ONLY == "policy" else train(lm, params, exs["support"], "support",
                                                                                  "[support]")

    fam_ex = LLMExecutor("hub", lm, family={"decide.claim_reason": "policy", "tool.select": "support"})
    cross_ex = LLMExecutor("hub", lm, family={"decide.claim_reason": "support", "tool.select": "policy"})
    out_dir = artifacts_dir() / ("capacity_smoke" if SMOKE else "capacity")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fam, kind, n, log, extra in (("policy", "goal_gated_experts", n_ii, log_ii,
                                      {"parents": parents, "variant": VARIANT, "new_expert_share": share}),
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

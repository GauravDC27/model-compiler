"""Phase-0 router atlas on a small real MoE (CPU).

Measures on granite-3.1-1b-a400m-instruct (24 layers, 32 experts, top-8):
  A. routing-margin distribution per layer (FP32);
  B. FP32 vs BF16 routing flips per layer, and how well a small margin predicts a flip;
  C. batch-mate invariance: each prompt alone vs inside a left-padded batch of 4
     (router flips, max logit deltas, greedy-continuation equality);
  D. task specificity of expert usage per layer (normalized MI vs a shuffled-label null);
  E. decision paraphrase invariance with label scoring vs greedy text decoding;
  F. ambient-input sensitivity: the model's chat template injects today's date.

Usage: python experiments/01_router_atlas.py [--out experiments/results/01_router_atlas.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dmc import moe_probe as mp  # noqa: E402
from dmc.metrics import paraphrase_invariance, unique_at_n  # noqa: E402
from prompts_v0 import DECIDE_LABELS, FAMILIES  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
FIXED_SYSTEM = (
    "Knowledge Cutoff Date: April 2024.\nToday's Date: October 21, 2026.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant."
)
GEN_TOKENS = 24


def render(tok, text: str, system: str = FIXED_SYSTEM) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def content_span(tok, rendered: str, text: str) -> tuple[int, int]:
    prefix = rendered.split(text, 1)[0]
    start = len(tok(prefix, add_special_tokens=False).input_ids)
    end = len(tok(prefix + text, add_special_tokens=False).input_ids)
    return start, end


def encode(tok, texts: list[str]):
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)


def greedy(model, tok, enc) -> list[list[int]]:
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=GEN_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id
        )
    return [row[enc.input_ids.shape[1]:].tolist() for row in out]


def label_scores(model, tok, rendered: str, labels: tuple[str, ...]) -> dict[str, float]:
    """Log P(label | prompt) for each label, summed over the label's tokens (one pass each)."""
    scores = {}
    base = tok(rendered, add_special_tokens=False).input_ids
    for lab in labels:
        lab_ids = tok(lab, add_special_tokens=False).input_ids
        ids = torch.tensor([base + lab_ids])
        with torch.no_grad():
            logp = model(input_ids=ids).logits[0].float().log_softmax(-1)
        scores[lab] = float(
            sum(logp[len(base) - 1 + i, t] for i, t in enumerate(lab_ids))
        )
    return scores


def parse_label(text: str, labels: tuple[str, ...]) -> str:
    t = text.strip().lower()
    for lab in labels:
        if t.startswith(lab):
            return lab
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/results/01_router_atlas.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    m32 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    m16 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    k = m32.config.num_experts_per_tok

    items = [
        (fam, g, p, text)
        for fam, groups in FAMILIES.items()
        for g, paras in enumerate(groups)
        for p, text in enumerate(paras)
    ]

    all_margins, all_flips_dtype, all_margins_dtype = [], [], []
    per_layer_flip = []
    usage: dict[str, torch.Tensor] = {}
    batch_router_flip, batch_logit_maxdiff, batch_gen_equal = [], [], []
    repeat_unique = []

    for idx, (fam, g, p, text) in enumerate(items):
        rendered = render(tok, text)
        enc = encode(tok, [rendered])
        s, e = content_span(tok, rendered, text)

        # A. margins (FP32), all positions
        rl32 = mp.router_logits(m32, enc.input_ids, enc.attention_mask)[:, 0]  # L,T,E
        mg = mp.margins(rl32, k)
        all_margins.append(mg)

        # B. FP32 vs BF16
        rl16 = mp.router_logits(m16, enc.input_ids, enc.attention_mask)[:, 0]
        fl = mp.flips(rl32, rl16, k)
        all_flips_dtype.append(fl)
        all_margins_dtype.append(mg)
        per_layer_flip.append(fl.double().mean(dim=1))

        # D. usage by family over the user-content tokens only
        cmask = torch.zeros(1, enc.input_ids.shape[1])
        cmask[0, s:e] = 1
        u = mp.expert_usage(rl32[:, None], k, cmask)
        usage[fam] = usage.get(fam, torch.zeros_like(u)) + u

        # C. batch-mates: same prompt inside a batch with 3 random other prompts
        others = random.sample([it[3] for it in items if it[3] != text], 3)
        batch = [rendered] + [render(tok, o) for o in others]
        encb = encode(tok, batch)
        rlb = mp.router_logits(m32, encb.input_ids, encb.attention_mask)[:, 0]
        rlb = rlb[:, -rl32.shape[1]:]  # left padding: the prompt's tokens are the suffix
        batch_router_flip.append(float(mp.flips(rl32, rlb, k).double().mean()))
        batch_logit_maxdiff.append(float((rl32 - rlb).abs().max()))
        if idx % 3 == 0:  # greedy continuation alone vs in batch (subsample for time)
            alone = greedy(m32, tok, enc)[0]
            inb = greedy(m32, tok, encb)[0]
            batch_gen_equal.append(alone == inb)
            # Run-to-run repeatability alone (3 runs).
            repeat_unique.append(unique_at_n([tuple(greedy(m32, tok, enc)[0]) for _ in range(2)] + [tuple(alone)]))
        print(f"[{idx + 1}/{len(items)}] {fam} g{g} p{p}  {time.time() - t0:.0f}s", flush=True)

    margins_flat = torch.cat([m.flatten() for m in all_margins])
    flips_flat = torch.cat([f.flatten() for f in all_flips_dtype])
    mflat_dtype = torch.cat([m.flatten() for m in all_margins_dtype])
    layer_flip = torch.stack(per_layer_flip).mean(0)
    layer_margin_median = torch.stack(
        [torch.cat([m[layer] for m in all_margins]).median() for layer in range(rl32.shape[0])]
    )

    # D. task specificity and a shuffled-label null (permute family labels across prompts)
    nmi = mp.task_specificity(usage)
    null_runs = []
    per_prompt = []
    for fam, g, p, text in items:
        rendered = render(tok, text)
        enc = encode(tok, [rendered])
        s, e = content_span(tok, rendered, text)
        cmask = torch.zeros(1, enc.input_ids.shape[1])
        cmask[0, s:e] = 1
        rl = mp.router_logits(m32, enc.input_ids, enc.attention_mask)
        per_prompt.append((fam, mp.expert_usage(rl, k, cmask)))
    fams = [f for f, _ in per_prompt]
    for r in range(20):
        rng = random.Random(r)
        shuffled = fams[:]
        rng.shuffle(shuffled)
        u2: dict[str, torch.Tensor] = {}
        for f, (_, u) in zip(shuffled, per_prompt):
            u2[f] = u2.get(f, torch.zeros_like(u)) + u
        null_runs.append(mp.task_specificity(u2))
    null = torch.stack(null_runs)

    # E. decision paraphrase invariance: label scoring vs greedy text
    scored_groups, text_groups, margins_dec = [], [], []
    for paras in FAMILIES["decide"]:
        sg, tg = [], []
        for text in paras:
            rendered = render(tok, text)
            sc = label_scores(m32, tok, rendered, DECIDE_LABELS)
            best = max(DECIDE_LABELS, key=lambda lab: (sc[lab], -DECIDE_LABELS.index(lab)))
            ordered = sorted(sc.values(), reverse=True)
            margins_dec.append(ordered[0] - ordered[1])
            sg.append(best)
            out = greedy(m32, tok, encode(tok, [rendered]))[0]
            tg.append(parse_label(tok.decode(out, skip_special_tokens=True), DECIDE_LABELS))
        scored_groups.append(sg)
        text_groups.append(tg)

    # F. ambient date: same prompt, template rendered on two different "today" dates
    other_day = FIXED_SYSTEM.replace("October 21, 2026", "October 22, 2026")
    date_changed = 0
    for fam in FAMILIES:
        for paras in FAMILIES[fam]:
            text = paras[0]
            a = greedy(m32, tok, encode(tok, [render(tok, text)]))[0]
            b = greedy(m32, tok, encode(tok, [render(tok, text, other_day)]))[0]
            date_changed += int(a != b)

    result = {
        "model": MODEL,
        "torch": torch.__version__,
        "n_prompts": len(items),
        "top_k": k,
        "A_margin": mp.MarginSummary.of(margins_flat).__dict__,
        "A_layer_margin_median": [round(float(x), 5) for x in layer_margin_median],
        "B_dtype_flip_rate_decisions": float(flips_flat.double().mean()),
        "B_dtype_tokens_with_any_flip": float(
            torch.cat([f.any(0) for f in all_flips_dtype]).double().mean()
        ),
        "B_dtype_flip_rate_by_layer": [round(float(x), 4) for x in layer_flip],
        "B_auc_small_margin_predicts_flip": mp.roc_auc(-mflat_dtype, flips_flat),
        "C_batch_router_flip_rate_mean": sum(batch_router_flip) / len(batch_router_flip),
        "C_batch_router_logit_maxdiff_max": max(batch_logit_maxdiff),
        "C_batch_greedy_equal_share": sum(batch_gen_equal) / len(batch_gen_equal),
        "C_repeat_unique_at_3_max": max(repeat_unique),
        "D_task_nmi_by_layer": [round(float(x), 4) for x in nmi],
        "D_null_nmi_mean_by_layer": [round(float(x), 4) for x in null.mean(0)],
        "D_null_nmi_p95_by_layer": [round(float(x), 4) for x in null.quantile(0.95, dim=0)],
        "E_decide_PIR_label_scoring": paraphrase_invariance(scored_groups),
        "E_decide_PIR_greedy_text": paraphrase_invariance(text_groups),
        "E_decide_groups_label_scoring": scored_groups,
        "E_decide_groups_greedy_text": text_groups,
        "E_decide_label_margins": [round(x, 4) for x in margins_dec],
        "F_date_change_alters_greedy_output": f"{date_changed}/{sum(len(v) for v in FAMILIES.values())}",
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

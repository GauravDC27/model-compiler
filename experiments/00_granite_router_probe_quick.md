# Quick router-fragility probe — granite-3.1-1b-a400m-instruct (CPU)

Date: 2026-10-21 · Script: `experiments/00_granite_router_probe_quick.py` · Torch 2.14.1+cpu, transformers 5.20.2

Model: GraniteMoE, 24 layers, 32 experts, top-8 routing (router logits computed in FP32, `torch.topk`).
Input: one 26-token prompt (loan-decision question), single forward pass.

| Measurement | Value |
|---|---|
| Router margin (8th − 9th logit), 1% / 5% / 10% / 50% quantiles | 0.0010 / 0.0044 / 0.0088 / 0.072 |
| Share of (layer, token) decisions with margin < 1e-2 | 11.2% |
| Share with margin < 1e-3 | 0.96% |
| FP32 vs BF16 weights: (layer, token) expert sets that differ | 4.3% |
| FP32 vs BF16: tokens with ≥1 layer routed differently | 69.2% |
| FP32 batch=1 vs batch=3 (left-padded): routing differences | 0 |
| FP32 batch=1 vs batch=3: max abs router-logit diff / max abs output-logit diff | 5.5e-6 / 2.1e-5 (not bitwise equal) |

Reading: a small MoE has a heavy mass of near-tie routing decisions; a precision change flips a few percent of
per-layer decisions but touches most tokens somewhere in the stack — the same shape as the published
Qwen3-30B-A3B engine-mismatch numbers (~10% decisions, 94% tokens). CPU eager PyTorch is not batch-invariant
either (logits differ at 1e-5 between batch sizes), even though no routing flipped on this single prompt.
This is a single-prompt smoke test; the systematic version is `dmc.moe_probe` (Phase 0).

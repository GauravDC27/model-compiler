# Pre-registration — router margin-regularization pilot (plan v2 §4.4; review finding F3)

Registered 2026-10-21, before any training run.

**Question.** Can fine-tuning only the router weights with a hinge penalty on the top-k vs
(k+1) logit gap make MoE routing robust to a numerics change, at negligible capability cost?

**Setup.** granite-3.1-1b-a400m-instruct (24 layers, 32 experts, top-8), FP32, CPU.
Trainable parameters: the 24 router matrices only (786k parameters); everything else frozen.
Each router row is renormalized to its original L2 norm after every step, so margins cannot be
widened by scaling logits (which would scale the numeric noise equally).
Loss = next-token cross-entropy on wikitext-2 train (128-token chunks, batch 4) +
λ · mean over (layer, token) of max(0, γ − (s_k − s_{k+1})), with λ = 5, γ = 0.1.
Adam, lr 2e-4, 120 steps, seed 0.

**Held-out evaluation** (before vs after), on 16 wikitext-2 validation chunks plus the 75
probe prompts (`experiments/prompts_v0.py`):
1. *Primary:* BF16 routing-flip rate — share of (layer, token) expert sets that differ between
   the FP32 model and a BF16 copy of the same weights.
2. Share of routing decisions with margin < 0.01 (FP32).
3. Held-out cross-entropy on the validation chunks.
4. Informational: Decide-profile decisions on the 30 decide/judge prompts — agreement with
   the original model, and FP32-vs-BF16 decision parity.

**Pass criterion (all must hold):** primary flip rate falls by ≥ 2× **or** margin mass below
0.01 falls by ≥ 2× while the flip rate does not rise; **and** held-out cross-entropy rises
by ≤ 1% relative. Otherwise the lever is reported as failed and removed from P5 (plan §4.4),
unless a follow-up with a different (λ, γ) is pre-registered separately.

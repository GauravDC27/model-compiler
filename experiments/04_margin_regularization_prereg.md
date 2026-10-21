# Pre-registration — router margin-regularization pilot (plan v3 §4.4; review findings F3, G2, G5)

**Amendment 1, registered 2026-10-21, before any training run.** The original registration
(same date) was never run; it is superseded because it could not falsify the hypothesis
(round-2 review G2): no control arm, an OR-clause pass criterion, no load-balance guard, an
R7-violating perturbation, and a single (λ, γ, seed).

**Question.** Does fine-tuning only the router weights with a hinge penalty on the top-k vs
(k+1) router-logit gap make routing robust to a reduction-order change, beyond what the same
fine-tune without the penalty does, at negligible capability cost?

**Model and trainable parameters.** granite-3.1-1b-a400m-instruct (24 layers, 32 experts,
top-8, `router_aux_loss_coef = 0.0`, i.e. no load-balance loss). Only the 24 router matrices
are trained (786k parameters). After every step each router row is renormalized to its
original L2 norm, so margins cannot be widened by scaling logits.

**Training.** FP32, CPU, `torch.use_deterministic_algorithms(False)` during training only.
Loss = next-token cross-entropy on wikitext-2 train (128-token chunks, batch 4) +
λ · mean over (layer, token) of max(0, γ − (s_k − s_{k+1})). Adam, lr 2e-4, 80 steps.
The data order is drawn from the seed.

**Arms.** Control λ = 0; grid γ ∈ {0.05, 0.1, 0.2} × λ ∈ {1, 5}. All seven arms on seed 0.
On seed 1: the control and every grid arm that passed on seed 0.

**Evaluation inputs.** 16 wikitext-2 validation chunks + the 75 probe prompts; Decide-profile
decisions on the 30 probe decisions and 126 synthetic-oracle decisions (`dmc.oracle.generate(7, seed=5)`).

**Primary metric (sole).** Routing-flip rate between two BF16 copies of the trained weights,
with R7-hardened routers (FP32 router GEMM, lowest-index tie-break), executed with 4 vs 3
CPU threads (same precision, different reduction order).

**Pass criterion for an arm (all must hold):**
1. Primary flip rate ≤ ½ × the control arm's flip rate on the same seed.
2. Held-out cross-entropy ≤ 1.01 × the original model's.
3. Mean per-layer normalized expert-load entropy ≥ 0.95 × the original model's.
4. Decide-profile agreement with the original model ≥ 90% on the 30 probe decisions and
   ≥ 95% on the 126 oracle decisions.
5. Oracle accuracy ≥ the original model's minus 2 points.

**Multiplicity rule.** The lever survives only if at least one grid arm passes on seed 0 *and*
on seed 1. Otherwise it is reported as failed and removed from P5 (plan §4.4).

**Secondary (informational, not used for the decision).** Share of routing decisions with
margin < 0.01; FP32-vs-BF16 (hardened) stress flip rate; Decide-profile parity between the two
BF16 thread configurations on the first 60 oracle decisions; training curves (the logged "ce" is pure
cross-entropy because the aux-loss coefficient is 0.0).

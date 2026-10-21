# Pre-registration — System-1 / code / System-2 cascade compiled by the scheduler (plan v5 items iv, v)

Registered 2026-10-21, before any run.

**Question.** Does a compile-time schedule built from calibrated executor cost models predict
the realized accuracy and token cost of a cascade, and how many LLM tokens does a cascade of
cheap executors save against always running the full LLM decision profile?

**Executors** (granite-3.1-1b-a400m-instruct, FP32, CPU; R7-hardened routers):
- `code.canonical` — regex parser for template 0 of each `dmc.oracle` family plus exact
  evaluation; abstains when the question does not parse. 0 LLM tokens.
- `s1.probe` — logistic regression (L2 = 1e-2, full-batch LBFGS) on the last-token hidden state of
  the bare question (no chat template); layer chosen from {6, 12, 18, 23} on the calibration
  split; abstains when max(p, 1-p) < τ, τ = smallest threshold whose calibration accuracy on
  answered items is ≥ 0.95 (if none, τ = 1, i.e. it always abstains). Tokens = question tokens.
- `s2.decide` — the Decide profile (R17 scorer, both label orders). Tokens = rendered prompt
  tokens over both orderings (no generated tokens).

**Data.** Probe training: `generate(60, seed=100)` (1,080 items). Calibration (coverage,
accuracy, forced accuracy, mean tokens per executor): `generate(10, seed=7)` (180 items).
Test: `generate(20, seed=0)` (360 items).

**Procedure.** Build `Executor` cost models from calibration; compile one node per test decision
with `required_accuracy` ∈ {0.6, 0.7, 0.8} (infeasible bars must raise `CompileError`, recorded);
run each compiled ladder on the test split.

**Measures and pre-registered checks.**
1. Predicted (scheduler) vs realized accuracy: realized lies inside its own Wilson 95% interval
   around the predicted value, or the gap is reported as a calibration-transfer failure.
2. Predicted expected tokens vs realized mean tokens per decision: within ±10%.
3. Tokens per decision and accuracy vs the always-`s2.decide` baseline (paired).
4. Worst-case tokens: every realized decision ≤ its compiled worst case (must hold by construction).

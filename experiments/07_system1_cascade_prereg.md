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

**Amendment 1, registered 2026-10-21, before any run** (round-4 review M6).

The scheduler's original estimate multiplied each rung's marginal accuracy and coverage, which
assumes a rung's errors are independent of which items earlier rungs pass up. In a cascade the
escalated items are the hard ones, so that assumption over-predicts accuracy. Changes:

- **Joint calibration.** The script records an `Outcome` (answered at threshold, correct,
  correct when forced, tokens spent) for every executor on every calibration item and passes them
  to `compile_schedule(calibration=...)`. The scheduler then replays each candidate ladder on these
  items: predicted accuracy is the share the ladder gets right, and predicted tokens are the
  tokens actually spent on the items that reached each rung. This joint prediction is the one
  checks 1 and 2 test. The independent prediction is computed too and reported beside it.
- **Bars.** `required_accuracy` ∈ {0.6, 0.7, 0.8, 0.9}; 0.9 is added as a secondary bar and is
  expected to raise `CompileError` unless `s2.decide` reaches it.
- **Consequence of a calibration-transfer failure** (check 1 fails at any feasible bar). The
  scheduler's accuracy promise is a point estimate from a finite calibration split, so a failure
  means it does not transfer at this calibration size. The plan then changes, recorded as a dated
  plan amendment: the goal compiler compares the bar against the Wilson 95% lower bound of the
  joint estimate (not its point value), and every (op, difficulty) bucket needs at least as many
  calibration items as that bound requires to clear the bar. A token-prediction failure (check 2)
  has the same consequence for tokens, with the upper bound of the per-item token mean.
- **Additional report.** For each bar, whether the joint and independent estimates choose the
  same ladder, and the gap between each prediction and the realized accuracy.
- **Compiled sizes, so check 4 holds by construction** (added the same day, still before any run,
  after a dry run on a tiny random model showed the original executor sizes were *mean* token
  counts, which a longer test question can exceed). Every question is truncated to a compiled
  cap of 128 tokens before any executor sees it (cap hits are counted and reported; none are
  expected, oracle questions are far shorter). Executor token fields are now bounds:
  `code.canonical` reads no tokens; `s1.probe` reads the capped question once; `s2.decide` reads a
  fixed template once per label ordering (bounded by the calibration maximum plus 4 tokens of
  tokenizer-boundary slack per ordering) plus the capped question once per ordering. Expected
  tokens (check 2) come from the joint calibration records, i.e. what calls really spent; the
  independent estimate has only the bounds and will over-predict tokens, which is reported, not
  tested.

**Amendment 2, registered 2026-10-21, before any run** (round-5 review N9). The question cap
truncates token ids and re-tokenizes the decoded text, which can add tokenizer-boundary tokens, so
`s1.probe`'s fixed size is now the same 4-token slack the Decide template uses. The pilot truncates
head-only at text level rather than through `dmc.scheduler.cap_input`; zero cap hits are expected
and their count is reported, so check 4 does not exercise `cap_input` and is not claimed to.

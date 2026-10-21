# Pre-registration — route-pinning pilot v2 (plan v3 R27, D2-fresh vs D2-replay; review G1, G5, G7)

Registered 2026-10-21, before any run of this design. Supersedes the v1 design of
`03_route_pinning.py`, which was never run (round-2 review G1).

**Stacks** (granite-3.1-1b-a400m-instruct, CPU, one request per forward pass):
- **REF** — BF16 weights, R7-hardened routers (FP32 router GEMM, lowest-index tie-break), 4 threads. Plays the locked reference lane.
- **ALT** — REF with 3 threads: same precision, different reduction order. *Primary* cross-engine proxy. Treatment-blind calibration before registration: ~4.5% of routing decisions differ between 4 and 3 threads on 10 prompts.
- **STRESS** — FP32 weights, hardened routers, 4 threads. Upper-bound perturbation (different precision).

**Data.** 360 synthetic-oracle decisions (`dmc.oracle.generate(20, seed=0)`: 6 families × 20 groups × 3 paraphrases, balanced yes/no) + the 30 decide/judge probe decisions; 25 greedy 32-token generations (first paraphrase of each probe group).

**Measures.**
1. Decision parity with REF for ALT and STRESS, *fresh* vs *pinned* (strict `RoutePinner` with REF's trace; any `TraceMismatch`/`TraceOverrun` is reported as a failure, never silently routed live). Wilson 95% CIs.
2. Risk–coverage: parity and oracle accuracy vs coverage when abstaining on REF's decision margin; reported at the pre-registered Decide coverage floor **C₀ = 0.85**.
3. Generation: teacher-forced per-position argmax agreement with REF's generated tokens (fresh vs pinned); free-running first-divergence index (fresh vs pinned; a pinned run ends at the first divergent token by strict mismatch).
4. Value of R7: REF-vs-ALT routing-flip rate with stock (BF16-GEMM) routers vs hardened routers, on the 75 probe prompts.
5. ND2 incremental test: split-half by paraphrase group. Combined abstention signal = decision margin − α · fragile-routing share (ε = 0.01), α chosen on half A to maximize parity at C₀, evaluated on half B; Δ parity vs margin-only at C₀ with a bootstrap 95% CI.
6. Oracle accuracy, paraphrase invariance (PIR) and consistent-correct rate (CCR) for REF.

**Directional hypotheses and consequences.**
- H1: pinned parity ≥ fresh parity (ALT and STRESS). If false, the plan's "D2-replay ≫ D2-fresh" hypothesis is withdrawn.
- H2: teacher-forced per-position agreement is higher with pinning.
- H3: hardened routers lower the REF-vs-ALT routing-flip rate relative to stock routers.
- H4: fragile-routing share adds parity at C₀ beyond decision margin alone (CI excludes 0). If false, ND2 is downgraded from a MUST to a diagnostic.

---

**Amendment 1, registered 2026-10-21, before any run** (round-3 review K1, K2, K3, K5, K6, K11
and §3 cut guidance).
- *Proxy calibration first:* `experiments/05_proxy_calibration.py` runs before this pilot and
  writes `05_proxy_calibration.json`. **ε for the fragile-routing share = p95 of |Δ router logit|
  REF vs ALT excluding the first position** (`eps_H4`); ε = 0.01 is a secondary. The proxy
  regime is "faithful" if mean per-position KL(REF‖ALT) ≤ 10 × 1.5e-3, else "stress"; in the
  stress regime results may remove levers or validate mechanisms but never set production C₀.
- *Scorer:* the R17 scorer (case and leading-space surface forms marginalized; now the default).
- *D2-fresh curves (K1):* primary curve ranks each stack's fresh decisions by **that stack's own
  decision margin**; the REF-margin ranking is secondary. Pinned curves rank by REF's margin
  (replay carries REF's receipt).
- *H1, paired (K5):* among decisions where fresh ≠ REF, *repaired* = pinned = REF; *introduced* =
  fresh = REF but pinned ≠ REF. H1 supported iff repaired > introduced with one-sided exact
  sign-test p < 0.05; "uninformative" if fewer than 5 fresh flips.
- *H4, three branches (K5), on the executing stack:* combined signal = ALT margin − α · ALT
  fragile share. **Support** (ND2 stays a MUST) if the bootstrap CI lower bound > 0; **demote**
  (ND2 becomes a diagnostic) if the CI upper bound < +0.5 points; otherwise **inconclusive**
  (confirmatory test at hub scale with n ≥ 2,000). Secondary: content-only fragile share
  (positions from the question onwards) and ε = 0.01. At n ≈ 300, H4 is exploratory.
- *Failures (K6):* a `TraceMismatch`/`TraceOverrun` is caught per item, recorded in
  `*_pinned_status`, and counted as a pinned disagreement; intermediate JSON is written after
  each part.
- *Baseline (K11):* REF's single-pass (teacher-forced) argmax vs its own incremental decode is
  reported, so teacher-forced agreement numbers are interpretable.
- *Cuts:* 15 oracle groups per family (270 oracle + 30 probe = 300 decisions); no free-running
  pinned generation for STRESS. Unchanged: ALT pinned arm, probe decisions, Part 4, 1,000
  bootstrap samples, all 25 generations.
- *Implementation note (2026-10-21, before the full run; no change to any quantity being
  estimated):* `HFBackend.score` now computes the prompt once and reads every single-token
  continuation from that pass (multi-token continuations still get a full pass). The scored
  probabilities are the same conditional probabilities; only the kernel call pattern changes,
  identically on every stack. Calibration result: ALT is in the **stress** regime (mean
  per-position KL 0.0205 = 13.7 × 1.5e-3), ε_H4 = 0.0331.

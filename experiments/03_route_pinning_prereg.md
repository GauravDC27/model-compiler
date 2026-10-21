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

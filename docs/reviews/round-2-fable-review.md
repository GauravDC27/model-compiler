# Adversarial review, round 2 — `docs/PLAN.md` v2 (DMC)

**Reviewer:** Fable 5.1 · **Date:** 2026-10-21 · **Materials:** PLAN.md v2; `docs/reviews/round-1-response.md`; research notes 01–07; `dmc/` (all modules), `tests/` (35 passing, run by me), `experiments/01–04` and `04_margin_regularization_prereg.md`; installed `transformers 5.20.2` router sources (`GraniteMoeTopKRouter`, `Qwen3MoeTopKRouter`) and the granite config (`router_aux_loss_coef = 0.0`). No web access was needed this round. Citations are `NN:line` into the research notes or `file:line` into the repo.

---

## Overall assessment

v2 is a materially better plan than v1: the hub is the model the evidence supports, D2 is a risk–coverage pair, adapter mixing is in the D1 definition and gate, the discovery program has non-optional deliverables, and the two pilots I asked for exist as code with a pre-registration. Most of my round-1 findings are genuinely resolved, not just acknowledged (table below). The remaining problems are concentrated in three places. First, **both pilots as designed can mislead**: the route-pinning pilot (03) re-executes a recorded route trace free-running, so after the first divergent token it pins routes that belong to a different token sequence, and it silently falls back to live routing on trace overrun without reporting it; the margin pilot (04) has no λ = 0 control arm, no load-balance guard on a model whose aux-loss coefficient is 0.0, a pass criterion in the pre-registration that is weaker than the one stated to me, and no capability guard on the decisions it exists to protect. Second, **the "other toolchain" proxy in both pilots violates the plan's own R7**: the BF16 granite model computes the router GEMM in BF16 and only then casts to FP32 (`modeling_granitemoe.py:129`), so a large share of the measured "flips" come from router-logit rounding that the production lane removes by construction; the pilots are measuring the wrong noise. Third, **the plan over-claims its code state** (`session_state_hash`, `parent_seed`, the receipt fields, "results in `experiments/results/`", "(done)/(running)") in a document whose premise is compiler-grade trust. On the two adjudications: the D2-fresh/D2-replay split is correct and the "no recorded route on first execution" argument is valid for *recorded* routes, but it misses two fresh-execution levers that follow directly from it (pinned prefix routes and a stack-independent router); and the batch-by-adapter-first ladder is defensible on memory grounds only if fallback 1 gets its own acceptance test, because only the merged-checkpoint fallback lands on the kernel set vLLM has validated. None of this requires re-architecting; it requires fixing the pilots before they run and making the status claims true.

---

## Adjudication of the two modifications

### (a) F3/F6 — D2-fresh vs D2-replay; "route pinning cannot be primary for first execution"

**Agree with the split; rebut the conclusion as incomplete.**

- *Agree:* a fresh request on a non-reference stack has no recorded route, and routes are a function of the full forward pass (hidden states at every layer, 02:15–26), so there is no cheap "route-only" pass that could supply them. D2-replay therefore cannot be the primary lever for first execution. The split is the right contract structure and should stay.
- *Rebut (1) — pinned prefix routes.* For Decide/Call/Judge, most tokens in a request are the versioned template prefix (system turn, policy text, few-shot exemplars), which is identical across requests and *known at lock time*. Its routes can be recorded once on the reference stack and shipped in the lock, exactly like a precompiled header. On a fresh execution only the user-content and generated tokens route live. This is a D2-fresh lever available today with the same engine support R27 already needs. The plan should add it to §4.4 and measure the share of fragile decisions that fall inside the prefix.
- *Rebut (2) — a stack-independent router.* The response treats routes as something that must be *recorded*; they can instead be *computed stack-independently*. StableMoE distills the routing into a lightweight router "decoupled from the backbone" and freezes it (02:205–209, 02:293); DeepSeek-V4 hash-routes its first three layers from token IDs (02:80, 02:111). A router whose input is the token embedding (an exact table lookup) rather than the residual stream can be evaluated in a pinned exact path on any stack and yields identical routes on first execution — "hash routing with a learned hash". The notes call task-ID/hash routing "the purest form of determinism" at an unquantified quality cost (02:111, 02:308). v2 lists StableMoE under S5 but only as a margin-widening lever; its real value is fresh-execution route invariance. Pre-register it as such (quality budget, per-layer: late layers tolerate intervention, 03:203–206).
- *Scope note for the contract table.* D2-replay's product role is (i) a **diagnostic decomposition** — it separates routing-induced from dense-induced cross-stack disagreement, which is exactly what the margin pilot needs as its target — and (ii) **migration/audit** (re-certifying a recorded corpus on a new SKU). It should be labelled that way in §1 so the "expected ≫ D2-fresh" line is not read as the headline D2 number. Also label "expected ≫" as a hypothesis until 03 has run.

### (b) F2 — batch-by-adapter ahead of merged checkpoints

**Agree with the ordering, conditional on an acceptance test the plan does not yet have.**

- The response's claim is "each forward pass contains one adapter, so mixed-adapter kernels never run". True, but it does not follow that the LoRA path is batch-invariant: with one adapter and 0..k same-adapter batch-mates, the request still runs through the LoRA shrink/expand kernels (Punica/SGMV-class), whose work partition depends on batch size. vLLM's batch-invariance docs pin only the LoRA *shrink* reduction ("8 fixed K chunks", 01:160); expand is unmentioned (verified live in round 1). So fallback 1 moves the problem from "mixed adapters" to "single-adapter LoRA kernels under batch-size variation" and needs its own cell in the matrix: {same-adapter batch-mates 0..k} × {adapter on/off} → identical logits.
- Only fallback 2 (merged checkpoints) lands on the kernel set vLLM has actually validated for Qwen3-30B-A3B (the tested list is the base model with no adapters, 01:150–156). That is an argument for keeping merged checkpoints as the *certification* path even if batch-by-adapter is the serving default. Two consequences the plan should state: (i) merging LoRA into BF16 weights is a numeric operation, so merged ≠ base+adapter at the ULP level — adapters trained unmerged and served merged must be re-certified against the merged hash (the lock already forces this); (ii) 6 × 61 GB = 366 GB fits one 8×80 GB node at one profile per GPU, TP = 1, so the memory objection is a cost, not a blocker.
- Batch-by-adapter also changes the scheduler (per-adapter queues), which Vosti identifies as a leak surface in its own right (01:31–38, 01:371). Add "scheduler policy = batch-by-adapter" to the lock.

Verdict on (b): keep the order; add the fallback-1 acceptance test; keep merged checkpoints as the certification reference.

---

## Findings

### BLOCKER

#### G1 — The route-pinning pilot (03) measures the wrong thing for generation and can silently fall back to live routing
**Where:** `experiments/03_route_pinning.py:57–75`; `dmc/routing.py:109–123`; PLAN §1 D2-replay row, R27.
**What is wrong.**
1. *Free-running replay of a trace recorded on a different token sequence.* The reference run records routes for *its* generated tokens. The pinned BF16 run is free-running: if dense numerics flip the argmax at step t, every subsequent decode step feeds a different token, but `RoutePinner` keeps consuming the trace by call order (`routing.py:113–116`) and pins routes recorded for the reference's token at that position. After the first divergence the pinned run is routing tokens with someone else's routes. `greedy32_text_equal_pinned` and `greedy32_first_divergence_pinned` therefore do not measure "D2-replay with the routing amplifier removed"; they measure a hybrid that can be *worse* than fresh after divergence.
2. *Silent overrun.* On trace exhaustion the pinner returns live routing and only increments a counter (`routing.py:117–119`). The generation loop never asserts or records `pin.overruns` (`03:66–75`); only the decision loop does (`03:48`). If the reference hits EOS before 32 tokens and the BF16 run does not (or vice versa), the pinned run partially routes live and the result is reported as "pinned".
3. *No token-identity check.* The trace stores only expert indices; nothing verifies that the re-executed input tokens match the recorded ones (shape only, `routing.py:117`). A receipt that cannot be validated against its inputs is not a receipt.
4. *Decisions are measured at full coverage only.* `abstain_margin=0.0` (`03:38–39`), so the pilot reports parity at coverage 1.0 and never the risk–coverage curve the D2 gate is defined on, although every margin needed for the curve is already computed.
5. *Power.* 30 decisions and 25 generations give ±10–15-point CIs on parity; the pilot cannot distinguish 99.5% from 90%, and "fresh vs pinned" differences of a few items are noise.
**Fix (before running).**
- Record `input_ids` per router call in the trace; `RoutePinner` verifies them and **raises** on mismatch or overrun in the deterministic lane (a `strict=True` default); receipts carry `pinned: full | partial | none`.
- For generation, report D2-replay as **teacher-forced per-position agreement**: feed the reference's token sequence to the BF16 model with pinned routes and compare argmax (and logit error) at every position; report free-running pinned equality separately, truncated at first divergence.
- Add the risk–coverage curve (parity vs abstain band, fresh and pinned) using the margins already computed.
- Raise n: use the `prompts_v0` families as the smoke set but add ≥300 synthetic-oracle decisions (R20) before quoting any parity number with a CI.

#### G2 — The margin-regularization pre-registration cannot falsify the hypothesis as written
**Where:** `experiments/04_margin_regularization.py`; `04_margin_regularization_prereg.md`; PLAN §4.4, §7 P0.
**What is wrong.**
1. *No control arm.* Any router-only fine-tune on wikitext-2 can change the flip rate (routers become more decisive on in-distribution text). Without a **λ = 0** arm with identical steps, LR, norm constraint and seed, a 2× flip reduction cannot be attributed to the hinge.
2. *No load-balance guard, on a model with no load-balance loss.* The plan says "load-balance terms unchanged" (§4.4); granite-3.1-1b-a400m has `router_aux_loss_coef = 0.0`, so there is no term to keep. The cheapest way to satisfy a hinge on the k/(k+1) gap at every token is to make routing more *concentrated*; expert-load collapse is the documented coupling (02:168–169) and a ≤ 1% CE rise on 2k held-out tokens after 120 steps will not reliably catch it. Add expert-load entropy (per layer, before/after) with a pre-registered bound.
3. *Pass criterion is weaker than stated.* The coordinator's message says "pass = flip rate ↓ ≥ 2× at ≤ 1% CE rise"; the pre-registration (`prereg.md:25–27`) and code (`04:143–146`) accept "**or** margin mass below 0.01 ↓ ≥ 2× while the flip rate does not rise". Mass below 0.01 can halve by nudging marginal decisions to 0.011 without any flip reduction — precisely the gaming the norm constraint was meant to prevent. Make flip rate the sole primary; mass is secondary/informational.
4. *No capability guard on the thing being protected.* "Decision agreement with the original FP32 model" is informational (`prereg.md:22–23`). The pilot can pass while changing a quarter of the Decide-profile decisions. Pre-register a bound (e.g. agreement ≥ 90% on the 30 decisions, and ≥ 95% on a larger synthetic set) as part of PASS.
5. *Garden of forking paths.* One seed, one (λ, γ), and "a follow-up with a different (λ, γ) may be pre-registered separately" (`prereg.md:28`). Pre-register the small grid now (γ ∈ {0.05, 0.1, 0.2}, λ ∈ {1, 5}) with a multiplicity rule, and require replication on a second seed before the lever survives.
6. *Primary metric is measured under an R7-violating perturbation* (see G5): the BF16 copy's router GEMM runs in BF16, so part of the "flip rate" the hinge is asked to reduce is router-logit rounding that the production lane removes by computing the router in FP32. The pilot may "pass" by widening margins against a noise source that does not exist in the lane, or "fail" against one that is twice as large as the real one.
7. *Bookkeeping.* Training runs with `torch.use_deterministic_algorithms(False)` (`04:112`) — fine for a pilot, but say so in the pre-registration; the train log's "ce" is `out.loss`, which (with aux coef 0.0) is pure CE here but would not be on OLMoE (aux coef 0.01), so label it.
**Fix (before running; the pre-registration must be amended now, not after results).** Add the λ = 0 arm, the load-entropy guard, the decision-agreement bound, the grid + second seed, make flip rate the sole primary, and compute flips under the R7-compliant perturbation of G5.

### MAJOR

#### G3 — The plan claims code and results that do not exist
**Where:** PLAN §3.1 ("`Context {now, locale, timezone, session_state_hash}` … v0 is implemented in `dmc/ir.py`"), §5.4 (`parent_seed`), §3 step 9 (receipt fields), §7 P0 ("*(done on CPU)*", "*(running)*"), §9 and README ("results in `experiments/results/`"); R12 acceptance ("cache key includes session state").
**Evidence.** `dmc/ir.py` is unmodified since round 1: `Context` has `now, locale, timezone` only (`ir.py:26–31`); there is no `parent_seed` or `session_state_hash` anywhere in `dmc/` (grep); the IR has six ops, not the twelve listed in §3.1; `Receipt` carries `margins, verifier, fragile_routing_share` only (`store.py:19–31`) — no session hash, coverage band, routing-trace ref, sampling-mask summary or normalization mode; `experiments/results/` does not exist; `result_key` therefore cannot include session state, so the D0/D1 promise "+ session state" (§1) is not what the code keys on.
**Why major.** The response table says "IR `Context` gains `session_state_hash`" under *what changed in v2*; it changed in prose. In a plan whose thesis is "trust comes from receipts and locks", status claims must be literally true.
**Fix.** Either implement the four small items (Context field, `parent_seed`, receipt fields, results directory with the JSON the scripts already write) or change every "(done)/(running)/implemented" to "planned/pending" and make §9 list only what `pytest` exercises. Add a CI check that every `file:symbol` cited in PLAN.md exists.

#### G4 — The GDN risk was moved from the hub to the Create spoke without resolving it
**Where:** §4.1 spoke table (Create → Qwen3.8-27B "dense hybrid; GDN audit"), §5.4 ("D1 per (IR, seed)").
**What is wrong.** Qwen3.8-27B is a Gated-DeltaNet hybrid (02:345–347). No batch-invariant GDN kernel exists in any engine (round 1, F1; vLLM's hybrid entry is broken, 01:173; SGLang's hybrid Mamba2 fails, 01:237). Create's stated D1 target therefore has no execution path on its named spoke. The plan's own R6 would disable the lane.
**Fix.** State Create v1 as **D0-only on Qwen3.8-27B** (record/replay, seeded sampling for reproducibility *within* one recorded run), or run Create on the hub / a plain-attention dense writer (e.g. Qwen3-32B; EQ-Bench standing unknown — measure in P3) until a GDN audit passes. Tie the Create spoke's D1 claim to the same go/no-go as the Qwen3.6 spike.

#### G5 — The FP32-vs-BF16 "other toolchain" proxy is not representative, and it violates R7 on the "other" side
**Where:** `03_route_pinning.py:3,36–37`; `04_margin_regularization.py:94–96`; `02_cpu_conformance.py`; PLAN §7 P0 pilots.
**What is wrong.**
1. *The BF16 model's router runs in BF16.* `GraniteMoeTopKRouter.forward` computes `F.linear(hidden_states, self.weight).float()` — the GEMM is in the model dtype and the cast comes after (`modeling_granitemoe.py:129`). With router logits of magnitude ~1–10, BF16 ULP is 0.004–0.04, so margins below ~0.01 are *unrepresentable* on the BF16 side. The 00 probe note's "router logits computed in FP32" is wrong for the BF16 model. R7 mandates an FP32 router on every certified stack; the production "other toolchain" would not have this noise. Part of the measured 4.3%/69% (00 probe) is therefore an artifact of a configuration the plan forbids.
2. *Structure of the perturbation.* Cross-engine disagreement on Qwen3-30B-A3B is BF16-vs-BF16 with different reduction orders (R3: KL 1.5e-3, 02:148); FP32-vs-BF16 is a systematic, correlated weight-rounding bias plus activation rounding throughout — a different noise model with a privileged "truth" side. The reference lane in production is itself BF16 (the lock), not FP32.
**Fix.**
- Patch the BF16 model's routers to `F.linear(hidden.float(), weight.float())` (weights stay BF16-stored, as in the hub checkpoint; vLLM's BI path upcasts the gate GEMM to FP32, 01:190, 02:511) so flips come only from backbone hidden-state differences. Report old vs new flip rate; this is itself a useful number (how much R7 buys).
- Make the primary cross-engine proxy **BF16-vs-BF16 with reduction-order variation** (thread count 1/2/4 and batch-mates, which `02_cpu_conformance.py` already varies for FP32) — same precision, different order, like two engines. Keep FP32-vs-BF16 as the stress upper bound and say so.
- Both pilots' primary metrics then use the R7-compliant perturbation.

#### G6 — `pin_routes` at production scale is unbudgeted engine work
**Where:** R27, §4.4, §7.
**What is wrong.** `dmc/routing.py` monkeypatches HF modules; nothing equivalent exists in vLLM serving. Routing replay exists in *training* frameworks (R3 in Megatron/ms-swift, 02:155, 03:298) and as "Router Replay" in the TRT-LLM RFC's phase 1 (01:248), not in vLLM's fused-MoE path. R27 ("support `pin_routes` re-execution") and the pinned-prefix lever above both need a fused-MoE entry point that accepts external top-k indices and recomputes gate weights. The 384 B/token trace is cheap; the kernel plumbing is not free.
**Fix.** Add an engineering item to P1/P6 ("external-route fused-MoE path in vLLM; estimate"), and until it lands, scope D2-replay certification to the HF reference stack (where it works) with the engine as the "other" side. Correct R27's "~1–2 KB/token": with one byte per index it is 48 × 8 = 384 B/token on the hub; 1–2 KB only if gate weights are stored too.

#### G7 — ND2's acceptance test is nearly tautological
**Where:** R24, §4.3 ND2.
**What is wrong.** "Abstention on high-fragility requests reduces D2-fresh disagreement" will pass for any signal correlated with the decision margin — including the decision margin itself, which R19 already uses. The interpretability program's one hard product deliverable must show *incremental* value.
**Fix.** At fixed coverage C₀, parity with (decision margin + fragile-routing share) must exceed parity with decision margin alone, with a CI that excludes zero, on a held-out cross-stack set; also report the margin→flip AUC (`moe_probe.roc_auc`) that S2 already computes.

#### G8 — C₀ is set from the baseline, which lets the gate be tuned to pass
**Where:** §5.1 targets, §6 release gates ("C₀ set in P0 from measured baselines").
**What is wrong.** If C₀ is chosen after seeing how much abstention is needed to reach 99.5%, the gate measures nothing. The round-1 fix asked for a coverage *floor*.
**Fix.** Pre-register per-profile floors as product requirements before P0 measurement (e.g. Decide C₀ ≥ 0.85, Call ≥ 0.90, Judge ≥ 0.80), report the full curve, and treat "99.5% is only reachable below the floor" as a gate failure that triggers the dense-student arm.

### MINOR

#### G9 — "The only open MoE with vLLM batch-invariance validation" (§0.6) is false. gpt-oss-20b/120b, DeepSeek-V3/R1/V3.1 and Granite MoE are also on the tested list (01:150–156). Say "the only one with all five assets".

#### G10 — The conformance matrix is a 9-way cross product with "every cell: identical logits" and "unique@1000 … across the full matrix" (§6). Infeasible as stated. Specify a covering-array (pairwise) design over the matrix with the full product only on a small golden subset, and unique@1000 only in the base configuration (unique@20–50 per cell).

#### G11 — R7's citation is loose: the merged V4-Flash path's "position tie-break" is for the DSA top-k indexer (01:66, 01:192); the lowest-expert-index MoE tie-break is in draft PR #54955 (02:512). Cite both and keep "lowest index" as our rule (`moe_probe.stable_topk` implements it).

#### G12 — R13a's mechanism is template-specific. "An explicit system turn suppresses the date" holds for granite; Llama-3.x templates emit `Today Date:` in the system header whether or not a system message is supplied. The date-shift acceptance test is the real guard — keep it — and additionally pin the template's `date_string`/`strftime_now` to `Context.now` and hash the *rendered prompt bytes* into the receipt.

#### G13 — Label scoring sums log-probs over label tokens (`runtime.py:84`, `decide.py:72`), so multi-token labels carry a length bias. Fine for yes/no/A/B/1–5; document the single-token-label rule for the IR's closed enums or length-normalize.

#### G14 — Hub checkpoint not pinned: "Qwen3-30B-A3B … Instruct checkpoint" (§4.2) — Instruct-2507, Thinking-2507 or the original? Pin the HF revision in the lock and say which.

#### G15 — Vosti's Triton analyzer is assumed available for the P6 extension; the notes do not say it was released (01:369–374). Confirm or budget a re-implementation.

#### G16 — R9 cites `runtime.py::position_seed` for "stable tie order in top-k/top-p truncation"; the sampler implements Gumbel-max only, no truncation. Fine for now; make the citation match.

#### G17 — `Compiler.dispatch` maps the op family name to a profile key (`compiler.py:98–103`); the plan's table maps `research.*` to Reason. Trivial, but the table and the code should agree.

#### G18 — `RouteTracer`/`RoutePinner` are tested in FP32 only (`tests/test_routing.py`). Add a BF16 own-trace identity test (the gate-weight recompute path is the same as the model's — I checked both router classes — but the test should prove it) and a strict-mode overrun test once G1 is fixed.

#### G19 — The 00 probe note says "router logits computed in FP32"; for the BF16 model that is false (G5). Correct the note so the baseline number is not misread.

---

## Round-1 findings: resolution status in v2

| ID | Status | Reason |
|---|---|---|
| F1 | **RESOLVED** | Hub = Qwen3-30B-A3B text-only; Qwen3.6 is a P0 spike with a written go/no-go; atlas on the shipping hub. (GDN risk re-surfaces on the Create spoke → G4.) |
| F2 | **PARTIAL** | Adapter-mix/mask fuzzing in R8/R26/§6 and the P1 gate: yes. Fallback 1 has no acceptance test of its own and only fallback 2 lands on validated kernels → adjudication (b). |
| F3 | **PARTIAL** | Risk–coverage gate with conformal τ: yes. Pilot pulled to P0: yes, but its pre-registration cannot falsify the hypothesis (G2) and C₀ is derived from the baseline (G8). |
| F4 | **PARTIAL** | Lock/R6/R9/R12 text: yes. The code the plan says implements `session_state_hash` does not (G3); sampler truncation tie order not implemented (G16, acceptable). |
| F5 | **RESOLVED** | ND1–ND3 mandatory, atlas diff gate, CLTs dropped, dictionaries re-validated per adapter. (ND2's test needs teeth → G7.) |
| F6 | **PARTIAL** | Route pinning, StableMoE, dense students added. Misses the fresh-execution route levers (adjudication (a)) and the engine work for R27 (G6). |
| F7 | **RESOLVED** | Qwen3-32B/8B controls; tax reported as a tuple. |
| F8 | **RESOLVED** | Hub + certified spokes with a spoke table; Skywork citation corrected to the Apache Qwen3-8B variant (05:33, 05:239); Tongyi atlas-transfer claim removed. |
| F9 | **RESOLVED** | RL baseline = R3 + FP32 router + FP16/TIS with sampling mask; bitwise RL is a costed stretch. |
| F10 | **RESOLVED** | Dense parser default; pick-flip and adversarial-flip metrics in R21; attack surface in §8. |
| F11 | **RESOLVED** | D1 wording downgraded; P6 analyzer extension with an estimate (availability caveat → G15). |
| F12 | **RESOLVED** | 1.6–2.1× / 91.3% corrected; Qwen3-Next status corrected; both-order judging is a MUST. R7 citation still loose (G11). |
| F13 | **RESOLVED** | Six profiles, Parse first. |
| F14 | **PARTIAL** | README and §9 rewritten and accurate about modules/tests (35 pass); but "results in `experiments/results/`" names a directory that does not exist (G3). |
| F15 | **RESOLVED** | Criterion-conditional normalization, recorded in the receipt. |
| F16 | **RESOLVED** | Static escalation constants in the dispatch table and lock. |
| F17 | **RESOLVED** | INT8 power-of-two scales and ReproFuse (M ≤ 64) in §4.4. |
| F18 | **RESOLVED** | Paraphrase ensembling in R14 (offline + optional online). |
| F19 | **RESOLVED** | CPU vertical slice exists (`compiler.py` + `RuleParser` + Decide + store + tests). |
| F20 | **PARTIAL** | R27 and receipt contents in prose; `Receipt` in code lacks trace ref and sampling-mask summary (G3). |
| F21 | **PARTIAL** | `parent_seed` semantics documented in §5.4; not in `dmc/ir.py` although §3.1 says v0 is implemented there (G3). |

---

## What v2 gets right — do not change

- The **D2-fresh / D2-replay split** and the risk–coverage framing with conformal thresholds.
- **Hub = Qwen3-30B-A3B text-only**, Qwen3.6 as a gated spike, matched dense controls, the tax as a tuple.
- **Hub + certified spokes** with a license-bearing spoke table, and dense students as a first-class, pre-stated outcome.
- **R26** (adapter mix and masks in the D1 definition and the P1 gate) and the fallback ladder's existence.
- **ND1–ND3 as mandatory release artifacts**, atlas diff as a gate, CLTs dropped from v1, dictionaries re-validated per adapter.
- **R13a** and the date-shift test (a real finding from the CPU work), the explicit system turn, `NondeterminismError` on divergent recompute, stable top-k with lowest-index ties, order-averaged label scoring with a logged margin.
- Pre-registering the pilots at all, with a norm-fixed router so margins cannot be widened by scaling — the right instinct; G2 is about making it bite.
- The RL baseline (R3 + FP32 router + FP16/TIS) and bitwise RL as a costed stretch.
- The dense parser default and the adversarial-flip metrics.

---

## VERDICT: APPROVE WITH CHANGES

Minimum changes required for approval:

1. **G1** — Fix the route-pinning pilot before it runs: token-ID verification and strict overrun handling in `RoutePinner`; teacher-forced per-position D2-replay for generation (free-running reported separately, truncated at first divergence); risk–coverage curve from the existing margins; ≥300 oracle decisions before any parity number is quoted with a CI.
2. **G2** — Amend the margin-regularization pre-registration before it runs: λ = 0 control arm; expert-load entropy guard (granite has no aux loss); flip rate as the sole primary (drop the OR clause); a pre-registered decision-agreement bound in PASS; the (λ, γ) grid and a second seed declared now.
3. **G5** — Make the "other toolchain" R7-compliant (FP32 router GEMM on the BF16 side) and add BF16-vs-BF16 reduction-order variants as the primary cross-engine proxy; both pilots' primary metrics use it.
4. **G3** — Make every status/code claim in PLAN.md true (implement `session_state_hash`, `parent_seed`, receipt fields and the results directory, or re-label as pending), and add a check that cited `file:symbol`s exist.
5. **G4** — Create v1 is D0-only on Qwen3.8-27B, or Create moves to a plain-attention model; its D1 claim is gated like the Qwen3.6 spike.
6. **G8** — Pre-register per-profile coverage floors as product requirements before P0 measurement.
7. **Adjudications** — add pinned prefix routes and the stack-independent (StableMoE/hash) router as pre-registered D2-fresh levers; add the fallback-1 acceptance test (same-adapter batch-mates 0..k) and keep merged checkpoints as the certification reference.

G6, G7 and the MINOR items are recommended but not required for approval.

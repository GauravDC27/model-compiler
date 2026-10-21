# Deterministic Model Compiler (DMC): Architecture Requirements and Implementation Plan

**Version:** v4 (after adversarial review rounds 1–3; architecture converged in round 3) · **Date:** 2026-10-21 · **Status:** final review round pending pilot results
**Review trail:** `docs/reviews/round-1-fable-review.md` → `docs/reviews/round-1-response.md` → `docs/reviews/round-2-fable-review.md` → `docs/reviews/round-2-response.md` → `docs/reviews/round-3-fable-review.md` → `docs/reviews/round-3-response.md`

> Analogy: people trust a compiler because (a) the same source and the same toolchain give the same binary, (b) the binary means what the source means, and (c) ill-formed source is rejected rather than guessed at. DMC aims to give a prompt the same three guarantees.

Evidence for every claim lives in `research_notes/Deterministic MoE model compiler/` (01–07, cited below as `NN:line`) and the synthesized report in `reports/`. Our own measurements are in `experiments/`.

---

## 0. Executive summary

1. **"Deterministic" is a stack of contracts**, not one property: D0 replay → D1 bitwise recompute → D2 decision parity across toolchains → D3 paraphrase invariance → D4 spec conformance, plus D⊥ defined failure. D0, D1 (on certified stacks) and D4-syntax are *guaranteed* by engineering. D2 and D3 are *certified statistically* as risk–coverage pairs; that is where the research lies.
2. **Bitwise determinism (D1) is now an engineering cost.** Batch-invariant kernels took Qwen3-235B-A22B from 80 unique completions per 1,000 temperature-0 runs to 1, at 1.6–2.1× runtime in the naive prototype (Thinking Machines, Sep 2025); a co-designed path (vLLM, DeepSeek-V4-Flash) runs at 91.3% of baseline throughput. vLLM's `VLLM_BATCH_INVARIANT=1` validates several MoEs but has open MoE leaks (vLLM #57016) and a broken hybrid-model entry (Qwen3-Next, since Oct 4 2026); formal checking (Vosti, Sep 2026) found leaks in both vLLM and SGLang deterministic modes. We own a conformance suite that also covers **multi-adapter batching**, which nobody has validated.
3. **MoE routing is the MoE-specific fragility.** Top-k routing is an argmax over continuous logits. On Qwen3-30B-A3B ~10% of router decisions differ between engines and 94% of tokens differ in at least one layer. Our Phase-0 atlas on granite-3.1-1b-a400m (75 prompts, §10; stock router, i.e. R7-violating BF16 router GEMM): 7.8% of router decisions sit within 0.01 logits of a tie; FP32-vs-BF16 flips 9.0% of expert sets and touches 79% of tokens, rising with depth from ~4% in early layers to ~17% in the last; a small margin predicts a flip with AUC 0.87. With R7-hardened routers, BF16 at different CPU thread counts (same precision, different reduction order) still flips 4.7% of routes (`experiments/results/05_proxy_calibration.json`), a stress-level perturbation (mean per-position KL 13.7× the published cross-engine value). The discovery program exists to **measure** these margins, **wire** them into abstention, **pin** routes for re-execution, and test whether training or a stack-independent router can **remove** the amplifier.
4. **The largest variance users see comes from the prompt, not the GPU.** Formatting alone swings accuracy by up to 76 points (FormatSpread). So the architecture is a **compiler pipeline**: NL → typed IR (a DSL) → static dispatch → profile → verifier; ambiguity is a compile error (ask), never undefined behaviour (guess).
5. **Architecture = one deterministic MoE hub + certified spokes.** Six profiles: **Parse** (NL→IR), **Decide**, **Call** (function calling), **Judge** (contrastive cross-evaluation), **Create**, **Reason**. Each profile is (adapter, optional expert mask, template, grammar, decoding policy, verifier, thresholds). Profiles run on the hub unless a spoke (a separately locked model) wins a pre-registered bake-off at equal determinism; dense students distilled from the hub are a first-class contender for the decision-bearing profiles. Task routing is **static dispatch on IR op type**; the only learned routing decision (intent selection inside Parse) runs on a small dense parser with a dead band.
6. **Hub model: Qwen3-30B-A3B (text-only; Instruct-2507 checkpoint, HF revision pinned in the lock).** Several MoEs are on vLLM's batch-invariance tested list (gpt-oss, DeepSeek-V3/R1, Granite MoE, Qwen3-30B-A3B); Qwen3-30B-A3B is the only one that also has published engine-mismatch (R3) data, all-layer SAEs (Qwen-Scope, Base checkpoint), a known super-expert list and a TransformerLens adapter. "Qwen 3.8 27B" exists but is a **dense** Gated-DeltaNet hybrid, not an MoE; it is a spoke candidate (Create) and a capability reference. Qwen3.6-35B-A3B (Apache-2.0, 256 experts) is a P0 spike with a written go/no-go, not the default.

---

## 1. The determinism contract

| Level | Name | Promise | Mechanism | Guarantee type |
|---|---|---|---|---|
| **D0** | Replay | Same canonical IR + session state + lock → byte-identical response | Content-addressed result store, record/replay | Hard (by construction) |
| **D1** | Bitwise recompute | Same canonical IR + session state + **lockfile** → byte-identical output under any load, batch mix (incl. mixed adapters), chunking or cache state | Batch/position-invariant kernels, fixed topology, counter-based RNG, conformance suite | Hard on certified stacks (tested; formal contract for dense kernels only) |
| **D2-fresh** | Decision parity, first execution | On any certified toolchain (different SKU / engine version), decision-bearing fields agree with the reference stack | One-pass candidate scoring, conformal abstention band, margin widening, dense students | Statistical: parity ≥ P at coverage ≥ C₀, with CIs |
| **D2-replay** | Decision parity, re-execution (diagnostic, migration and audit) | Re-executing a recorded result on another certified stack with the recorded routes pinned reproduces its decision-bearing fields | Routing traces in receipts + token-verified `pin_routes` execution | Statistical. Hypothesis (H1, under test in `experiments/03_route_pinning.py`): D2-replay ≥ D2-fresh because the routing amplifier is removed. Its product roles are (i) decomposing cross-stack disagreement into routing-induced vs dense-induced, and (ii) re-certifying a recorded corpus on a new SKU; it is not the headline D2 number. |
| **D3** | Semantic invariance | Meaning-preserving paraphrases / reformattings → same IR → same output | Canonical IR, dense parser with invariance training, paraphrase ensembling, clarification | Statistical: PIR on paraphrase suites at stated coverage |
| **D4** | Spec conformance | Output satisfies its typed contract; where an executable oracle exists, it satisfies the oracle | Grammar-constrained decoding, validators, executable verifiers, bounded deterministic repair | Hard for syntax/schema; oracle-dependent for semantics |
| **D⊥** | Defined failure | Ill-formed, out-of-grammar or ambiguous input yields `CompileError` / `Clarify` / `Abstain`, never a silent guess | Ambiguity slot in IR, calibrated abstention | Hard (behavioural contract); quality measured |

**D1 definition (Vosti-style):** bitwise-identical logits at corresponding output positions under variation of batch composition, adapter mix, chunk boundaries, prefill-vs-decode phase, and KV-cache state, for a fixed lock.

**Non-goals:** no cross-vendor bitwise equality (no published demonstration for full LLM inference as of Oct 2026); no correctness promise beyond D4 oracles (determinism makes errors reproducible, not absent); creative output is deterministic per (IR, seed), not unique per "meaning".

---

## 2. Architectural requirements for consistent output parity

"MUST" items gate release. Each has an acceptance test.

### 2.1 Toolchain identity and reproducibility

| ID | Requirement | Acceptance test |
|---|---|---|
| R1 | MUST pin every output-affecting artifact in a **lockfile**: weights (HF repo + revision), tokenizer files, chat template, adapters (merged or unmerged), expert masks; engine + version; `tokenizers`/`transformers`/torch/CUDA/driver versions; kernel set and determinism flags/env; **CPU/GPU thread or SM partitioning where it changes reduction order** (Phase 0: BF16 at 4 vs 3 CPU threads changes logits by up to ~4 and flips ~4.5% of routes); **grammar engine + version + grammar hash**; `async_scheduling`, speculative/MTP, prefix-cache and CUDA-graph settings; **scheduler policy** (e.g. batch-by-adapter); SKU; TP/EP/PP/DP; IR schema version; decoding policy; dispatch constants (`max_tier`, `max_repair`, `max_escalations`). | Lock hash on every receipt; replays refuse lock mismatch (`dmc/store.py::ResultStore`). |
| R2 | MUST canonicalize every hashed artifact (RFC 8785-style JSON, NFC text). | Property tests (`tests/test_canonical.py`). |
| R3 | MUST store every response content-addressed by `H(lock, canonical IR incl. session_state_hash, profile version, seed)` and support replay; a recompute that differs under an identical key is a hard error. | Replay byte-identical; divergence raises (`tests/test_store.py`). |
| R4 | MUST version and shadow-test routing tables, thresholds, templates and grammars as code. | Release diff lists every changed decision on the golden set. |

### 2.2 Numerical determinism of execution (D1)

| ID | Requirement | Acceptance test |
|---|---|---|
| R5 | MUST run deterministic-lane traffic on **batch- and position-invariant kernels** (matmul, RMSNorm, attention incl. split-KV with fixed split *size*, MoE grouped GEMM, LoRA shrink/expand, sampler). | `unique@1000 == 1` per golden prompt under randomized concurrent load. |
| R6 | MUST make execution invariant to chunked-prefill boundaries, prefill-vs-decode path and preemption/recompute; v1 lock **disables** prefix caching, speculative decoding/MTP and async scheduling in the deterministic lane until each passes the matrix. | Conformance matrix (§6): identical logits at every position per variation. |
| R7 | MUST compute router logits with an **FP32 GEMM** (not a model-dtype GEMM cast afterwards, which is what stock BF16 checkpoints do) and fixed reduction order, and use a **stable top-k tie-break** (lowest expert index). Precedents: vLLM's merged DeepSeek-V4-Flash consistent path (FP32 gate GEMM; position tie-break in the DSA indexer) and draft PR #54955 (lowest-index MoE tie-break). Implemented for HF models in `dmc/routing.py::harden_routers`. | Router-decision equality across load variations; exact-tie unit tests (`tests/test_routing.py`); value of R7 measured in Phase 0 (stock vs hardened flip rate). |
| R8 | MUST NOT use per-batch dynamic quantization scales or token-count-dependent kernel selection (incl. expert-mask-induced token-count changes) in the deterministic lane. | Batch-mate + adapter-mix + mask fuzzing on MoE reproducers. |
| R9 | MUST sample with a **counter-based RNG keyed by (request seed, position)**; any top-k/top-p truncation MUST use a stable tie order (prob desc, token id asc). Current code: Gumbel-max over the full distribution with `dmc/runtime.py::position_seed`; truncation is not implemented yet. | Same seed → same sample regardless of batch-mates. |
| R10 | MUST fix TP/EP/PP per certified lock (or use TP-invariant kernels, TBIK). | Topology in lock; cross-topology runs are D2 only. |
| R11 | SHOULD offer **selective determinism** (per-request SLA via verify-and-rollback, LLM-42 / MarginGate) so throughput traffic does not pay the tax. | Deterministic requests meet R5 while co-scheduled with fast-lane traffic. |
| R26 | MUST certify D1 with **≥3 co-resident adapters and expert masks on/off**; if multi-LoRA kernels fail, fall back to (1) **batch-by-adapter scheduling** (serving default; no extra memory), then (2) **merged per-profile checkpoints** (the certification reference, because only merged weights run on the kernel set vLLM has validated; 6 × ~61 GB fits one 8×80 GB node at one profile per GPU). Merged ≠ base+adapter at ULP level, so a merged profile is re-certified under its own weights hash. | Mixed-adapter fuzzing passes; for fallback 1, **{same-adapter batch-mates 0..k} × {adapter on/off} → identical logits** (vLLM pins only the LoRA shrink reduction); scheduler policy recorded in the lock. |

### 2.3 Semantic determinism (D3) and defined failure (D⊥)

| ID | Requirement | Acceptance test |
|---|---|---|
| R12 | MUST compile every request to a **closed, typed IR** (discriminated union, required nullable slots) whose `Context` carries time, locale, timezone and a **`session_state_hash`** (prior turns, prior IRs, snapshotted tool results); parsers keep only the ambient fields a request depends on (**context minimization**: `now` is null unless the request refers to time), so time-independent requests keep a stable hash; no re-dispatch mid-program. Implemented in `dmc/ir.py::Context`. | No free-text path to a backend; cache key includes session state (`tests/test_compiler.py`). |
| R13 | MUST give the IR an explicit **ambiguity** channel; `Clarify` when Parse's decision margin is below its calibrated threshold. | Ambiguity suites: clarification precision/recall targets. |
| R13a | MUST render prompts only from explicit IR context: no template may read the clock, environment or host. Mechanism is template-specific: granite-3.1 inserts `strftime_now` only when no system turn is given (an explicit system turn suppresses it); Llama-3.x emits a date in the system header regardless. So every template's date hook (`strftime_now`, `date_string`) is bound to `Context.now`, or to a fixed epoch when the context carries no time (`dmc/runtime.py::template_clock`), and the **rendered prompt bytes are hashed into the receipt** (`Receipt.rendered_prompt_hash`, written by the compiler from the profile result). Phase 0: shifting only the injected date by one day changed 1 of 25 greedy outputs. | "Date-shift" test: the rendered prompt depends only on `Context.now` (`tests/test_runtime.py`); rendered-prompt hash on every receipt (`tests/test_compiler.py`). |
| R14 | MUST measure **paraphrase invariance** of NL→IR on curated paraphrase/format suites and gate releases on it, **always paired with accuracy**: report PIR together with the **consistent-correct rate** (CCR, share of paraphrase groups answered correctly in every variant), because a constant answer scores PIR = 1 (Phase 0: greedy text had PIR 0.8 while answering "no" to 13 of 15 decisions, "yes" to 1, and one unparseable). SHOULD use **paraphrase ensembling** (k paraphrases must agree on the IR hash; disagreement → Clarify) offline and optionally online. | PIR and CCR ≥ targets at stated coverage. |
| R15 | SHOULD render the IR deterministically for user confirmation in high-stakes modes and run IR→NL→IR round-trip checks offline. | Round-trip agreement tracked per release. |

### 2.4 Spec conformance (D4) and decision extraction

| ID | Requirement | Acceptance test |
|---|---|---|
| R16 | MUST generate structured output under **grammar-constrained decoding** compiled from the IR/tool schema with the engine pinned in the lock (llguidance or XGrammar-2; the other engine as offline cross-check); allow a free reasoning span before the constrained span where reasoning is needed (CRANE). | 100% schema validity on golden + fuzzed inputs. |
| R17 | MUST extract decision-bearing values by **scoring candidates in one forward pass**, averaged over a fixed set of candidate orderings, with the label's probability **marginalized over its surface forms** (e.g. "no"/"No"/"NO": scoring only one casing misreads a model that prefers another); closed enums use single-token labels or length-normalized scores; never sample text. | Decision = argmax of order-averaged, case-marginalized scores; margin logged (`dmc/profiles/decide.py::DecideProfile`). |
| R18 | MUST run profile verifiers and allow only **bounded deterministic repair** (`max_repair` in lock; repair prompt is a pure function of the validator error). | Repair terminates; repair outputs are D1-reproducible. |
| R19 | MUST calibrate confidence and set abstention thresholds by **split-conformal calibration** on a frozen set; thresholds versioned with a dead band. | ECE ≤ target; coverage/risk curve in the release report. |

### 2.5 Verification, certification, operations

| ID | Requirement | Acceptance test |
|---|---|---|
| R20 | MUST maintain golden corpora, metamorphic relations and a **synthetic-oracle generator** ("Csmith for prompts"). | Nightly differential runs. |
| R21 | MUST report the parity metric suite every release (§6), including **paraphrase pick-flip rate with dead band** and **adversarial-suffix / keyword-injection flip rate** for Parse. | Release blocked on any gate failure. |
| R22 | MUST run multi-op programs on a durable executor with recorded activities and idempotent tools. | Kill/restart mid-program → identical result. |
| R23 | MUST snapshot external inputs (retrieval, web, tool results) by content hash. | Re-run with snapshot → identical output. |
| R24 | MUST compute the **fragile-routing share** (share of routed decisions with margin < ε, ε calibrated to measured cross-stack noise) for every deterministic-lane request (`dmc/routing.py::RouteTracer`), and use it in confidence/abstention **only if it adds value beyond the decision margin**. | Incremental test: at fixed coverage C₀, parity with (decision margin + fragile share) exceeds parity with decision margin alone, with a CI excluding zero, on a held-out cross-stack set (Phase-0 test: `experiments/03_route_pinning.py`, H4). Three outcomes, pre-registered: CI lower bound > 0 → MUST; CI upper bound < +0.5 points → diagnostic only; otherwise inconclusive → confirmatory test at hub scale (n ≥ 2,000). The fragile share's ε is the measured p95 of cross-stack router-logit noise, not a fixed constant. |
| R25 | MUST ship the **non-optional discovery deliverables** ND1–ND3 (§4.3) with every release, and block release on an unexplained atlas diff. | Atlas diff report attached to release. |
| R27 | MUST record a **routing trace** (per-token top-k per layer: 48 × 8 = 384 B/token on the hub at one byte per index, more if gate weights are stored) plus the token ids it was recorded on and the sampling-mask summary, and support **token-verified** `pin_routes` re-execution: a token mismatch or an exhausted trace is an error, never a silent fall-back to live routing (`dmc/routing.py::RoutePinner`, strict by default). Receipts carry `pinned: full | partial | none`. Production needs an external-route entry point in the fused-MoE kernel path (none exists in vLLM serving; budgeted in P1). | D2-replay parity measured across certified stacks; strict-mode tests (`tests/test_routing.py`). |

---

## 3. System architecture: the compiler pipeline

```
 NL prompt ─► [1 Normalize] ─► [2 Parse NL→IR (dense parser spoke)] ─► [3 Analyze/Typecheck] ─► [4 Canonicalize+Hash]
                                │ intent: scored, dead band            │ CompileError / Clarify       │
                                │ slots: constrained decoding          ▼                              ▼
                                └─ exact cache norm(prompt)+session → IR          result cache H(lock, IR, session, profile, seed)
 [5 Dispatch: static table IR op → profile; max_tier/max_repair/max_escalations constants]
   ─► [6 Codegen: template + grammar + policy + adapter + mask]
   ─► [7 Execute: deterministic lane (D1, routing trace) | fast lane (D0 via recording)]
   ─► [8 Verify / bounded repair] ─► [9 Result + receipt]
 Multi-op IR programs (DAGs) run on [10 Durable executor] with recorded, idempotent tool activities.
```

1. **Normalize** (deterministic code): NFC, quotes, whitespace; time, locale and "today" come from the explicit `Context`, never from the host.
2. **Parse**: a small **dense** parser spoke (Qwen3-4B/8B class, LoRA-tuned, certified separately) emits IR under the IR grammar. Intent = closed-set scoring with an ε dead band; slots = constrained greedy decoding on the deterministic lane; low margin or paraphrase-ensemble disagreement ⇒ `Clarify`. The MoE Parse profile is kept as comparator. Exact-match cache keyed by normalized prompt + session hash; a bounded-error similarity lookup (vCache-style) may only *propose* an IR candidate for re-validation.
3. **Analyze**: type check, reference resolution, tool availability, policy; typed actionable errors.
4. **Canonicalize**: sort commutative collections, normalize units, canonical JSON, IR hash; paraphrases converge here.
5. **Dispatch**: versioned static table (`decide.*`→Decide, `tool.*`→Call, `evaluate.*`→Judge, `create.*`→Create, `analyze.*|research.*`→Reason). Cost-tiering is a deterministic cascade with versioned thresholds, dead bands and **static escalation budgets** (forced-deferral attacks are bounded by construction).
6. **Codegen**: versioned template (explicit system turn), compiled grammar cached by schema hash, decoding policy, adapter id, expert mask, seed schedule.
7. **Execute**: deterministic lane (R5–R11, R26, routing trace) or fast lane (D0 only).
8. **Verify**: schema (always), profile verifiers (executable where possible), bounded repair.
9. **Receipt** (`dmc/store.py::Receipt`): lock hash, IR hash, session hash, rendered-prompt hash, profile version, seed, decision margins, coverage band, fragile-routing share, routing-trace ref, pinning status, sampling-mask summary, normalization mode (Judge), verifier verdicts. (Phase 0 populates lock/IR/session/profile/seed/margins/coverage/verifier; the routing and sampling fields are filled once the deterministic lane emits them.)
10. **Durable executor**: Temporal-class engine; LLM and tool calls are recorded activities; idempotency key = call hash.

### 3.1 IR design (dmc-IR)

Discriminated union on `op` (`decide.classify`, `decide.select`, `tool.call`, `tool.plan`, `evaluate.pairwise`, `evaluate.rubric`, `evaluate.verify`, `create.text`, `analyze.reason`, `analyze.data`, `research.report`, `meta.clarify`); all slots required and typed (nullable instead of optional); closed enums; explicit units; `Context {now, locale, timezone, session_state_hash}` with context minimization (R12); `ambiguities` (must be empty to execute); `seed` and `parent_seed` only on `create.*`; static budgets (`tool.plan.max_steps`) inside the IR so they are hashed. Flat schemas sized to hosted grammar limits (Anthropic: 20 strict tools, 24 optional / 16 union params) so the same IR can target external models. Versioned; migrations are deterministic functions. All twelve ops exist as types in `dmc/ir.py` (v0); only `decide.classify` has an executing profile in Phase 0. Dispatch table: `dmc/compiler.py::DISPATCH_TABLE`.

---

## 4. The Deterministic MoE hub and its spokes

### 4.1 Hub + certified spokes

- **Hub:** one MoE base with per-profile LoRA adapters (LoRA matches full fine-tuning on small/medium SFT and at rank 1 for RL, must cover MLP/MoE layers; "LoRA Without Regret", Sep 2025). One lock family, one atlas, shared serving.
- **Spokes:** separately locked models that win a profile's bake-off. Every spoke carries its own lock, conformance run and super-expert/fragility profile if it is an MoE.
- **Dense students are a first-class outcome.** For Decide, Call and Judge, a dense student distilled on-policy from the hub (dense logprob noise is ~2.4× lower than MoE in the R3 measurements and dense models have no routing flips) competes in the bake-off under the same risk–coverage gate. If students win at equal or better determinism, the MoE hub becomes the **teacher and the generative/reasoning engine**, and the decision executors are dense. This is an acceptable outcome, stated up front.
- **Expert masks are a hypothesis, and Phase 0 evidence is against clean task experts.** Production MoEs route per token at fine granularity; on granite-3.1-1b-a400m, expert usage depends on the task family 5.0–7.6× more than a shuffled-label null (per layer), but the normalized mutual information is at most ~0.04 (middle layers highest). Masks ship only if S2/S5 show a determinism gain within the capability budget on the hub, and they are part of the fuzzed D1 surface (R8, R26).

**Spoke table (v2 candidates):**

| Profile | Spoke candidate | License | Lock / conformance | Atlas scope |
|---|---|---|---|---|
| Parse | Qwen3-4B / Qwen3-8B (dense) + LoRA | Apache-2.0 | own lock; D1 + D3 suites | none (dense) |
| Decide (alt.) | dense student of hub; gpt-oss-safeguard-20b comparator (harmony template) | Apache-2.0 | own lock | fragility profile if MoE |
| Judge (ranking) | Skywork-Reward-V2-**Qwen3-8B** (RB2 78.2, JudgeBench 73.4) | Apache-2.0 | own lock | none (dense) |
| Create | Qwen3.8-27B (dense Gated-DeltaNet hybrid; EQ-Bench Creative v3 Elo 1671) | Apache-2.0 | own lock; **D0 only** (record/replay) until a GDN batch-invariance audit passes under the same go/no-go as the Qwen3.6 spike; bake-off vs a plain-attention writer (Qwen3-32B or the hub) in P3 for a D1 path | none (dense) |
| Reason (`research.report`) | Tongyi DeepResearch-30B-A3B | Apache-2.0 | own lock | super-expert + fragility re-profile required |

The Llama-licensed Skywork-Reward-V2-Llama-3.1-8B (RB2 84.1) is used only with an explicit license exception.

### 4.2 Hub selection and controls

| Role | Model | Status |
|---|---|---|
| **v1 hub** | **Qwen3-30B-A3B-Instruct-2507** (128 experts, top-8, 48 layers, Apache-2.0), text-only, HF revision pinned in the lock | Default. Discovery assets exist for the family (Qwen-Scope SAEs for the Base checkpoint, super-expert list and R3 data for the original/thinking checkpoints, TransformerLens `qwen3_moe`). vLLM's batch-invariance tested list names `Qwen3-30B-A3B` and `Qwen3-30B-A3B-Thinking-2507-FP8`, not Instruct-2507: **P0 must run the batch-invariance validation on the pinned Instruct-2507 checkpoint itself** and re-profile its super experts; Instruct/adapters require dictionary re-validation (S3). |
| P0 spike | Qwen3.6-35B-A3B (256 experts, 8+1, GDN hybrid, vision, MTP) | **Go** only if GDN kernels pass batch-mate, chunk-boundary and prefix warm/cold fuzzing with MTP and vision off, and a GDN dictionary plan is costed. Otherwise a P6+ option. |
| Dense control | Qwen3-32B (BI 22/22 on GB300), Qwen3-8B for cheap runs | Same generation, same attention as the hub. |
| Capability reference | Qwen3.8-27B (the model the user named; dense hybrid) | Capability comparison and Create spoke, not a tax control. |
| Scale-up | DeepSeek-V4-Flash (284B/13B, MIT; hash-routed first 3 layers; bitwise train/infer path in vLLM) | P6+. |
| CPU lab | granite-3.1-1b-a400m (32 experts, top-8), OLMoE-1B-7B | Probes, pilots, vertical slice. |

**MoE determinism tax** is reported as a tuple per pair: (unique@N in default mode, first-divergence index, fragile-token share, D2-fresh parity at C₀).

### 4.3 Deep discovery program (every layer, circuit, router)

Kernels make outputs bitwise-reproducible; discovery makes the MoE **legible and steerable**. Its product-facing jobs: measure fragility, feed it into abstention, pin routes for replay, find each profile's decision components, and test levers that widen margins.

**Non-optional deliverables (ship every release, R25):**
- **ND1 — Router atlas & Fragility Map** for the hub *and each adapter*: per-layer margin distributions, fragile-token share, expert load by profile/domain, co-activation, cross-layer route paths, super-expert locations (never ablate: pruning 3 of 6,144 experts collapses Qwen3-30B-A3B Math-500 88→1.4).
- **ND2 — Fragility-aware confidence:** fragile-routing share feeds the abstention signal (R24) only if it passes the incremental test: at the pre-registered coverage floor, (decision margin + fragile share) beats decision margin alone with a CI excluding zero on a held-out cross-stack set. If it does not, ND2 ships as a reported diagnostic (the share and the margin→flip AUC per release) rather than a confidence input.
- **ND3 — Atlas diff** per release: routing-path drift, margin-distribution shift, super-expert check; unexplained drift blocks release.

**Stages:**

| Stage | What | Artifacts | Success metric |
|---|---|---|---|
| S1 Instrumentation | Hooks on routers, attention, MoE blocks (nnsight / TransformerLens 3.x `qwen3_moe`); routing traces (R27). | Trace schema; receipts. | 100% layers; overhead measured. |
| S2 Router & expert atlas | ND1; perturbation flip rates (dtype, batch, adapter mix, TP, paraphrase); margin→flip AUC; task-specificity (NMI vs shuffled null). | ND1, ND3 baselines. | Fragility map reproducible across runs; flips explained by margins. |
| S3 Dictionaries | Qwen-Scope residual SAEs (32K/128K, all 48 layers); **re-validate (or re-fit) after Instruct and after each adapter**. No CLTs in v1 (no 30B-MoE budget exists); circuits use EAP-IG over coarse nodes {heads, experts, shared expert, router logits}. | Feature catalog per profile. | Fidelity thresholds; features validated by intervention. |
| S4 Behaviour circuits | Contrastive sets per profile (yes↔no, tool A↔B, prefer A↔B); EAP-IG localization; expert-level attribution. | Circuit cards. | Patching recovers ≥ X% of logit diff; stable across seeds. |
| S5 Interventions (pre-registered) | Margin regularization; StableMoE router freeze/distill; expert masks/router bias; decision-feature steering. Controls: size-matched random-expert nulls, prompting baseline, off-target ≤ ⅓ on-target, capability budget. | Intervention cards. | D2-fresh / D3 gains within capability budget, or a negative result that removes the lever. |

### 4.4 Router hardening toolbox

Inference-time:
- **FP32 router GEMM + stable tie-break** (R7, `dmc/routing.py::harden_routers`).
- **Routing traces + token-verified `pin_routes` re-execution** (R27): removes the discrete amplifier when re-executing a recorded result on another stack (D2-replay). A fresh request on a non-reference stack has no recorded route for its own tokens, so D2-fresh needs the levers below.
- **Pinned prefix routes (D2-fresh lever, pre-registered for P1).** For Decide/Call/Judge most routed tokens are the versioned template prefix (system turn, policy text, few-shot exemplars), identical across requests and known at lock time. Their routes are recorded once on the reference stack and shipped in the lock like a precompiled header; only user-content and generated tokens route live. Measure: share of fragile routing decisions that fall in the prefix, and D2-fresh parity with vs without prefix pinning at C₀. Needs the same engine entry point as R27; the KV cache for the prefix must be computed under the pinned routes.
- **Stack-independent router (D2-fresh lever, pre-registered for P5).** A router whose input is exact on every stack (the token embedding, i.e. a table lookup; or a hash of the token id as in DeepSeek-V4's first three layers) yields identical routes on first execution everywhere. StableMoE distils routing into such a lightweight, frozen router. Pre-registration: per-layer replacement starting from the late layers (which tolerate intervention best), quality budget ≤ 1% held-out loss and ≤ 2 points on profile goldens, success = D2-fresh parity gain at C₀. This is the only lever that removes the routing amplifier for fresh requests rather than abstaining around it.
- **Fragility-aware abstention** (R24, ND2).
- **Profile expert masks / router biases** (hypothesis; always include super-experts and shared experts).
- **Cross-SKU numeric path** (D2→D1 upgrade): power-of-two INT8 scales made CUTLASS and Triton kernels agree bitwise at +0.3–0.5% perplexity (2609.00363); ReproFuse fused-upcast GEMMs are batch-invariant only for M ≤ 64.

Training-time (all pre-registered in S5):
- **Router margin regularization** — hinge `max(0, γ − (s_k − s_{k+1}))` on router logits with router row norms held fixed (so margins cannot be widened by scaling). **P0 pilot**, pre-registered in `experiments/04_margin_regularization_prereg.md` (amendment 1): λ = 0 control arm; grid γ ∈ {0.05, 0.1, 0.2} × λ ∈ {1, 5}; sole primary = BF16 4-vs-3-thread routing-flip rate with hardened routers, must be ≤ ½ of the control's; guards on held-out CE (≤ +1%), expert-load entropy (≥ 95% of original; granite has no load-balance loss), Decide agreement with the original model (≥ 90% probe, ≥ 95% oracle) and oracle accuracy (≥ −2 points); survives only if an arm passes on two seeds. Otherwise removed from P5.
- **StableMoE-style router freeze/distillation** during profile fine-tuning (see stack-independent router above).
- **R3 routing replay** in any RL stage (<3% overhead; prevents MoE RL collapse).
- **Batch-invariant train/infer numerics** — stretch only (see §4.5).

### 4.5 Determinism-hardening training (per profile)

| Method | Applies to | Effect (literature) | Guardrail |
|---|---|---|---|
| Invariance/consistency training (BCT, ACT, Flip-Flop Consistency) on paraphrase + format augmentation | Parse, Decide, Call, Judge | +11.6% cross-variant agreement (F²C); −86% biased reasoning (BCT) | Held-out paraphrase families |
| Mode-seeking on-policy distillation; majority-vote targets (ScPO/TTRL) | Decide, Call, Judge, Reason final answers; dense students | Collapses onto one answer | Entropy–capability coupling: monitor pass@k; never on Create |
| Calibration + abstention (RLCR Brier reward; ~1k graded examples) | Parse, Decide, Judge, Reason | Explicit abstention instead of guessing | ECE, abstention precision |
| Router margin regularization / StableMoE | Hub adapters (not Create) | Wider margins (hypothesis) | P0 pilot gate, capability budget |
| RL baseline: R3 + FP32 router on rollout and trainer + FP16 or TIS/MIS correction with sampling mask exported | Any RL stage | Stable MoE RL without bitwise kernels | KL monitoring |
| Bitwise RL (batch-invariant backward passes for grouped GEMM/top-k/combine) | Stretch, or at DeepSeek-V4-Flash scale-up where it ships | KL = 0 | Separate engineering estimate |

---

## 5. The six profiles

Each profile = `(adapter, expert mask?, template, grammar, decoding policy, verifier, thresholds)`, locked and versioned. All decision-bearing values are scored (R17); all thresholds are conformal (R19).

### 5.0 Parse (front end) — strictest D3 targets
Dense parser spoke; closed intent scoring with dead band; constrained slot filling; paraphrase ensembling; clarification; invariance + calibration training; certified on paraphrase and adversarial-flip suites before any other profile.

### 5.1 Decide
- Optional greedy reasoning span → **order-averaged label scoring** → conformal abstention band. Thinking off by default (reasoning-tuned models abstain ~24% less, AbstentionBench).
- Contenders: hub adapter; **dense student**; linear-probe head on frozen hub states; gpt-oss-safeguard-20b; Bradley-Terry scalar head.
- Training: classification SFT + calibration (~1k graded examples) + abstention + paraphrase invariance + label-permutation augmentation.
- Targets: D1 TARa = 1; D2-fresh parity ≥ 99.5% at coverage ≥ C₀ = 0.85; PIR ≥ 97% with CCR reported, at C₀; ECE ≤ 0.03.

### 5.2 Call (function calling only)
- (1) **Compile from IR first** (all arguments present as typed slots ⇒ deterministic code, no model call); (2) tool selection = scoring over {tools, `no_tool`, `clarify`}; (3) arguments under strict JSON-Schema grammar, greedy; (4) RFC 8785 canonical call; (5) validation; (6) idempotency key = call hash; recorded activity.
- Data: APIGen-MT / ToolACE / xLAM-style trajectories + IR-derived synthetic calls. Benchmarks: τ²-bench pass^k (every model loses 10–30 points pass^1→pass^4), BFCL v4 with care (Epoch AI: 48% of sampled tasks defective).
- Fixed interface is the main lever: BFCL format-sensitivity deltas reach 81.5 points for small specialists vs 8.0 for Qwen3-235B. Call stays narrow; planning goes to `tool.plan` + durable executor. Dense-student arm in the bake-off.

### 5.3 Judge (contrastive cross-evaluation)
- Executable verifiers first; scalar RM (Skywork-Reward-V2-Qwen3-8B) for ranking; generative verifier profile on the hub (maj@k over fixed seeds) for out-of-domain cases.
- **Both-order** pairwise scoring is a MUST (position bias grows with the quality gap); verdict = mean log-odds across orders; tie band.
- **Style normalization is conditional on the criterion type** (content criteria: render to canonical plain form, since style/markdown bias dominates 2026 audits at 0.10–0.76 and SOTA RMs fall to 46.6% under style interference; formatting/instruction-following criteria: no normalization). Mode recorded in the receipt.
- Rubric scores = expected value of the score-token distribution. Cross-family panels with agreement statistics. Dense-student arm.

### 5.4 Create
- Output is a pure function of `(IR, seed)`; default seed = H(IR); **`parent_seed` inheritance** so a revision keeps the base draft's seed schedule (documented: any IR edit without `parent_seed` yields an unrelated draft).
- Plan → draft → critique → revise; N candidates per stage with seeds `H(IR, stage, i)` on counter-based RNG; ranking by a locked RM + Judge with stable tie-break. Diversity from an explicit enumerated latent (Verbalized Sampling recovers 1.6–2.1×), recorded in the IR.
- Spoke: Qwen3.8-27B (best small open writer, Apache-2.0), **D0 only** (record/replay) until its Gated-DeltaNet kernels pass a batch-invariance audit; the P3 bake-off includes a plain-attention writer (hub or Qwen3-32B) that can run on the D1 lane. Kimi K3 / GLM-5.3 are stronger but custom-licensed. No mode-seeking on this profile.

### 5.5 Reason
- Thinking-mode generation on the deterministic lane (BF16 diverges ~token 65 vs ~1,826 in FP32 across stacks per LayerCast; long chains need D1 kernels, and FP32 accumulation is an option for D2). Answer-level determinism via K fixed-seed chains + majority vote with deterministic tie-break; final answer by grammar. Pinned data sandbox; snapshotted retrieval (R23).
- Verifiers: code execution, numeric checks, PRM/Judge on steps, citation-to-snapshot checks.
- Hub reasoning adapter; Tongyi DeepResearch-30B-A3B spoke for `research.report` (own super-expert/fragility profile; no assumed atlas transfer); RLVR per §4.5 baseline.

---

## 6. Verification and certification

**Metric suite (per profile, per lock, per release):**
1. *D1:* TARr@N, unique@N (N = 1000 on a sample), first-divergence index; under randomized load, **adapter mix**, **expert-mask on/off**, chunk sizes, phase and cache variations.
2. *D2:* **risk–coverage curves** of cross-toolchain parity (D2-fresh, each stack's decisions ranked by **that stack's own** margin, since a fresh execution has no other) and pinned-route parity (D2-replay, ranked by the reference margin carried in the receipt), with CIs; fragile-routing share.
3. *D3:* PIR on paraphrase/format families at stated coverage; paraphrase pick-flip rate with dead band; adversarial-suffix / keyword-injection flip rate (Parse).
4. *D4:* schema validity, oracle pass rate, pass^k, ECE, abstention precision/recall.
5. *Release parity:* negative-flip rate and changed-output rate vs previous release; **atlas diff** (ND3).

**Conformance matrix (D1):** dimensions {batch-mates 0..k} × {adapters co-resident 0..3, mixed; same-adapter batch-mates for fallback 1} × {expert masks on/off} × {chunk size} × {prefill vs decode} × {prefix cache cold/warm, if enabled} × {preemption} × {spec/MTP on/off, if enabled} × {async scheduling, if enabled}. The full 9-way product is infeasible, so: a **pairwise covering array** over all dimensions on the full golden set, the **full product on a small golden subset** (≈20 prompts), `unique@1000` only in the base configuration, and `unique@20–50` per covering-array cell. Every executed cell: identical logits at every position.

**Methods:** golden corpora per profile; metamorphic relations; differential testing across toolchains and releases; batch-mate and adapter-mix fuzzing; long-generation tests; synthetic-oracle generator (`dmc/oracle.py`: families of paraphrased decision prompts with known answers, a pure function of the seed); MMD-style endpoint-identity tests.

**Coverage floors (pre-registered product requirements, fixed before any P0 measurement):** Parse C₀ = 0.90, Decide C₀ = 0.85, Call C₀ = 0.90, Judge C₀ = 0.80, Reason C₀ = 0.80. If a profile reaches its parity target only below its floor, that is a gate failure and triggers the dense-student arm (and, for the hub, the stack-independent-router lever); the floor is not lowered.

**Release gates:** D1 `unique@1000 = 1` on 100% of sampled golden prompts in the base configuration and identical logits in every executed matrix cell; D2-fresh parity ≥ 99.5% at C₀ (Decide/Call/Judge), ≥ 98% at C₀ (Reason final answers), with the full risk–coverage curve and CIs reported; D3 PIR ≥ 95% (Parse), ≥ 97% (Decide) at C₀, **always reported with accuracy and CCR**; D4 schema validity 100%; negative-flip rate ≤ 0.5%; no unexplained atlas diff.

---

## 7. Roadmap

| Phase | Weeks | Scope | Exit gate |
|---|---|---|---|
| **P0 Foundations & de-risking** | 0–6 | *Done on CPU (this repo):* lockfile, hashing, metrics, IR v0 (12 op types), store with replay, receipts, R7 router hardening, token-verified route tracing/pinning, synthetic-oracle generator, `decide.classify` vertical slice; router atlas on granite (§10). *Running / pending on CPU:* conformance (02), route-pinning pilot (03), margin-regularization pilot (04), all pre-registered. *Pending GPU:* Qwen3-30B-A3B-Instruct-2507 vs Qwen3-32B/8B, default vs batch-invariant vLLM; Qwen3.6-35B-A3B go/no-go spike. | Baseline report; hub confirmed; tax tuple measured; pilot outcomes recorded against their pre-registrations. |
| **P1 Deterministic runtime** | 3–10 | Deterministic lane: vLLM BI + FP32 router GEMM + stable tie-break + static scales + pinned MoE kernel configs; full conformance matrix (covering array) incl. **adapter mix, same-adapter batch-mates and masks**; counter-based RNG; routing traces; **external-route entry point in vLLM's fused-MoE path** for R27 and pinned prefix routes (engineering estimate in P1 planning; until it lands, D2-replay is certified on the HF reference stack); selective-determinism evaluation (LLM-42 / MarginGate). | D1 on hub + ≥3 co-resident adapters on one SKU, or fallback in lock. |
| **P2 Compiler front end** | 6–14 | IR v1; dense Parse spoke; constrained decoding; clarification; paraphrase ensembling; caches; round-trip checks; adversarial-flip suite. | Parse PIR, flip and clarification targets met. |
| **P3 Profiles v1 + bake-offs** | 10–20 | Decide, Call, Judge, Create, Reason; spokes locked; **dense-student arms**; durable executor. | Each profile meets D1/D4 gates and benchmark floors; bake-off decisions recorded. |
| **P4 Discovery program** | 6–24 | S1–S5 on the shipping hub; ND1–ND3 operational; dictionaries re-validated per adapter. | ND1–ND3 in the release pipeline (mandatory); S5 effect sizes or negative results recorded. |
| **P5 Hardening training** | 14–32 | Invariance, mode-seeking (non-Create), calibration/abstention; margin regularization only if the P0 pilot survived; stack-independent router (pre-registered); RL baseline R3 + FP32 router + TIS. | D2-fresh/D3 improve at fixed C₀ without capability regression beyond budget. |
| **P6 Certification & ops** | 24–36 | Release process; differential testing at scale; SKU certification matrix; **formal batch-invariance analysis of MoE kernels** (grouped GEMM, top-k, permute/combine): extend Vosti's Triton analyzer if its code is released, otherwise budget a re-implementation (the notes do not confirm a release); drift monitoring. | First certified release with D0–D4 report and atlas diff. |

**Indicative compute:** P0–P2: one 8×H100/H200 node. P4: +1 node; residual-SAE work streams activations rather than caching (~20 TB per 100M tokens if cached naively). P5: 2–4 nodes for LoRA/RL on the hub. Fallback memory: merged per-profile checkpoints cost ~61 GB BF16 each on Qwen3-30B-A3B (six profiles fit one 8×80 GB node at one per GPU). Engine work: external-route fused-MoE entry point (P1, estimate pending); formal MoE-kernel analysis ~1 engineer-quarter if Vosti's analyzer is available, more if re-implemented.

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| D1 overhead (1.6–2.1× naive) | Medium | Selective determinism; CoRun-style fixed-shape padding; deterministic lane only for traffic that needs it. |
| Undiscovered MoE / multi-LoRA kernel leaks | High | Own conformance matrix incl. adapter mix; batch-by-adapter / merged-checkpoint fallback; Vosti extension in P6. |
| Margin widening fails or costs capability | Medium–High | P0 pilot with pre-registered gate; D2-fresh falls back to abstention + dense students; D2-replay via pinning is unaffected. |
| D3 targets unreachable for open-ended domains | High | Narrow IR to closed domains first; clarification over guessing; report PIR at stated coverage. |
| Mode-seeking reduces capability | Medium | Capability budget; never on Create; pass@k monitoring. |
| Interpretability yields no levers | Medium | ND1–ND3 ship regardless; S5 negative results remove levers rather than block release. |
| Router/expert-level attacks (GateBreaker, BadMoE), adversarial routing suffixes, forced deferral | Medium | Adversarial-flip suite (R21); static escalation budgets; expert-integrity checks in atlas diff. |
| Hosted/closed models needed | Medium | D0 only (record/replay); flagged in receipts. |
| License creep via spokes | Medium | Spoke table with licenses; exceptions explicit. |
| Base-model obsolescence (Qwen3-30B-A3B is a 2025 model) | Medium | P0 Qwen3.6 spike; DeepSeek-V4-Flash scale-up; methods are hub-agnostic. |

---

## 9. What exists in this repo now

- `dmc/canonical.py` (RFC 8785-style hashing), `dmc/lockfile.py` (toolchain identity), `dmc/ir.py` (typed IR v0: twelve op types, `Context` with `session_state_hash`, `parent_seed`), `dmc/store.py` (content-addressed replay; a divergent recompute raises `NondeterminismError`; `Receipt` fields per §3 step 9), `dmc/metrics.py` (TARr/TARa, unique@N, first divergence, PIR, flips, ECE), `dmc/moe_probe.py` (router margins, flips, stable top-k, AUC, task specificity), `dmc/routing.py` (R7 `harden_routers`; token-verified `RouteTracer`/`RoutePinner`; fragile-routing share), `dmc/runtime.py` (deterministic CPU backend: R7 router hardening **on by default** and recorded in the lock, explicit system turn, template clock bound to `Context.now`, one request per pass, counter-based sampling; table backend for tests), `dmc/http_backend.py` (OpenAI-compatible backend for GPU Phase 0; its `score` relies on the server returning echoed prompt log-probs, which must be verified under the batch-invariant flag before use), `dmc/profiles/decide.py` (order-averaged label scoring, marginalized over case and leading-space surface forms by default, with abstention and a rendered-prompt hash), `dmc/compiler.py` (pipeline, rule parser with context minimization, `DISPATCH_TABLE`), `dmc/conformance.py` (variation harness), `dmc/oracle.py` (synthetic-oracle generator).
- `tests/`: unit tests for all of the above, plus `tests/test_plan_citations.py`, which fails if this document cites a `dmc/...` file or symbol that does not exist.
- `experiments/`: CPU experiments on granite-3.1-1b-a400m with pre-registrations; JSON results in `experiments/results/`. `experiments/gpu/`: engine-conformance script and runbook for GPU Phase 0 (not run here).

## 10. Phase-0 evidence (CPU, granite-3.1-1b-a400m-instruct)

| Experiment | Status | Key numbers |
|---|---|---|
| 00 quick probe (`experiments/00_granite_router_probe_quick.md`) | done | Single prompt; stock BF16 router (R7-violating). Superseded by 01. |
| 01 router atlas (`experiments/results/01_router_atlas.json`), 75 prompts, stock routers | done | Margin < 0.01: 7.8% of routing decisions (< 0.001: 1.0%); FP32-vs-BF16 flips 9.0% of decisions and touches 79% of tokens, by layer 2–4% early → 16–17% late; margin→flip AUC 0.87; batch of 4 vs alone: 0 routing flips, router-logit drift ≤ 3.4e-5, but 24% of greedy 24-token continuations differ; same-shape repeat: unique@3 = 1; task NMI 0.013–0.040 vs shuffled null p95 ≤ 0.0095; Decide PIR: label scoring 0.40 (accuracy 11/15) vs greedy text 0.80 (accuracy 10/15, "no" on 14/15); flipped label-scored items all had margins 0.23–0.54, so an abstain band of ~0.6 gives PIR 1.0 at 67% coverage, with 8/10 decided items correct and CCR 4/5 groups; one-day date shift changed 1/25 greedy outputs. |
| Proxy calibration, informal (treatment-blind, 10 prompts; no artifact) | superseded by 05 | R7-hardened routers, BF16, 4 vs 1/2/3 threads: max abs Δlogit 3.8–4.8, routing flips 3.5–4.6%. Motivated the thread-count proxy; replaced by `experiments/05_proxy_calibration.py`. |
| 05 proxy calibration (`experiments/results/05_proxy_calibration.json`), 75 prompts, hardened routers | done | REF = BF16/4 threads vs ALT = BF16/3 threads: router-logit |Δ| median 0.0048, p95 0.033 (→ ε_H4 = 0.033); per-position max output-logit |Δ| median 0.63, p95 2.25; per-position KL mean 0.0205 (median 0.0028, p95 0.096); routing flips 4.7%. REF vs STRESS (FP32): KL mean 0.0205, flips 5.8%. **Regime: stress** (mean KL 13.7× the published cross-engine 1.5e-3), so emulated-BF16 thread variation on this CPU is about as disruptive as a precision change; CPU pilot results can remove levers or validate mechanisms but cannot set production C₀ (pre-registered). BF16 forward ≈ 1.0–1.1 s vs FP32 0.62 s (no native BF16 on this CPU). |
| 02 CPU conformance (`experiments/results/02_cpu_conformance.json`), 25 generations + 30 decisions | done | Started before R7 hardening became the backend default: FP32 rows are R7-compliant (an FP32 model's router GEMM is FP32), BF16 rows use the stock BF16 router GEMM (R7-violating); decisions used the case-only scorer. Same-shape repeats: greedy 32-token text identical in 25/25 prompts × 3 runs (D1 on a fixed stack). FP32 at 4/2/1 threads: text 100% and decisions 100% identical (FP32 CPU kernels here are thread-count invariant). FP32 vs BF16: greedy text identical in only 68% of prompts, but decisions agree on 96.7% (29/30); the single flip had FP32 margin 0.026 (median 1.02), and an abstain band at margin 1.0 gives 100% parity at 47% coverage. Decision extraction by scoring (R17) is far more robust to a numerics change than generated text. |
| 03 route pinning, D2-fresh vs D2-replay (pre-registered + amendment 1) | running | — |
| 04 margin regularization (pre-registered, amendments 1–2) | queued | — |

Caveat: the 01 FP32-vs-BF16 numbers were measured with granite's stock router, which runs the router GEMM in BF16 (R7-violating); they overstate what the deterministic lane would see. The hardened-router numbers above are the relevant baseline.

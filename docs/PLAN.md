# Deterministic Model Compiler (DMC): Architecture Requirements and Implementation Plan

**Version:** v1 (pre-review draft) · **Date:** 2026-10-21 · **Status:** under adversarial review

> Analogy: people trust a compiler because (a) the same source and the same toolchain give the same binary, (b) the binary means what the source means, and (c) ill-formed source is rejected rather than guessed at. DMC aims to give a prompt the same three guarantees.

Evidence for every claim below lives in `research_notes/Deterministic MoE model compiler/` (01–07) and the synthesized report in `reports/`. Measurements we made ourselves are in `experiments/`.

---

## 0. Executive summary

1. **"Deterministic" is not one property; it is a stack of five contracts** (D0 replay → D1 bitwise → D2 decision parity across toolchains → D3 paraphrase invariance → D4 spec conformance). D0, D1 and D4 can be *guaranteed* today with engineering. D2 and D3 can only be *certified statistically*, and that is where the research work lies.
2. **Bitwise determinism (D1) is now an engineering cost, not a research problem.** Batch-invariant kernels (Thinking Machines, Sep 2025) took Qwen3-235B-A22B from 80 unique completions per 1,000 temperature-0 runs to 1, at 1.3–2.1× cost. vLLM `VLLM_BATCH_INVARIANT=1` validates several MoE models, but has open MoE leaks (vLLM #57016: per-batch FP8 activation scales, token-count-dependent Triton configs). Formal checking (Vosti, Sep 2026) found leaks in both vLLM and SGLang deterministic modes. We must own a conformance suite.
3. **MoE routing is the MoE-specific fragility.** Top-k routing is an argmax over continuous logits. On Qwen3-30B-A3B ~10% of router decisions differ between engines and 94% of tokens differ in at least one layer. Our own CPU probe on Granite-3.1-1B-A400M: 11% of router decisions have a top-k/(k+1) gap < 0.01; fp32→bf16 changes 4.3% of expert sets and touches 69% of tokens. The deep "discovery" program exists to measure and then *widen* these margins, and to locate the components that drive each task's decisions.
4. **The biggest source of output variance users actually see is not numerics; it is the prompt.** Formatting alone swings accuracy by up to 76 points (FormatSpread). So the architecture is a **compiler pipeline**: NL → typed IR (a DSL) → static dispatch → specialist execution profile → verifier, with ambiguity treated as a compile error (ask) rather than undefined behaviour (guess).
5. **Specialists = execution profiles on one deterministic MoE base** (adapter + expert mask/router bias + decoding policy + grammar + verifier), plus best-in-class external specialists where they win. Five profiles: **Decide, Call (function calling), Judge (contrastive cross-evaluation), Create, Reason**. Task routing is **static dispatch on IR type**, not a learned per-query router (independent 2026 benchmarks: learned routers rarely beat a per-task-type table and flip on paraphrase).
6. **Base model:** "Qwen 3.8 27B" exists but is a **dense** hybrid model, not an MoE. Recommended MoE bases: **Qwen3-30B-A3B** (most-studied: routing-mismatch data, Qwen-Scope SAEs for all 48 layers, known super-experts, vLLM-validated) for the discovery program, and **Qwen3.6-35B-A3B** (Apache-2.0, 256 experts, 8 routed + 1 shared) as the production candidate, with Qwen3.8-27B (dense) as the control that measures the "MoE determinism tax".

---

## 1. The determinism contract (what we promise, precisely)

| Level | Name | Promise | Mechanism | Guarantee type |
|---|---|---|---|---|
| **D0** | Replay | Same request id / same canonical IR + lock → byte-identical response | Content-addressed result store, record/replay | Hard (by construction) |
| **D1** | Bitwise recompute | Same canonical IR + same **lockfile** (weights, tokenizer, template, engine, kernels, flags, SKU, TP/EP topology) → byte-identical output, under any load, batch mix, chunking or cache state | Batch/position-invariant kernels, fixed topology, counter-based RNG, conformance suite | Hard on certified stacks (tested + Vosti-style contract) |
| **D2** | Decision parity | Across certified toolchains (different SKU / engine version), the **decision-bearing fields** (label, tool call, verdict, final answer, IR) agree; prose may differ | Margin-widening training, abstention on low-margin decisions, decision extraction by scoring not sampling | Statistical (certified rate, e.g. ≥99.5%) |
| **D3** | Semantic invariance | Meaning-preserving paraphrases / reformattings → same IR → same output | Canonical IR, parser invariance training, clarification on ambiguity | Statistical (certified PIR on paraphrase suites) |
| **D4** | Spec conformance | Output satisfies its typed contract; where an executable oracle exists, it satisfies the oracle | Grammar-constrained decoding, validators, executable verifiers, bounded deterministic repair | Hard for syntax/schema; oracle-dependent for semantics |
| **D⊥** | Defined failure | Ill-formed or ambiguous input yields a typed `CompileError` / `Clarify` / `Abstain`, never a silent guess | Ambiguity slot in IR, calibrated abstention | Hard (behavioural contract), quality measured |

Non-goals (stated so that nobody is misled):
- **No promise of cross-vendor bitwise equality** (NVIDIA vs AMD vs TPU): no published demonstration exists for full LLM inference as of Oct 2026. Cross-SKU within NVIDIA is possible only by owning every kernel (2609.25624 got linear layers bitwise across Ampere/Ada/Hopper; attention still leaked). We promise D2 across vendors, not D1.
- **No promise that the output is "correct"** beyond D4 oracles. Determinism makes errors reproducible; it does not remove them.
- **Creative output is deterministic per (IR, seed)**, not unique per prompt "meaning". Diversity is an explicit input, not an accident.

---

## 2. Architectural requirements for consistent output parity

Each requirement has an ID, a rationale and an acceptance test. "MUST" items gate release.

### 2.1 Identity and reproducibility of the toolchain

| ID | Requirement | Rationale | Acceptance test |
|---|---|---|---|
| R1 | MUST pin every output-affecting artifact in a **lockfile** (weights hash, tokenizer hash, chat-template hash, adapter hash, engine+version, kernel set, determinism flags/env, SKU, TP/EP/PP/DP sizes, IR schema version, grammar hash, decoding policy). | Any of these changes bits. Silent API updates broke "same prompt" behaviour (RETAIN, Ma et al.). | Lock hash recorded on every response; replays refuse lock mismatch. |
| R2 | MUST canonicalize every hashed artifact (RFC 8785-style JSON, NFC text). | Equal values must give equal keys. | Property tests: permuted keys / equal floats → equal hash. |
| R3 | MUST store every response content-addressed by `H(lock, canonical IR, profile version, seed)` and support replay. | D0; also the only way to make hosted/non-pinned models reproducible (OpenAI `seed` is best-effort; Anthropic T=0 is not deterministic). | Replay returns byte-identical result; cache key collision tests. |
| R4 | MUST version and shadow-test every routing table, threshold, prompt template and grammar as code. | Routers and thresholds are part of the program (vLLM Semantic Router v0.3 shadow/activate/revert). | Release diff report lists every changed decision on the golden set. |

### 2.2 Numerical determinism of execution (D1)

| ID | Requirement | Rationale | Acceptance test |
|---|---|---|---|
| R5 | MUST run deterministic-lane traffic on **batch- and position-invariant kernels** (matmul, RMSNorm, attention incl. split-KV with fixed split *size*, MoE grouped GEMM, sampler). | The dominant cause of T=0 variance is batch-dependent reduction order. | `unique@1000 == 1` per golden prompt under randomized concurrent load. |
| R6 | MUST make execution invariant to chunked-prefill boundaries, prefill-vs-decode path, prefix-cache hit/miss and preemption/recompute — or disable those features in the deterministic lane. | Vosti found production deterministic modes leak on prefill/decode and sliding-window cases. | Conformance suite: same request under each variation → identical logits at every position. |
| R7 | MUST compute router logits in FP32 with fixed reduction order and a **stable top-k tie-break** (lowest expert index). | Router is a discrete argmax; ties and ULP noise flip experts. vLLM's DeepSeek-V4 PR adds exactly this. | Router-decision equality across load variations; exact-tie unit tests. |
| R8 | MUST NOT use per-batch dynamic quantization scales or token-count-dependent kernel autotuning in the deterministic lane (static scales or BF16; pinned kernel configs). | Root causes of vLLM #57016 MoE leaks. | Batch-mate fuzzing on MoE reproducers (granite-1b-a400m, Qwen3-30B-A3B). |
| R9 | MUST use a **counter-based RNG keyed by (request seed, position)** for any sampling (e.g. Philox / seeded Gumbel). | A global RNG stream couples requests that share a batch. | Same seed → same sample regardless of batch-mates. |
| R10 | MUST fix the parallel topology (TP/EP/PP) per certified lock, or use TP-invariant kernels (TBIK). | All-reduce order changes with TP size. | Topology is part of the lock; cross-topology runs flagged as D2-only. |
| R11 | SHOULD offer **selective determinism** (per-request determinism SLA via verify-and-rollback, LLM-42 / MarginGate) so throughput traffic does not pay the tax. | Server-wide modes cost 24–36% throughput for everyone; mixing one deterministic request into 10 others cost ~56% in SGLang. | Deterministic requests meet R5 while co-scheduled with fast-lane traffic. |

### 2.3 Semantic determinism (D3) and defined failure (D⊥)

| ID | Requirement | Rationale | Acceptance test |
|---|---|---|---|
| R12 | MUST compile every request to a **closed, typed IR** (discriminated union of ops with required, typed slots) before any task model sees it. | The IR is the "source code"; equal IR ⇒ equal downstream behaviour by construction. | Every executed request has a validated IR; no free-text path to backends. |
| R13 | MUST give the IR an explicit **ambiguity / clarification** channel and emit `Clarify` when the parser's decision margin is below a calibrated threshold. | Ambiguity is irreducible variance (GPT-4 disambiguates correctly 32% of the time on AmbiEnt). Guessing is undefined behaviour. | Ambiguity suites (AmbiEnt/AMBROSIA-style): clarification precision/recall targets. |
| R14 | MUST measure **paraphrase invariance** of the NL→IR step on curated paraphrase/format suites and gate releases on it. | NL→IR is the only stochastic step left once D1 holds. | PIR ≥ target per domain; regressions block release. |
| R15 | SHOULD provide a deterministic, non-LLM **rendering of the IR** for user confirmation in high-stakes modes (TypeChat pattern) and run IR→NL→IR round-trip checks offline. | Confirmation turns semantic risk into an explicit contract. | Round-trip agreement rate tracked per release. |

### 2.4 Spec conformance (D4) and decision extraction

| ID | Requirement | Rationale | Acceptance test |
|---|---|---|---|
| R16 | MUST generate every structured output under **grammar-constrained decoding** compiled from the IR/tool schema (XGrammar-2 or llguidance), allowing a free reasoning span before the constrained span where reasoning is needed (CRANE). | Guarantees syntax/schema; constrained decoding does not hurt accuracy when reasoning space is preserved. | 100% schema validity on golden + fuzzed inputs. |
| R17 | MUST extract decision-bearing values (labels, verdicts, tool names, scores) by **scoring candidates in one forward pass** (log-prob over the closed set), not by sampling text. | Removes the sampling path and the long-generation divergence; yields a margin usable for abstention. | Decision equals argmax of scored candidates; margin logged. |
| R18 | MUST run profile-specific **validators/verifiers** and allow only **bounded, deterministic repair** (repair prompt is a pure function of the validator error). | TypeChat / Fowler validate-and-repair loop; keeps repair inside the contract. | Repair loop terminates; repair outputs are D1-reproducible. |
| R19 | MUST calibrate confidence (ECE target) and route low-confidence decisions to `Abstain` / escalation with **versioned thresholds and a dead band**. | Escalation is deterministic only if thresholds are constants and signals are computed on deterministic inference. | ECE ≤ target; threshold changes appear in release diff. |

### 2.5 Verification, certification and operations

| ID | Requirement | Rationale | Acceptance test |
|---|---|---|---|
| R20 | MUST maintain per-profile **golden corpora**, metamorphic relations and a **synthetic-oracle generator** ("Csmith for prompts": programmatic tasks with known answers). | Compiler trust came from differential + random testing (Csmith found 325+ bugs). No LLM equivalent exists yet. | Nightly differential runs; new failures triaged. |
| R21 | MUST report the **parity metric suite** on every release: TARr@N, unique@N, first-divergence index (D1); cross-toolchain TARa (D2); PIR, flip rate, POSIX (D3); schema validity, oracle pass, ECE, abstention quality (D4); negative-flip rate vs previous release (MUSCLE). | Measurement is the certification. | Release blocked on any gate failure. |
| R22 | MUST run multi-step programs on a **durable executor** where LLM and tool calls are recorded activities and tools are idempotent (idempotency key = call hash). | Agent loops are otherwise non-replayable (LangGraph replay re-runs LLM calls). | Kill/restart mid-program → identical final result. |
| R23 | MUST snapshot external inputs (retrieved documents, web pages, tool results) by content hash. | Deep-research outputs depend on a moving world; determinism requires a frozen input. | Re-run with snapshot → identical output. |
| R24 | MUST log router telemetry (per-layer top-k, margins) for sampled deterministic-lane traffic. | Routing fragility is the leading indicator of D2/D3 breakage in MoE. | Fragile-token share tracked; alerts on drift. |

---

## 3. System architecture: the compiler pipeline

```
 NL prompt ──► [1 Normalize] ──► [2 Parse NL→IR] ──► [3 Analyze/Typecheck] ──► [4 Canonicalize+Hash]
                                  │  (Decide-style scoring,      │ CompileError / Clarify      │
                                  │   constrained decoding)      ▼                             ▼
                                  └──────────────► exact cache: norm(prompt) → IR     result cache: H(lock,IR,profile,seed)
                                                                                                │ miss
 [5 Dispatch (static table: IR op → profile)] ──► [6 Codegen: template+grammar+policy+adapter+mask]
        ──► [7 Execute on deterministic runtime (lane: D1 | fast)] ──► [8 Verify / bounded repair]
        ──► [9 Emit result + receipt (lock hash, IR hash, margins, verifier verdicts)]
 Multi-op IR programs (DAGs) run on [10 Durable executor] with recorded tool activities.
```

**1 Normalize (deterministic code).** Unicode NFC, whitespace, quote/number normalization, template stripping; locale/unit/date resolution against an explicit context object (time zone, "today" is an input, never ambient).

**2 Parse (the only learned step in the front end).** The *Parse* profile of the base model (or a small dedicated parser) emits IR under the IR grammar. Intent selection is a closed-set scoring decision (R17) with a margin; slot filling is constrained decoding with greedy, batch-invariant execution. Few-shot exemplars per intent (cheapest prompt-sensitivity reducer per POSIX/ProSA). Low margin ⇒ `Clarify(question, candidates)`. Exact-match cache from normalized prompt → IR; an optional bounded-error similarity lookup (vCache-style) may only *propose* an IR candidate, which is then re-validated — never returned as a result.

**3 Analyze.** Type check, resolve references, check tool availability and policy, compute required capabilities. Errors are typed and actionable.

**4 Canonicalize.** Sort commutative collections, normalize units, canonical JSON, IR hash. This is where paraphrases converge.

**5 Dispatch.** A versioned static table maps IR op types to profiles (`decide.*`→Decide, `tool.*`→Call, `evaluate.*`→Judge, `create.*`→Create, `analyze.*|research.*`→Reason). Cost-tiering inside a profile (small → large model) is a deterministic cascade with versioned thresholds and dead bands, using confidence computed on deterministic inference.

**6 Codegen.** Profile builds the backend request: versioned prompt template, compiled grammar (cached by schema hash), decoding policy, adapter id, expert mask / router bias, seed schedule.

**7 Execute.** Two lanes: **deterministic lane** (batch-invariant kernels, fixed topology, R5–R10; certified per SKU) and **fast lane** (standard kernels; D0 only via recording). Selective determinism (R11) lets one fleet serve both once proven.

**8 Verify.** Schema validation (always), profile verifiers (executable where possible), bounded deterministic repair.

**9 Receipt.** Every response carries a receipt: lock hash, IR hash, profile version, seed, decision margins, fragile-routing share, verifier verdicts. Receipts make D2/D3 failures diagnosable after the fact.

**10 Durable executor.** IR programs that contain several ops (plan DAGs, LLMCompiler-style) run on a durable workflow engine (Temporal-class) with recorded activities and idempotent tools.

### 3.1 IR design (dmc-IR v0)

- Discriminated union on `op`: `decide.classify`, `decide.select`, `tool.call`, `tool.plan`, `evaluate.pairwise`, `evaluate.rubric`, `evaluate.verify`, `create.text`, `analyze.reason`, `analyze.data`, `research.report`, plus `meta.clarify`.
- All slots **required and typed** (nullable instead of optional), closed enums where possible, units explicit, `context` object (time, locale, user role), `constraints` (length, style, format), `seed` (only for `create.*`; default derived from IR hash), `ambiguities: list` (must be empty to execute).
- Flat, small schemas sized to fit hosted grammar limits (Anthropic: 20 strict tools, 24 optional / 16 union params) so the same IR works with external models.
- Versioned (`ir_schema_version` in the lock); migrations are deterministic functions.

---

## 4. The Deterministic MoE (D-MoE) core

### 4.1 Why one MoE base with profiles, and when to use separate models

- **One base, many profiles** keeps a single lock, a single conformance surface, a single discovery atlas and shared serving. LoRA on a shared base is the best-supported specialization route ("LoRA Without Regret", Sep 2025: matches full fine-tuning on small/medium SFT, rank 1 suffices for RL, must cover MLP/MoE layers; vLLM/SGLang/LoRAX serve dozens of adapters per GPU).
- **Expert masks are a hypothesis, not an assumption.** From-scratch MoEs show domain/vocabulary specialization (OLMoE) and task routing is concentrated enough for ESFT to tune only task experts, but production MoEs route per token at fine granularity and there is no 2026 evidence that their experts align with our five task families. Masks/router biases ship only if S2/S5 show a determinism gain within the capability budget.
- **Separate specialists** are used where an external model is clearly better (e.g. a dedicated reward model for Judge/Create ranking) — each one gets its own lock and conformance run.
- Decision rule (Phase 3 gate): a profile stays on the shared base unless an external specialist beats it by a pre-registered margin on that profile's golden suite *at equal or better determinism metrics*.

### 4.2 Base model selection (verified Oct 2026)

| Role | Model | Why |
|---|---|---|
| Discovery & methodology base | **Qwen3-30B-A3B** (128 experts, top-8, 48 layers, Apache-2.0) | Most published routing-mismatch data (R3), Qwen-Scope SAEs for all 48 layers, known super-experts, vLLM batch-invariance validated. |
| Production candidate | **Qwen3.6-35B-A3B** (256 experts, 8 routed + 1 shared, 40 layers hybrid Gated DeltaNet + gated attention, Apache-2.0) | Newest permissive small MoE, strong benchmarks. Its linear-attention (GDN) kernels need a batch-invariance audit (vLLM validated Qwen3-Next, which shares the design). |
| Dense control | **Qwen3.8-27B** (dense hybrid, Apache-2.0) | The model the user named; measures the MoE determinism tax (routing fragility) on the same family. |
| Scale-up option | DeepSeek-V4-Flash (284B/13B, MIT; hash-routed first 3 layers; vLLM bitwise rollout/training path) | Path to a larger D-MoE once methods are proven. |
| CPU-scale lab | granite-3.1-1b-a400m (32 experts, top-8), OLMoE-1B-7B | Fast iteration on probes and conformance tests without GPUs. |

Choice between the Qwen3-30B-A3B and Qwen3.6-35B-A3B for production is a **Phase 0 gate** decided by measured determinism tax, router-margin distributions and profile benchmarks.

### 4.3 Deep discovery program (every layer, circuit, router)

Interpretability does not by itself make outputs deterministic; kernels do (D1). Its jobs here are to (a) **measure** where decisions are fragile, (b) **find** the components that carry each profile's decisions, (c) **build levers** that widen margins or pin behaviour, and (d) **diagnose** parity regressions between releases. Staged, with artifacts and gates:

| Stage | What | Artifacts | Success metric |
|---|---|---|---|
| S1 Instrumentation | Hooks on every router, attention and MoE block (nnsight / TransformerLens 3.x `qwen3_moe` adapter); routing logs on golden traffic. | Routing log schema; per-request routing receipts. | 100% of layers instrumented; overhead measured. |
| S2 Router & expert atlas | Per-layer margin distributions (top-k vs k+1), fragile-token share, expert load by profile/domain/token type, co-activation graphs, cross-layer route paths, **super-expert profiling** (massive activations; never ablate: pruning 3 of 6,144 experts collapses Qwen3-30B-A3B Math-500 88→1.4), perturbation flip rates (dtype, batch, TP, paraphrase). | **MoE Atlas** (versioned dataset + report), **Fragility Map**, super-expert list. | Fragility map reproducible across two runs; flip rates explained by margins (AUC). |
| S3 Feature dictionaries | Reuse Qwen-Scope residual SAEs (32K/128K, all layers of Qwen3-30B-A3B); train transcoders for MoE blocks only where S4 needs them. | Feature catalog per profile (decision, format, tool-choice, confidence features). | Reconstruction/fidelity thresholds; features validated by intervention. |
| S4 Behaviour circuits | Contrastive datasets per profile (yes↔no, tool A↔B, prefer A↔B, answer X↔Y); EAP-IG / attribution patching to localize; attribution graphs on samples; expert-level attribution. | **Circuit cards** per profile decision; ranked component lists. | Faithfulness (patching recovers ≥ X% of logit diff); stable across seeds. |
| S5 Interventions (pre-registered) | Expert pinning/masking, router bias, decision-feature steering, margin-widening fine-tunes. Controls: random-expert null baselines, prompting baseline, off-target ≤ ⅓ on-target, capability budget. | **Intervention cards** with effect sizes. | Measured improvement in D2/D3 metrics (cross-toolchain TARa, PIR, fragile share) at ≤ budgeted capability loss. |

### 4.4 Router hardening toolbox

Inference-time (no retraining):
- **FP32 router + stable tie-break** (R7). Removes exact-tie and ULP-order flips on one stack.
- **Profile expert masks / router biases** (ESFT-style task expert sets from S2/S4), always including super-experts and shared experts. Shrinks the candidate set so near-ties with irrelevant experts cannot flip. Must be validated against capability budget.
- **Routing receipts + fragile-decision accounting** (R24): margin < ε (ε calibrated to measured cross-stack noise) marks the token "fragile"; requests whose decision tokens pass through many fragile routes get lower confidence and may abstain or be recomputed on the reference stack.

Training-time:
- **Router margin regularization**: hinge penalty `max(0, γ − (s_k − s_{k+1}))` on router logits during profile fine-tuning (with load-balance terms unchanged), aiming to move the margin distribution away from zero. (Our hypothesis; to be validated in S5 — no published result yet.)
- **Routing replay (R3)** during any RL so training sees the inference engine's routes (<3% overhead; prevents MoE RL collapse).
- **Batch-invariant training/inference numerics** so RL is truly on-policy (KL = 0 demonstrated by Thinking Machines; vLLM DeepSeek-V4-Flash path at 91% throughput).
- Optional for scale-up: hash routing for early layers (DeepSeek-V4 design) — requires pretraining-scale changes; not in scope for the first base.

### 4.5 Determinism-hardening training (per profile)

| Method | Applies to | Effect | Guardrail |
|---|---|---|---|
| Invariance / consistency training (BCT, ACT, Flip-Flop Consistency) on paraphrase + format augmentation with self-targets | Parse, Decide, Call, Judge | +11.6% cross-variant agreement (F²C); 86% less biased reasoning (BCT) | Held-out paraphrase families |
| Mode-seeking distillation (on-policy distillation; majority-vote targets à la ScPO/TTRL) | Decide, Call, Judge, Reason (final answers) | Collapses distribution onto one answer | Entropy–capability coupling (R = −a·e^H + b): monitor pass@k, never apply to Create |
| Calibration + abstention (RLCR Brier reward) | Parse, Decide, Judge, Reason | Turns low-confidence into explicit abstention | ECE and abstention precision gates |
| Router margin regularization | All except Create (optional) | Wider routing margins ⇒ fewer cross-stack flips | Load balance, capability budget |
| R3 + batch-invariant RL | Any RL stage | On-policy RL for MoE | KL monitoring |

---

## 5. The five specialist profiles

Each profile = `(adapter, expert mask/router bias, prompt template, grammar, decoding policy, verifier, thresholds)`, all locked and versioned.

### 5.1 Decide (decision models)
- **Input IR:** `decide.classify | decide.select` with closed label set, policy text, evidence.
- **Execution:** optional greedy reasoning span (deterministic lane) → **label scoring** over the closed set in one forward pass; confidence = calibrated softmax over label scores; margin < τ ⇒ `Abstain/Escalate`. Variant: linear probe head on frozen hidden state for high-volume decisions (cheapest, most stable).
- **Verifier:** policy rule checks where rules exist (deterministic code), consistency under label-order permutation.
- **Training:** classification SFT + calibration fine-tuning (~1k graded examples is enough per published recipes) + explicit abstention training + paraphrase invariance; label-permutation augmentation. Thinking mode is *off* by default: reasoning-tuned models abstain ~24% less (AbstentionBench).
- **External comparators:** gpt-oss-safeguard-20b (policy-conditioned reasoning classifier, Apache-2.0); scalar-head classifiers (Bradley-Terry heads à la Skywork-Reward-V2; distilled small LLM + classification head, the Kaggle LMSYS-winning recipe).
- **Targets:** TARa = 1 on lock (D1), cross-toolchain decision parity ≥ 99.5% on non-abstained items (D2), PIR ≥ 97% (D3), ECE ≤ 0.03.

### 5.2 Call (function calling only)
- **Input IR:** `tool.call | tool.plan` with tool registry ids.
- **Execution:** (1) **compile from IR first**: if all arguments are present as typed IR slots, the call is generated by deterministic code with no model call; (2) otherwise tool selection = Decide-style scoring over {tools, `no_tool`, `clarify`}; (3) argument filling under the tool's JSON-Schema grammar (strict), greedy; (4) canonical JSON of the call (RFC 8785); (5) argument validation (types, ranges, referential integrity); (6) idempotency key = call hash; executed as a recorded activity.
- **Training data:** APIGen-MT / ToolACE / xLAM-style synthesized trajectories, plus our IR-derived synthetic calls.
- **Benchmarks:** BFCL v4 (with care: Epoch AI rated it "Flawed" in Sep 2026, 48% of sampled tasks defective), τ²-bench, plus internal golden suites; metric pass^k (reliability) not only pass@1 — every model loses 10–30 points from pass^1 to pass^4 on τ² retail/airline.
- **Why the fixed interface matters most here:** BFCL v4 format-sensitivity deltas reach 81.5 points for small specialists (ToolACE-2-8B) vs 8.0 for Qwen3-235B. The IR + compiled grammar *is* the fixed interface; specialists (xLAM-2, ToolACE-2, Hammer2.1) excel on AST categories but collapse on web-search/memory categories, so Call stays narrow (calls only) and delegates planning to `tool.plan` + the durable executor.
- **Targets:** schema validity 100%; exact-call match (canonical) ≥ target; no hallucinated tool names (grammar-enforced).

### 5.3 Judge (contrastive cross-evaluation)
- **Input IR:** `evaluate.pairwise | evaluate.rubric | evaluate.verify`.
- **Execution:** candidates are first **style-normalized** (rendered to a canonical plain form: markdown stripped, whitespace/list formatting normalized) because style/markdown bias is the dominant judge bias in 2026 audits (0.10–0.76) while position bias is small (≤0.04), and SOTA RMs drop to 46.6% under style interference (RM-Bench). Pairwise judgments are then evaluated in **both orders**; verdict = mean log-odds of preference tokens across orders; tie band → `tie`.
- **Models:** scalar Bradley-Terry RM for ranking (Skywork-Reward-V2-Llama-3.1-8B: RewardBench 2 84.1, JudgeBench 80.0, RM-Bench 92.8 vs ~65 for prompted GPT-4o) plus a generative verifier profile on the base (better out-of-domain; uses maj@k over fixed seeds). Rubric scores = expected value of the score-token distribution, not a sampled number. Panels of heterogeneous judges (different families than the generator to avoid self-preference) with agreement statistics; disagreement is surfaced, not hidden.
- **Verifiers first:** executable verifiers (tests, solvers, checkers) dominate learned judges whenever available; PRMs for step-level reasoning checks.
- **Targets:** position-swap consistency ≥ 98%; cross-toolchain verdict parity; agreement with human gold sets reported with κ.

### 5.4 Create (best-of creative content)
- **Determinism definition:** output is a pure function of `(IR, seed)`; default seed = H(IR). Users get variety by changing the seed explicitly.
- **Execution:** plan → draft → critique → revise pipeline; at each sampling stage N candidates with seeds `H(IR, stage, i)` using counter-based RNG on the deterministic lane; ranking by a locked reward model / Judge profile with stable tie-break. Quality from search + selection; reproducibility from seeds + deterministic ranking.
- **Diversity from an explicit, enumerated latent, not temperature noise:** Verbalized-Sampling-style "list k distinct approaches" (recovers 1.6–2.1× diversity lost to preference tuning) or mode-conditioning; the chosen mode index is part of the IR/seed, so it is reproducible.
- **Models:** best small open writer is **Qwen3.8-27B** (dense, Apache-2.0, EQ-Bench Creative v3 Elo 1671); best open writers overall are Kimi K3 (2082) and GLM-5.3 (2075) under custom licenses, ~90–100 Elo behind the top closed models. Create is therefore the profile most likely to run on a separate specialist rather than the MoE base; rankers: LitBench / WritingBench-critic reward models + Judge profile.
- **No mode-seeking or entropy minimization** on this profile (diversity collapse is the main quality risk).
- **Targets:** D1 per (IR, seed); quality on EQ-Bench Creative Writing / WritingBench-style evals and blind human preference vs. the best open creative model.

### 5.5 Reason (deep reasoning and analysis)
- **Input IR:** `analyze.reason | analyze.data | research.report`.
- **Execution:** thinking-mode generation on the deterministic lane (BF16 diverges ~token 65 vs ~1,826 in FP32 across stacks per LayerCast, so long chains need D1 kernels and, for D2, optional FP32 accumulation); **answer-level determinism** via K fixed-seed chains + majority vote with deterministic tie-break; final answer extracted by grammar. Data work runs in a pinned sandbox (pinned interpreter and libraries, fixed seeds). Research retrieval is snapshotted by content hash (R23).
- **Verifiers:** code execution, numeric checks, PRM/Judge on steps, citation-to-snapshot verification.
- **Models:** reasoning adapter on the base; **Tongyi DeepResearch-30B-A3B** (Apache-2.0, Qwen3-30B-A3B architecture, so the discovery atlas transfers) for `research.report`; Qwen3.8-27B as dense comparator. Training recipe: RLVR (GRPO-family, strictly on-policy) with R3 + batch-invariant numerics.
- **Targets:** TARa across toolchains, pass^k, accuracy on AIME/GPQA/LiveCodeBench-style evals within budget of the base model.

### 5.6 Parse (front-end profile)
The NL→IR parser is itself a profile with the strictest D3 targets: closed intent scoring + constrained slot filling + clarification; trained with paraphrase/format invariance and calibrated abstention; certified on paraphrase suites before any other profile is certified.

---

## 6. Verification and certification

**Metric suite (reported per profile, per lock, per release):**
1. *Bitwise (D1):* TARr@N, unique@N (N=1000 on a sample), first-divergence index; under randomized concurrent load and cache/chunking variations.
2. *Decision parity (D2):* TARa across certified toolchains; fragile-routing share.
3. *Semantic (D3):* PIR on paraphrase/format families, flip rate, POSIX.
4. *Conformance (D4):* schema validity, oracle pass rate, pass^k, ECE, abstention precision/recall.
5. *Release parity:* negative-flip rate and changed-output rate vs. previous release; every changed decision listed in the release diff.

**Test methodology:**
- Golden corpora per profile (from production-like traffic, Shopify-style), metamorphic relations (paraphrase, option-order permutation with label remapping, unit changes, irrelevant-context insertion), **differential testing** across toolchains and releases, **batch-mate fuzzing** (random concurrent loads, chunk sizes, cache states) for D1, long-generation tests for divergence, and a **synthetic-oracle generator** (programmatic tasks with known answers) to scale D4 testing.
- Statistical rigor: confidence intervals on every rate; MMD-style distribution-equality tests for "is this endpoint serving the locked model" (11 of 31 hosted Llama endpoints failed such a test in published work).

**Release gates (initial targets, tuned in Phase 0–1):** D1 `unique@1000 = 1` on 100% of sampled golden prompts; D2 decision parity ≥ 99.5% (Decide/Call/Judge), ≥ 98% (Reason final answers); D3 PIR ≥ 95% (Parse), ≥ 97% (Decide); D4 schema validity 100%; negative-flip rate ≤ 0.5%.

---

## 7. Roadmap

| Phase | Weeks | Scope | Exit gate |
|---|---|---|---|
| **P0 Foundations** | 0–4 | Lockfile, canonical hashing, metric suite, IR v0, result store, CPU router probe (granite/OLMoE), GPU baselines: Qwen3-30B-A3B, Qwen3.6-35B-A3B, Qwen3.8-27B under default vs batch-invariant vLLM; determinism-tax and routing-fragility measurements. | Baseline report; base-model choice; cost of D1 measured. |
| **P1 Deterministic runtime** | 3–10 | Deterministic lane on vLLM batch-invariant + MoE fixes (FP32 router, stable tie-break, static scales, pinned MoE kernel configs); conformance suite (batch, chunking, prefill/decode, cache, preemption); counter-based RNG; receipts. Evaluate selective determinism (LLM-42 / MarginGate). | D1 gate passes on chosen base under load on one SKU. |
| **P2 Compiler front end** | 6–14 | IR v1, Parse profile (scoring + constrained decoding + clarification), caches, IR rendering, round-trip checks, paraphrase suites. | Parse PIR and clarification targets met. |
| **P3 Profiles v1** | 10–20 | Decide, Call, Judge, Create, Reason profiles (adapters, policies, grammars, verifiers); external-specialist bake-offs; durable executor for IR programs. | Each profile meets its D1/D4 gates and benchmark floors. |
| **P4 Discovery program** | 8–24 (parallel) | S1–S5 on Qwen3-30B-A3B, then port to production base; MoE Atlas, Fragility Map, circuit and intervention cards. | ≥1 intervention with measured D2/D3 gain within capability budget, or a documented negative result. |
| **P5 Hardening training** | 18–32 | Invariance, mode-seeking (non-Create), calibration/abstention, router-margin regularization; R3 + batch-invariant RL. | D2/D3 metrics improve vs P3 without capability regression beyond budget. |
| **P6 Certification & ops** | 24–36 | Release process, differential testing at scale, drift monitoring, SKU certification matrix, documentation of the determinism contract. | First certified release (D0–D4 report). |

**Indicative compute:** P0–P2: 1 node of 8× H100/H200 (serving + probes). P4: +1 node (activation caching for SAEs/attribution; ~61 GB weights, activation caches in the tens of TB per 100M tokens if cached naively — stream instead). P5: 2–4 nodes for LoRA/RL on a 30B-A3B base.

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| D1 overhead (1.3–2.1×) too costly | Medium | Selective determinism; CoRun-style fixed-shape padding; deterministic lane only for traffic that needs it. |
| Undiscovered MoE kernel leaks | High | Own conformance suite + batch-mate fuzzing; track Vosti-style formal checks; pin engine versions. |
| D3 targets unreachable for open-ended domains | High | Narrow IR to closed domains first; clarification over guessing; report PIR honestly per domain. |
| Mode-seeking training reduces capability | Medium | Capability budget per profile; never on Create; pass@k monitoring. |
| Interpretability yields no usable levers | Medium | Pre-registered S5 with negative results accepted; D1–D4 do not depend on S5. |
| Router-margin regularization hurts quality or load balance | Medium | Small γ sweeps; ablations; keep as optional. |
| Hosted / closed models needed for some profiles | Medium | They can only be D0 (record/replay); flag in receipts. |
| License constraints at frontier scale | Low | Stay on Apache/MIT bases (Qwen3.x A3B, DeepSeek-V4-Flash). |

---

## 9. What exists in this repo now (Phase 0 scaffold)

See `README.md` for the current component list and how to run the tests and the router probe.

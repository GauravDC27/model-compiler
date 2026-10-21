# Adversarial review, round 1 — `docs/PLAN.md` v1 (DMC)

**Reviewer:** Fable 5.1 · **Date:** 2026-10-21 · **Materials:** PLAN.md; research notes 01–07; `experiments/00_granite_router_probe_quick.{md,py}`; `dmc/` scaffold; two live fetches (vLLM batch-invariance docs, vLLM tracking issue #27433). One web search (batch-invariance status of Qwen3.6-35B-A3B's Gated DeltaNet kernels) could not be run because the session's shared search budget was exhausted; that gap is flagged where it matters (F1).

Citations are `NN:line` into `research_notes/Deterministic MoE model compiler/NN_*.md`, or URLs.

---

## Overall assessment

The plan is unusually honest about what determinism is (a stack of contracts, D0–D4/D⊥), it reads the numerics literature correctly (nearly every number I checked matches the notes), and its compiler decomposition (NL → typed IR → static dispatch → profile → verifier) is the right skeleton: it takes the strongest 2026 evidence (static per-task tables beat learned routers; the prompt, not the GPU, is the dominant variance source) and builds on it. Its weaknesses are concentrated in exactly the places the user cares about most. (1) The chosen production base, Qwen3.6-35B-A3B, is a Gated-DeltaNet hybrid with a vision tower and MTP for which no batch-invariance evidence exists; its nearest relative (Qwen3-Next) is currently *broken* in vLLM's batch-invariant mode, and none of the discovery assets the plan relies on (Qwen-Scope SAEs, TransformerLens adapter, super-expert list, R3 data) are for that model, so "S1–S5 on Qwen3-30B-A3B, then port" quietly turns the deep-discovery program into a methodology demo on a model that will not ship. (2) The "profiles on one base" architecture introduces a batch-variance surface (multi-LoRA batching) that the plan's own notes warn about and the conformance suite does not test. (3) The headline D2 target (≥ 99.5% cross-toolchain decision parity "on non-abstained items") is unfalsifiable without a coverage floor, and the only mechanism offered to reach it (router-margin widening) is admitted to be an untested hypothesis that is not de-risked until weeks 8–32, even though a 1-GPU pilot is possible in week 1 with the probe script that already exists. The critical path is therefore P0 base selection → P1 deterministic lane *including adapters* → the D2 mechanism; the riskiest assumption is that an MoE's cross-stack decision parity can be pushed to 99.5% by training rather than by pinning routes. The plan can be made approvable with targeted edits; it should not be executed as written.

---

## Findings

### BLOCKER

#### F1 — The production base (Qwen3.6-35B-A3B) is unsupported by the plan's own evidence, and choosing it forfeits the discovery program
**Section:** §0.6, §4.2, §7 (P0, P4 "then port to production base").
**What is wrong.** The plan recommends Qwen3.6-35B-A3B as production candidate on the strength of "vLLM validated Qwen3-Next, which shares the design" (§4.2). Three problems:
1. *Qwen3-Next is not validated today.* The tracking issue records that since PR #40193 "the Qwen3-Next entry in the tested list no longer starts in batch-invariant mode" (Oct 4, 2026; 01:173; confirmed live at https://github.com/vllm-project/vllm/issues/27433). SGLang's deterministic flag likewise fails on hybrid Mamba2 granite-4.0-h (01:237). Qwen3.5/3.6, Gated DeltaNet and Mamba are mentioned nowhere in the vLLM docs or issue (live fetch). The GDN chunked-scan recurrent kernels are a whole kernel family with chunk-size-dependent reduction order that nobody has audited; the notes list it as an open checklist item (02:549), and I could not find any public result for Qwen3.6 (search unavailable this session). The plan treats this as a footnote ("needs a batch-invariance audit") rather than as a go/no-go.
2. *Vision + MTP are extra surfaces the plan never scopes out.* Qwen3.6-35B-A3B ships a vision encoder and MTP head (02:361). Image preprocessing is a determinism surface; MTP/speculative decoding has one bitwise data point (GLM-4.5-Air on B300, 01:171) and is explicitly unsupported on the DeepSeek-V4 consistent path (01:197). The plan is silent on both (see also F4).
3. *The atlas does not port.* Qwen-Scope SAEs cover Qwen3-30B-A3B-Base and Qwen3.5-35B-A3B-Base (03:87) — not Qwen3.6; the TransformerLens adapters are `qwen3_moe`/`qwen2_moe`/`olmoe` (03:335), with no hybrid-GDN adapter; super-expert locations are known for Qwen3-30B-A3B only (03:140); R3/fragility data are Qwen3-30B-A3B only (02:141–163, 02:187–193). The notes also warn that behaviour experts need not survive post-training (03:491) and that each target model needs its own validation (03:497). "Port" is a second discovery program on a model whose 3/4 of layers have no attention heads to attribute through.
**Why blocker.** The user's core ask is a deterministic MoE whose *every layer, circuit and router* has been discovered and hardened. As written, the discovery is done on model A and the product ships model B.
**Fix.**
- Make **Qwen3-30B-A3B (text-only) the v1 production base** by default; it is the only MoE with (a) vLLM BI validation, (b) R3 engine-mismatch data, (c) all-layer SAEs, (d) a known super-expert list, (e) a TransformerLens adapter. Accept the capability gap for v1 and document it.
- Keep Qwen3.6-35B-A3B as a **P0 spike with a written go/no-go**: GDN kernels pass batch-mate fuzzing + chunked-prefill boundary tests + prefix warm/cold; MTP disabled in the deterministic lane; vision disabled (text-only lock); an SAE/transcoder plan for GDN layers costed. No pass → it stays a P6+ scale-up option alongside DeepSeek-V4-Flash.
- Rewrite P4's exit gate so the atlas is produced **for the base that ships**, not "then ported".

#### F2 — Multi-LoRA profile serving is an untested batch-variance surface; D1 cannot be claimed "under any batch mix" without it
**Section:** §2.2 (R5–R8), §4.1, §6 test methodology, §7 P1 exit gate.
**What is wrong.** Every profile is "adapter + mask + policy" served on a shared base with "dozens of adapters per GPU" (§4.1). Multi-LoRA serving uses grouped/segmented kernels (SGMV/BGMV-class) whose work partition depends on *which adapters* are co-batched — the batch-composition dependence TML identified, in a new place. The plan's own note says so verbatim: "Batch invariance must cover adapter batching. Multi-LoRA batching mixes requests for different adapters in one batch, which compounds Thinking-Machines-style batch non-invariance" (05:617), and lists "no benchmark directly comparing determinism of multi-LoRA serving vs separate model deployments" as a gap (05:621). vLLM's BI docs fix only the LoRA *shrink* reduction ("8 fixed K chunks", 01:160; live fetch); LoRA expand, Punica/SGMV and multi-adapter batching are not mentioned anywhere in the docs or the tracking issue (live fetch). The conformance suite (§6) fuzzes batch-mates, chunking and cache state but never *mixed adapters*, and P1's exit gate is "D1 passes on chosen base" — base, not base + adapters.
**Fix.**
- Add to R5/R8 and the §6 conformance matrix: **adapter-mix fuzzing** (same request, co-batched with 0..k requests on other adapters, with expert masks on/off) → identical logits at every position.
- P1 exit gate: D1 on **base + ≥3 adapters co-resident**, not base alone.
- Pre-register the fallback: if multi-LoRA kernels cannot be made invariant in budget, serve **merged per-profile checkpoints** (LoRA merged into weights; one model per profile) with Symbolic-MoE-style *batch-by-profile* scheduling (06:109, 06:126), which makes batch composition a deterministic function of the dispatch plan. State the memory cost (one 61 GB copy per profile on Qwen3-30B-A3B) in §7 compute.

#### F3 — The D2 target is unfalsifiable as written, and the mechanism behind it is de-risked far too late
**Section:** §1 (D2 row), §5.1 targets, §6 release gates, §4.4, §7 P4/P5.
**What is wrong.**
1. *Vacuous metric.* "Cross-toolchain decision parity ≥ 99.5% on non-abstained items" can be met by abstaining on every low-margin item. There is no coverage floor, no risk–coverage curve, and no statement of how τ is set. The notes point the way: selective-prediction / risk–coverage framing (06:257), conformal guarantees that are explicitly *marginal* and calibration-distribution-bound (07:186, 07:369).
2. *Load-bearing hypothesis, tested last.* The plan's only training-time D2 lever for MoE is router-margin regularization — "Our hypothesis; to be validated in S5 — no published result yet" (§4.4) — and the notes confirm the gap: no margin distributions published for any production MoE (02:240), margin widening and expert pinning as variance reducers untested (03:496). Yet this sits in P4/P5 (weeks 8–32). The riskiest assumption in the plan gets no early experiment, although the repo already has a CPU probe that measures exactly the margin distribution the hypothesis is about.
3. *No fallback mechanism.* If margins cannot be widened without capability loss (the notes warn load balancing and capability are coupled, 02:168–169), the plan has nothing left for D2 on MoE except abstention.
**Fix.**
- Redefine the D2 gate as a **risk–coverage pair**: parity ≥ 99.5% *at coverage ≥ C₀ per profile* (e.g. C₀ = 90% for Decide, set in P0 from baseline margins), with τ chosen by split-conformal calibration on a frozen set and reported with CIs. Report the full curve in receipts/release diff.
- Pull a **margin-regularization pilot** (granite-1b-a400m / OLMoE, 1 GPU or CPU, hinge on top-k gap with LBL unchanged) into **P0–P1 (weeks 1–6)** using `experiments/00_granite_router_probe_quick.py` as the measurement. Pre-register: margin-mass-below-ε must fall by ≥ 2× at ≤ 1-point held-out loss, else the lever is dropped from P5.
- Add **route pinning** as the primary non-training D2 mechanism (see F6): record per-token top-k masks in the receipt and allow a certified reference stack to re-execute with pinned routes, removing the discrete amplifier entirely.

### MAJOR

#### F4 — The D1 contract and conformance suite omit known failure modes
**Section:** §1 (D1), §2.1 R1, §2.2 R6, §6.
**What is wrong / evidence / fix (one line each):**
- *Speculative decoding / MTP:* not mentioned. One positive data point (01:171), unsupported on the V4 path (01:197); the verify pass is a prefill-shaped forward, i.e. a prefill-vs-decode variation of exactly the kind Vosti found leaking (01:35). → Disable in deterministic lane or add "spec-on vs spec-off → identical tokens *and* logits" to the matrix.
- *Async scheduling:* reported to break BI even with the flag (06:161, secondary) → add `async_scheduling=off` to the lock flags and to the matrix.
- *Multi-turn / session state:* D0/D1 keys are `H(lock, IR, profile, seed)`, but the IR `Context` (dmc/ir.py) carries only time/locale/timezone. Conversation history, prior tool results and prior IRs change the output; they are not in the key. → Add `session_state_hash` (canonical hash of prior turns + snapshotted tool results, R23) to the IR context and to the cache key; add a "no re-dispatch mid-program" rule (vLLM-SR hard locks, 06:38, 06:262).
- *Vision inputs:* Qwen3.6 has a vision tower (02:361); decode/resize is a determinism surface. → Text-only in the v1 lock, or hash decoded pixel tensors and pin the preprocessing library.
- *Tokenizer implementation version:* R1 pins the tokenizer hash, not the `tokenizers`/`transformers` version; template drift already bit 2609.25624 (01:80). → pin library versions in `RuntimeLock`.
- *Grammar engine:* R16 names XGrammar-2 or llguidance; XGrammar had the most under-constrained failures (38 categories, 04:179). → pin the engine + version + grammar hash in the lock, and run the other engine as an offline cross-check (04:215).
- *CUDA-graph padding ladder and expert masks:* a profile mask changes active-expert token counts, i.e. the very input to token-count-dependent MoE kernel selection (02:461, 01:47). → masks must be part of the fuzzed dimensions (also in F2).
- *Sampler ties:* R9 fixes the RNG but not top-k/top-p tie order. → stable sort by (prob, token id).
- *Prefix caching:* vLLM has "no stated support"; "might work in some certain cases" (01:172; live fetch). R6 allows "or disable"; make the v1 lock *disable* it and say so.

#### F5 — The deep-discovery program has no teeth, and its stated coverage ("every layer, circuit, router") is not what S1–S5 deliver
**Section:** §4.3, §7 P4 exit gate, §8.
**What is wrong.**
- P4's exit gate accepts "a documented negative result", and §8 says "D1–D4 do not depend on S5". Together these make the user's headline ask optional. Nothing from S2–S4 is a release artifact except R24 telemetry; the atlas is never re-run or diffed per release even though the notes say behaviour experts may not survive fine-tuning (03:491) and recommend re-running the atlas after every fine-tune (03:490).
- S3 relies on Qwen-Scope SAEs trained on **Base** checkpoints (03:87) while the product runs Instruct + per-profile LoRA; GDM found chat-specialized SAEs close half the probe gap (03:98). Feature labels will drift under the plan's own adapters. CLT training for MoE blocks is "where S4 needs them" with no budget; no GPU-hour figure exists for a 30B MoE (03:389, 03:495) and the only reference points are enormous (Gemma Scope > 20% of GPT-3 compute, 03:77).
- Attribution graphs give "satisfying insight" on ~25% of prompts (03:43); interventions beat prompting rarely (AxBench, 03:100); steering is resisted mid-generation (ESR, 03:256). The plan knows this (its S5 controls are good) but never says what the *minimum guaranteed* deliverable is.
**Fix.**
- Define a **non-optional discovery deliverable** that ships with every release regardless of S5's outcome: (i) the per-layer router-margin atlas and Fragility Map for the shipping base *and each adapter*; (ii) fragile-routing share wired into the confidence/abstention signal (already sketched in §4.4 — make it a MUST with an acceptance test); (iii) an **atlas diff** (routing-path drift, super-expert check, margin-distribution shift) as a P6 release gate next to the negative-flip gate.
- Make **route pinning/replay** (F6) the discovery program's product-facing lever, so S2's routing logs are the artifact that makes outputs deterministic, not just a report.
- Either budget CLT training explicitly (GPU-hours, storage) or drop CLTs from S3 and commit to residual-SAE + EAP-IG over coarse nodes {heads, experts, shared expert, router logits}, which the notes say is the realistic stack (03:103).
- Add "re-fit or re-validate dictionaries after each profile adapter" to S3.

#### F6 — Missed levers that directly address "a deterministic MoE": route pinning at inference, router freezing, and dense students
**Section:** §4.4, §5.1–5.3, §7 P3.
**What is wrong.** The plan uses routing replay only for RL (R3, §4.4 training-time). The notes list inference-time route logging and replay as lever #4 of 5, which "turns routing into a recorded 'compilation artifact'" (02:231), and note that cross-stack bitwise MoE reproduction is unattainable "unless the router input is bitwise identical, or routing decisions are logged and replayed" (02:226). With pinned routes, a different toolchain differs only by dense numerics — the discrete amplifier is gone. This is the single most compiler-like idea available for MoE and it is absent. Also absent: StableMoE-style **frozen/distilled router** (02:205–209, 02:293; a fine-tune-time lever, unlike hash routing which the plan correctly defers), and **dense-student distillation** for the three decision-bearing profiles (Decide, Call, Judge) — the notes show dense logprob noise is 2.4× lower (KL 6.4e-4 vs 1.5e-3, 02:148–149, 02:226), dense models have no routing flips at all, and the best-documented decision recipe is a distilled 9B dense student (05:36, 05:550). The plan mentions a "distilled small LLM + classification head" only as a comparator.
**Fix.**
- §4.4 inference-time toolbox: add **routing receipts → route-pinned re-execution** (`routes` field in the receipt; a `pin_routes` execution mode; D2 certification compares pinned-route re-execution across stacks). Cost: 48 layers × top-8 ≈ 1–2 KB/token (03:385).
- §4.4 training-time: add **router freeze/distillation (StableMoE)** as an S5 candidate with the same pre-registration as margin regularization.
- §5.1/5.2/5.3 + P3 bake-offs: add a **dense student** arm (MoE hub as teacher; on-policy distillation, 07:73–81) per decision profile, judged on the same risk–coverage gate. If the student wins at equal determinism metrics, the MoE is the teacher/hub and the deterministic executors are dense — say this possibility out loud in §4.1.

#### F7 — The "MoE determinism tax" control is confounded
**Section:** §0.6, §4.2 (Dense control row), §7 P0.
**What is wrong.** Qwen3.8-27B is a 64-layer GDN+attention hybrid (02:345–347). Qwen3-30B-A3B is plain-attention. Comparing them measures MoE-vs-dense *and* GDN-vs-attention at once. R3's dense control for Qwen3-30B-A3B was Qwen3-8B (02:148, 02:438); Qwen3-32B passes BI 22/22 on GB300 (live fetch of #27433).
**Fix.** Pair each MoE with its same-generation, same-attention dense sibling: Qwen3-30B-A3B ↔ Qwen3-32B (and Qwen3-8B for cheap runs); Qwen3.6-35B-A3B ↔ Qwen3.6-27B (the notes infer it exists, 02:349) or Qwen3.8-27B. Keep Qwen3.8-27B as "the model the user named" for capability comparison only. Define the tax as a tuple (unique@N default mode, first-divergence index, fragile-token share, cross-stack TARa), not a single number.

#### F8 — The specialist sprawl contradicts the "single lock, single atlas" argument and the license policy
**Section:** §4.1, §5.1–5.5, §8 (license row).
**What is wrong.** By the plan's own text, Create "is the profile most likely to run on a separate specialist" (Qwen3.8-27B), Judge ranking uses Skywork-Reward-V2-**Llama-3.1-8B**, `research.report` uses Tongyi DeepResearch, and Decide names gpt-oss-safeguard (harmony template, 05:27). That is three to four of five profiles off the hub. Each spoke needs its own lock, conformance run and (per F5) its own atlas — none is budgeted. Specific errors: (a) Skywork-Reward-V2-Llama-3.1-8B is Llama-licensed; the Apache variant is the Qwen3-8B one at RB2 78.2 / JudgeBench 73.4, not 84.1 / 80.0 (05:33, 05:239) — §8 says "stay on Apache/MIT bases"; (b) "Tongyi DeepResearch-30B-A3B … Qwen3-30B-A3B architecture, so the discovery atlas transfers" is unsupported — the notes give no architecture statement (05:435–439) and agentic continual pretraining changes weights; only super-expert *locations* are shown to survive post-training (03:139, 03:491).
**Fix.** Rename the architecture honestly: **hub (D-MoE) + certified spokes**. Add a spoke table (model, license, lock, conformance owner, atlas scope = none). Fix the Skywork citation to the Apache variant or justify the license exception. Delete the atlas-transfer claim for Tongyi; replace with "SE profile re-run required".

#### F9 — P5 assumes bitwise/batch-invariant RL for Qwen3-30B-A3B that does not exist publicly
**Section:** §4.4 (training-time), §4.5, §7 P5 and compute.
**What is wrong.** The only bitwise MoE train/inference path is DeepSeek-V4-Flash in vLLM on GB200 (01:187–198); the dense prototype (Qwen3-1.7B, vLLM+TorchTitan) ran 2.4× slower with two model-code copies and no `torch.compile` (01:477). For Qwen3-30B-A3B there is no published bitwise trainer path; building one means custom backward passes for batch-invariant grouped GEMM/top-k/combine. R3 itself "does not reproduce the whole rollout forward pass" (02:164). The plan's "KL = 0" line (§4.4) and "2–4 nodes" estimate assume this work is free.
**Fix.** Baseline P5 on what exists for this model: R3 routing replay (ms-swift/slime, 02:155–157, 03:298) + BF16-or-FP32 router on both sides (02:171–177) + FP16 or TIS/MIS (01:472, 01:476) with the sampling mask exported. Make bitwise RL a stretch item with its own engineering estimate (kernel backward passes) or defer it to the DeepSeek-V4-Flash scale-up where it is already shipped.

#### F10 — Parse *is* the router; the plan relocates the routing-flip problem rather than removing it, and ignores adversarial routing
**Section:** §0.5, §3 step 2, §5.6, §2.5 R21.
**What is wrong.** Static dispatch on IR op type is correct, but intent selection in Parse is a learned classifier with exactly the paraphrase-flip profile of production routers (OrcaRouter pick-flip 16.7% → 2.4% with an ε-tie-break, 06:80; 98% flip under keyword injection, 06:74; adversarial suffixes, 06:75). If Parse runs on the MoE base it inherits MoE routing fragility in the one step the plan calls "the only learned step". There is no threat model, and R21's metric suite has no adversarial-flip rate although the notes say it belongs in CI (06:90). Router-level attacks on MoE (GateBreaker, BadMoE, 03:165) are also unaddressed.
**Fix.** Default Parse to a **small dense parser** (certified separately; the notes give the determinism argument in F6) and keep the MoE Parse profile as the comparator. Add to R21: paraphrase pick-flip rate with the ε/dead-band tie-break, and an adversarial-suffix / keyword-injection flip test. Add one line to §8 on router/expert-level attack surface.

#### F11 — "Vosti-style contract" is claimed in the D1 guarantee type but not resourced, and does not cover MoE or GDN kernels
**Section:** §1 (D1 row), §2.2.
**What is wrong.** Vosti's proofs cover a scheduler, paged prefix cache and Triton attention kernels for *dense* Llama 3.x / Gemma 3/4 (01:369–374). No MoE grouped-GEMM, top-k, permute/combine, or linear-attention kernel has a formal batch-invariance proof. The plan allocates no work item for extending the Triton analyzer, yet labels D1 "Hard … (tested + Vosti-style contract)".
**Fix.** Either downgrade D1's guarantee type to "hard on certified stacks (tested; formal contract for dense kernels only)" or add a P6 work item: run the Vosti Triton analyzer on the MoE kernel set and GDN scan (if F1's spike passes), with a budget. Keep the formal spec language ("bitwise-identical logits at corresponding positions under batch, chunk, phase, cache variations") as the *definition* of D1 — that part is right.

### MINOR

#### F12 — Number and status drift
- §0.2 "1.3–2.1× cost": TML measured 1.6–2.1× (01:108–116); the 1.3 is an inference in 07:337. Say "1.6–2.1× (naive), ~1.1× (co-designed, V4-Flash)".
- §2.2 R7 "vLLM's DeepSeek-V4 PR adds exactly this": PR #54955 is a draft marked DO NOT MERGE with formal results "still TODO" (02:509–515). Cite the merged V4-Flash path (01:190–192) instead.
- §4.2 "vLLM validated Qwen3-Next": stale since Oct 4, 2026 (F1).
- §5.3 "position bias is small (≤0.04)": true for the TMLR audit, but *Judging the Judges* shows it scales with the quality gap (05:283–286); keep both-order judging as a MUST, not an optimization.

#### F13 — Internal inconsistency: "five profiles" (§0.5) vs six in §5 (Parse). Count Parse explicitly; it has the strictest targets.

#### F14 — §9 points to `README.md` for "the current component list and how to run the tests"; the README is two lines and `tests/` is empty. Either fill the README or make §9 list `dmc/{ir,lockfile,canonical,store,metrics,moe_probe}.py` and state that tests are a P0 deliverable.

#### F15 — Judge style-normalization is unconditional (§5.3). Stripping markdown is right when the criterion is content, wrong when the rubric is about formatting/IF. Make normalization a function of `evaluate.rubric.criterion` type and record it in the receipt.

#### F16 — Cascades are attackable and the plan's dispatch has no cost bounds. Forced-deferral attacks (06:259–260); escalation budgets should be static constants in the lock (06:274). Add `max_tier`/`max_repair` constants to the dispatch table.

#### F17 — Cross-kernel integer path missing from the toolbox. Power-of-two INT8 scales made CUTLASS and Triton kernels agree bitwise at +0.3–0.5% perplexity (01:419); ReproFuse's batch invariance holds only for M ≤ 64 (01:414). Add both to §4.4 as the D2→D1 upgrade path across SKUs, with the M-limit noted.

#### F18 — Paraphrase ensembling as a certification tool. Parse k paraphrases, require IR-hash agreement; disagreement is a free Clarify signal (04:281, 04:333). Cheap offline; add to R14/R15.

#### F19 — No early vertical slice. P0–P3 are horizontal layers; integration risk (lock → IR → grammar → receipt → store) surfaces at week 10+. Add a week-5 thin slice: `decide.classify` end-to-end on granite-1b-a400m on CPU with lock, receipt, replay and a `unique@100 = 1` test.

#### F20 — Receipt should carry the sampling mask and routing trace. The sampling-support mismatch (01:476) and R3 fallback both need them; cheap (03:385).

#### F21 — "Deterministic per (IR, seed)" for Create with default seed = H(IR) means any IR edit (even a `constraints.length` change) yields an unrelated draft. Fine, but document it and offer `seed` inheritance for revisions so edits are local.

---

## What the plan gets right — do not change

- Determinism as a **stack of contracts** (D0 replay, D1 bitwise, D2 decision parity, D3 semantic, D4 spec, D⊥ defined failure) with honest guarantee types; "hard" only where engineering can deliver it.
- The **non-goals**: no cross-vendor bitwise, no correctness promise beyond oracles, creative output deterministic per (IR, seed) not per "meaning".
- **Lockfile + canonical hashing + content-addressed replay + receipts** (R1–R4, step 9). This is the compiler "toolchain version" and it is already scaffolded.
- **NL → closed typed IR with an ambiguity channel**, ambiguity as a compile error, all-required-nullable slots, canonicalization as the point where paraphrases converge (R12–R15, §3.1).
- **Static dispatch on IR type over a learned per-query router**, with versioned thresholds and dead bands (§3 step 5) — correctly grounded in 2608.23023 / 2608.14641 / 2601.07206.
- **Decision extraction by one-pass scoring with a margin** (R17), grammar-constrained output with a free reasoning span (R16/CRANE), validators + bounded deterministic repair (R18).
- **FP32 router + stable lowest-index tie-break, static scales, pinned MoE kernel configs, counter-based RNG, fixed topology** (R7–R10).
- **Never ablate super experts**; pre-registered interventions with size-matched random-expert nulls, off-target ≤ ⅓, capability budget (§4.3 S5).
- **No mode-seeking on Create**; diversity from an explicit enumerated latent; selection by a locked RM.
- **Judge**: executable verifiers first, scalar RM before generative judge, both orders, expected-value rubric scores, cross-family panels.
- **R3 + routing replay in any RL stage**; thinking off by default for Decide (AbstentionBench).
- The **metric suite** (TARr/TARa, unique@N, first-divergence, PIR, pass^k, negative flips, MMD endpoint identity) and the "Csmith for prompts" synthetic-oracle idea.
- The **experiment-first culture**: the granite probe is small, reproducible, and already says the right thing about margins.

---

## VERDICT: APPROVE WITH CHANGES

Minimum changes required for approval:

1. **F1** — Qwen3-30B-A3B (text-only) becomes the v1 production base; Qwen3.6-35B-A3B becomes a P0 spike with a written GDN/MTP/vision go/no-go; P4's atlas is produced for the shipping base.
2. **F2** — Adapter-mix (and expert-mask) fuzzing added to R5/R8 and §6; P1 exit gate = D1 on base + ≥3 co-resident adapters; merged-checkpoint + batch-by-profile fallback pre-registered with its memory cost.
3. **F3** — D2 gate restated as parity ≥ 99.5% *at coverage ≥ C₀* with conformal τ and CIs; margin-regularization pilot moved to weeks 1–6 on granite/OLMoE with a pre-registered pass/fail; route pinning added as the primary non-training D2 mechanism.
4. **F4** — Speculative decoding/MTP, async scheduling, session-state hash, prefix caching, tokenizer/grammar-engine versions and sampler tie order added to the lock and the conformance matrix.
5. **F5 + F6** — A non-optional discovery deliverable (fragility map per base and adapter, fragile share → abstention, atlas diff as a P6 gate) and routing receipts → route-pinned re-execution; dense-student arm in the P3 bake-offs.
6. **F8 + F9** — Hub-and-spokes stated honestly with a spoke lock table and the Skywork license fixed; P5 baselined on R3 + FP16/TIS with bitwise RL as a costed stretch item.
7. **F11** — D1 guarantee wording downgraded, or a formal-kernel work item budgeted.

Everything under MINOR is recommended but not required for approval.

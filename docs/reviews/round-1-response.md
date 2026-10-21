# Round 1 — author response to Fable 5.1 review

**Plan version reviewed:** v1 · **Revised to:** v2 · **Date:** 2026-10-21

Disposition key: **Accepted** (adopted as proposed) · **Accepted, modified** (adopted with a change, reason given) · **Rejected** (reason given).

| ID | Disposition | What changed in v2 |
|---|---|---|
| F1 | Accepted | Qwen3-30B-A3B (text-only) is the v1 production hub. Qwen3.6-35B-A3B is a P0 spike with a written go/no-go (GDN kernels pass batch-mate, chunk-boundary and prefix warm/cold fuzzing; MTP off; vision off; GDN dictionary plan costed). P4's atlas is produced for the shipping hub. §0, §4.2, §7. |
| F2 | Accepted, modified | Adapter-mix and expert-mask fuzzing added (R8, R26, §6). P1 exit gate = D1 on hub + ≥3 co-resident adapters. Fallback ladder pre-registered: (1) **batch-by-adapter scheduling** (each forward pass contains one adapter, so mixed-adapter kernels never run; no extra memory), then (2) merged per-profile checkpoints (~61 GB BF16 each on Qwen3-30B-A3B). We put the no-memory option first; the reviewer's merged-checkpoint fallback is kept as step 2. |
| F3 | Accepted, modified | D2 is now a risk–coverage gate: parity ≥ 99.5% at coverage ≥ C₀ per profile, τ set by split-conformal calibration on a frozen set, CIs reported (§1, §6). Margin-regularization pilot moved to P0 (weeks 1–6) with the reviewer's pre-registered pass/fail; it runs on the CPU testbed now. Route pinning is adopted, **but not as the primary lever for first-time execution**: pinning needs a recorded route, which a fresh request on a different stack does not have. In v2, D2 is split: **D2-fresh** (first execution on any certified stack; levers: one-pass scoring, abstention band, margin widening, dense students) and **D2-replay** (re-execution of a recorded result on another stack with pinned routes; lever: route pinning). |
| F4 | Accepted | Lock gains library versions (tokenizers/transformers/engine), grammar engine + version, async-scheduling flag, speculative/MTP flag, prefix-cache flag (disabled in v1 lock), vision off. IR `Context` gains `session_state_hash`; no re-dispatch mid-program. Sampler tie order = stable (prob desc, token id asc). All added to the conformance matrix (R1, R6, R9, R12, §6). |
| F5 | Accepted | Non-optional discovery deliverables ND1–ND3 ship with every release (fragility map per hub and adapter; fragile-routing share wired into abstention with an acceptance test; atlas diff as a release gate). CLTs dropped from v1; S3 = residual SAEs + EAP-IG over coarse nodes (heads, experts, shared expert, router logits). Dictionaries re-validated after each adapter (§4.3). |
| F6 | Accepted, modified | Routing traces in receipts + `pin_routes` re-execution mode (R27). StableMoE router freeze/distillation added as an S5 candidate. Dense-student arm added to Decide/Call/Judge bake-offs and §4.1 now says the decision executors may end up dense with the MoE as teacher. Modification: see F3 on the limits of pinning. |
| F7 | Accepted | Control pairs: Qwen3-30B-A3B ↔ Qwen3-32B (Qwen3-8B for cheap runs); Qwen3.6-35B-A3B ↔ Qwen3.6-27B/Qwen3.8-27B. The tax is a tuple, not a number (§4.2). |
| F8 | Accepted | Architecture renamed **hub (D-MoE) + certified spokes** with a spoke table (model, license, lock, conformance, atlas scope). Skywork citation corrected to the Apache Qwen3-8B variant (RewardBench 2 78.2, JudgeBench 73.4); Llama variant only with an explicit license exception. Tongyi atlas-transfer claim deleted; super-expert re-profiling required (§4.1, §5). |
| F9 | Accepted | P5 baseline = R3 routing replay + FP32 router on both sides + FP16 or TIS/MIS correction with the sampling mask exported. Bitwise RL for the hub is a costed stretch item (custom backward passes for batch-invariant grouped GEMM/top-k/combine) or deferred to the DeepSeek-V4-Flash scale-up (§4.5, §7). |
| F10 | Accepted | Parse defaults to a small **dense** parser (Qwen3-4B/8B-class, certified as a spoke), MoE Parse profile kept as comparator. R21 adds paraphrase pick-flip rate (with ε dead band) and adversarial-suffix / keyword-injection flip tests. §8 adds router/expert-level attacks (GateBreaker, BadMoE) and forced-deferral attacks. |
| F11 | Accepted | D1 guarantee wording: "hard on certified stacks (tested; formal contract for dense kernels only)". P6 work item: extend the Vosti Triton analyzer to the MoE kernel set (grouped GEMM, top-k, permute/combine), budgeted. |
| F12 | Accepted | Numbers corrected: 1.6–2.1× naive (TML), ~1.1× co-designed (V4-Flash path at 91.3% throughput). R7 now cites the merged V4-Flash path, not draft PR #54955. Qwen3-Next status corrected. Both-order judging is a MUST. |
| F13 | Accepted | Six profiles: Parse + five task profiles. |
| F14 | Accepted | README rewritten; §9 lists real modules and tests (30 passing at time of writing). |
| F15 | Accepted | Style normalization is a function of the rubric's criterion type and recorded in the receipt. |
| F16 | Accepted | `max_tier`, `max_repair`, `max_escalations` are static constants in the dispatch table and lock. |
| F17 | Accepted | Power-of-two INT8 scales (2609.00363) and ReproFuse (M ≤ 64 limit noted) added as the cross-SKU D2→D1 upgrade path. |
| F18 | Accepted | Paraphrase ensembling (k paraphrases must agree on IR hash; disagreement → Clarify) added to R14 as an offline certification tool and optional online mode. |
| F19 | Accepted | Week-5 vertical slice is in the roadmap and already exists on CPU (`dmc.compiler` + Decide profile on granite-1b-a400m with lock hash, receipt, replay). |
| F20 | Accepted | Receipt carries sampling mask summary and routing trace reference (R27). |
| F21 | Accepted | `create.text` gets `parent_seed`/seed inheritance so revisions of an IR keep the base draft's seed schedule; documented. |

## New evidence added in this round (our own experiments)

1. **Chat templates read ambient state.** granite-3.1's chat template calls `strftime_now` and inserts today's date when no system message is given, so the same prompt is different input on different days. v2 adds R13a: templates are rendered only from explicit IR context; the backend always supplies the system turn. (`dmc/runtime.py::HFBackend.render_chat`.)
2. **Router atlas and CPU conformance experiments** (`experiments/01_router_atlas.py`, `02_cpu_conformance.py`) and the **route-pinning and margin-regularization pilots** requested by F3/F6 are running on the CPU testbed; results are folded into v2/v3 as they land.

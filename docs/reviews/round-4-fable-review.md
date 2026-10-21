# Adversarial review, round 4 — `docs/PLAN.md` v5 (GCAM: Goal-Compiled Agentic MoE)

**Reviewer:** Fable 5.1 · **Date:** 2026-10-21 · **Materials:** PLAN.md v5 (Parts I–IV); the user's A3B training report (all 52 pages); wave-2 notes W2-01…05 (all); wave-1 notes as before; `dmc/scheduler.py`, `dmc/expansion.py` and their tests (72 tests pass, run by me); pilots 06/07 with pre-registrations; `experiments/gpu/RUNBOOK.md`; results 01/02/03-partial/05. Citations: `W2-NN:line`, `A3B-ch.N` (report chapter), `file:line`.

---

## Overall assessment

v5 is an honest and well-sourced rethink. It maps every one of the user's five items to a concrete mechanism, it names the real gap (agentic, not reasoning), it carries the A3B team's methodology over intact, and it keeps the trust layer as the contract everything else runs under. The scheduler and expansion prototypes are real code with tests, not slideware. But v5 has two faults that a senior ML-systems engineer would stop on before anything else is funded. First, the **pre-registered product requirements in I.1 are not jointly satisfiable by the plan's own evidence**: tokens-per-solved-task ≤ 0.5× the 27B's and cost-of-pass ≤ 0.33× cannot coexist with "closing the gap will require *more* tokens" (W2-05:95) and with the plan's own agentic-parity mitigation (escalating hard nodes to the 27B spoke) — any 27B escalation at all breaks the 0.33× line under the plan's own cost basis — and the reasoning non-inferiority margin (δ = 1.5 points) is unpassable on AIME 2026 (30 items) and demands *superiority* on GPQA-Diamond (198 items). A plan whose premise is "write the gate down first" cannot ship gates that fail by construction. Second, **the served artifact is undefined**: item (i) merges per-goal adapters into the weights (one checkpoint per goal family, as the A3B team does), while item (ii) puts goal-gated experts "in the checkpoint" with bit-identity to a base that, after merging, no longer exists in serving; the per-request router bias, the 256+k expert count and the per-goal replicas all need engine work the plan does not budget. Beyond these, the token-economics story leans on prefix/KV reuse that the hub's own serving stack has *disabled* (vLLM corrupts Gated-DeltaNet state on cache hits — A3B-ch.4, PLAN §II.4.2), the riskiest assumption (agentic gap is capacity, not recipe) is tested last, the training compute (15–40K H100-h) has no cluster behind it, guards/recompilation exist only as an analogy, and pilot 06 lacks the baseline that would let it fail. The skeleton is right; the contract and the hub's serving design are not yet real.

---

## Fidelity to the user's brief (summary; details in findings)

| User item | v5 mechanism | Verdict |
|---|---|---|
| Groq-style compile-time scheduling | Goal IR → DAG → static ladders, budgets, worst-case bound, schedule hash (`dmc/scheduler.py`) | Substantive for ladders/budgets/bounds; decorative for data movement and lanes (M6, M18); guards unspecified (M7) |
| Goal-based MoE | Compile-time goal selects adapters / goal-gated experts / residency; token-time router within that capacity | Right idea; served artifact undefined (M2); engine support unbudgeted (M9) |
| Agentic, real work | Goal IR with verifiers, durable executor, T3 agentic RL, compaction with pinned rules | Thin on memory/state, permissions, and the internal goal suite (M12) |
| 27B parity from Qwen3.6-35B-A3B | I.1 pre-registered requirements; T1 OPD + T3 RL; Flash-Next fallback | Targets not jointly feasible and δ ill-posed (M1); riskiest assumption tested last (M4) |
| (i) train discovered experts | Hot-expert LoRA per goal family (A3B rule), ESFT/experts-only ablations | Delivered; conflicts with (ii) in serving (M2); atlases go stale after T1 (M11) |
| (ii) new layers + router | Goal-gated new experts, routed/additive, compile-time bias (`dmc/expansion.py`) | Delivered as prototype; "new blocks" correctly demoted with evidence; pilot cannot test the distinctive claim (M8) |
| (iii) specialist ensemble | Spokes + selection-over-fusion policies | Delivered |
| (iv) System-One models | Decision heads, encoders, code executors with conformal abstention | Delivered |
| (v) classic ML/DL | GBDT/TabPFN/Chronos/rerankers as executors | Delivered as a table; integration contract thin (M12) |
| Token economics | I.3 mapping + context policy | Prefix caching assumed but disabled on the hub (M3); targets over-reach (M1) |

---

## Findings

### BLOCKER

#### M1 — The pre-registered requirements (I.1) are mutually inconsistent and partly unpassable
**Where:** PLAN §I.1 (Reasoning parity δ = −1.5; Token economics ≤ 0.5× tokens, ≤ 0.33× cost-of-pass, ≤ 0.5× latency; "separately for the raw hub and for the compiled system"), §I.7 risk row 1 (mitigation = escalate nodes to the 27B spoke).
**What is wrong.**
1. *Tokens vs parity.* The plan's own note says Qwen3.6-35B-A3B is already *less* verbose than the 27B (160M vs 200M output tokens on the AA index) and that "closing the gap will probably require *more* tokens per task on agentic work" (W2-05:95); W2-02:179–184 says token-efficiency methods buy −25–50% tokens at a 0–2 pp *loss* on hard sets, while the hub needs a +3 to +10 pp *gain*. A raw-hub requirement of ≤ 0.5× the 27B's tokens *at parity* is therefore contradicted by the evidence the plan cites. 
2. *Cost-of-pass vs escalation.* Let the hub's per-token cost be c_h = c_t/3 (AA throughput 134 vs 44 tok/s, W2-05:64) or c_t/6 (A3B GPU-time per turn 0.075 vs 0.449 s, A3B-ch.9). At equal token volume, cost-of-pass ratio = 1/3 + T_esc/T_27B (or 1/6 + T_esc/T_27B). The ≤ 0.33× line is met only with **zero** 27B escalation on the throughput basis, or ≤ 1/6 of the 27B's token volume on the most optimistic basis. But §I.7's mitigation for the Medium–High agentic risk is precisely to escalate hard nodes to the 27B. The two requirements cancel each other; the plan never writes down the cost basis or an escalation budget.
3. *δ = 1.5 is unpassable or demands superiority.* AIME 2026 has 30 problems: one problem is 3.3 points and the paired SE on 30 items is ≥ 8 points, so a lower-CI bound ≥ −1.5 cannot occur. GPQA-Diamond has 198 items; with 20% item-level disagreement the paired SE is ≈ 3.2 pts (CI half-width ≈ 6), so passing needs a point estimate of about +4.5 — superiority, not non-inferiority. The agentic row at least scales δ by the noise band; the reasoning row does not.
4. *Predictability "never exceeded"* is asserted for schedules whose inputs include unbounded tool outputs (M6) and bounded-but-unspecified recompiles (M7); "realized within ±15%" is asserted for suites where the plan's own note reports 30× run-to-run token variance (W2-02:391).
**Fix (before any pre-registration is frozen or GPU-hour spent).** (a) Write a one-page feasibility model: declare the per-token cost basis (measured, not AA), the expected token ratio at parity per suite, and an **escalation budget** (max share of tokens on the 27B) consistent with the cost target; derive the targets from it. Likely outcome: raw-hub tokens ≤ 1.0× at parity on reasoning suites, ≤ 0.5× only for the compiled system on the goal suite; cost-of-pass ≤ 0.5× with a ≤ 15% escalation budget, ≤ 0.33× only with no escalation. (b) Set δ per benchmark = max(1.5 pts, one item, measured 27B-vs-itself SD) with paired avg@k (k ≥ 16) and item-level bootstrap; pool AIME 2024/25/26 or drop AIME from the gate. (c) Scope predictability: the bound holds by construction *given* compiled tool-output caps and the recompile allowance; ±15% applies to suite means over ≥ 200 tasks, with the bound-hit rate reported.

#### M2 — The served artifact is undefined: merged per-goal adapters, goal-gated experts and "one hub" cannot all be true
**Where:** §I.2.3 items 1–2, §I.0.5, §I.7 ("goal experts are real experts in the checkpoint, not runtime adapters"), §II.4.1/R26.
**What is wrong.**
- Item 1 merges each goal family's hot-expert LoRA into the weights for serving ("LoRA-on-experts serving is fragile", W2-03:324–329, 371–376; the A3B team serves a merged FP8 checkpoint, A3B-ch.3). With G goal families that is **G merged checkpoints** (72 GB bf16 each on this base, A3B-ch.2; ~36 GB FP8), each with *different* expert weights.
- Item 2 says goal-gated experts live "in the checkpoint" and are "bit-identical to the base when the goal is off". Which checkpoint? If merged into family A's checkpoint they are not in family B's; if kept in a single un-merged base, item 1's adapters are not merged. The identity guarantee is proven (`tests/test_expansion.py`) relative to a base that serving no longer contains.
- "One hub, one atlas, shared serving" (§II.4.1) is then G hubs. "Goal selects adapters" at compile time means a **model switch per node**, which the plan elsewhere says loses the prompt cache (§I.2.4; W2-02:415). The expert-residency plan (§I.2.1) silently assumes one resident model.
- The engine side is unbudgeted: a per-request (per-token-row) router bias, `num_experts = 256 + k` (EP divisibility, re-sharding, FP8 scales for new experts — W2-03:376–377), and R3 routing replay over a custom router (M9).
**Fix.** Choose and state the served artifact, with its memory and cache consequences, before T1 starts. The options the evidence supports: (A) **one checkpoint per goal family, one replica per GPU, nodes pinned to replicas, cross-family edges pay full prefill** — simplest, matches A3B practice; goal-gated experts then belong to that family's checkpoint and "goal off" means "not routed there", not bit-identity to a base; (B) **one checkpoint**: express item (i) *as* item (ii) — UMoE-style in-place reallocation or new experts carrying the family's delta — so every family's capacity is goal-gated inside one 256(+k)-expert model with a frozen base; identity-when-off then holds in serving; requires the engine work in M9; (C) multi-LoRA serving (rejected by the plan's own evidence). Write the choice into §I.2.3 and R26, and re-derive the §I.3 cache claims from it.

### MAJOR

#### M3 — The token-economics lever the plan leans on hardest is disabled on the chosen hub
**Where:** §I.3 rows "Data movement planned ahead" (1.8–3.6× from KVFlow/Halo; "cached input ~10× cheaper") and §I.2.1 "prefix groups placed back-to-back so shared context is computed once"; §II.4.2 ("Production also disables prefix caching on this hybrid (vLLM bug restoring corrupted linear-attention state; A3B report)"); A3B-ch.4 ("garbage tokens after ~8.6k cached tokens"), ch.2 (vLLM disables prefix-cache reads when prompt log-probs are requested).
**What is wrong.** Every KV-reuse gain in §I.3 presupposes a working prefix cache on a Gated-DeltaNet hybrid. The hub's own production stack turns it off. Nothing in G0–G4 fixes or validates it, and the I.1 token targets implicitly assume it.
**Fix.** Add a G0 engineering gate: "GDN prefix caching validated (state checkpointing at chunk boundaries) on the pinned vLLM version, with the A3B 40-example cached/uncached/repeat probe extended to ≥ 8.6k-token prefixes." Until it passes, remove KV-reuse gains from the expected-effect column and from the feasibility model (M1); keep prefix groups in the scheduler as a *placement* preference only.

#### M4 — The riskiest assumption (the agentic gap is capacity, not recipe) is tested last
**Where:** §I.0.1 ("about two-thirds of the reasoning gap comes from newer post-training, which can be transferred"), §I.2.3 ("same-generation MoE-vs-dense loss is small"), §I.6 (agentic gate only at G3, weeks 10–24).
**What is wrong.** The two-thirds decomposition holds for reasoning (W2-05:89–93). For agentic work the *same-generation* MoE-vs-dense gap is the majority: SkillsBench −19.5, Claw-Eval pass^3 −10.6, Terminal-Bench 2.0 −7.8, NL2Repo −6.8 (W2-05:44); on Qwen3.5 the pair is 4.4 vs 27.2 on SkillsBench. The note itself says "the agentic gap has a large architecture/capacity component" (W2-05:10). "Evidence-backed" in §I.2.3 is selective. The only cheap, direct evidence that RL moves this class — Qwen-AgentWorld's +4 to +14 pts from ~100 terminal-RL steps on Qwen3.5-35B-A3B (W2-05:253–257) — is not scheduled until G3.
**Fix.** Pull a ~100-step terminal/SWE RL spike (AgentWorld-style, R3 on, 4×B200-class as in the Agent Lightning example, W2-03:285) into G0–G1 as the agentic de-risk, with a pre-registered gate; define the Flash-Next switch criteria now (and note its `qwen-community-1.0` license and 51B host-RAM n-gram tables, W2-05:18, which §I.7 omits); report the escalation share needed for system parity at G1, not G3.

#### M5 — The compute plan has no training cluster and the timeline does not fit it
**Where:** §I.4 (T3 10–30K H100-h; T1 1–3K; T5 1–2K), §I.6 compute line ("Serving for evaluation: one 8×H200 node"), roadmap of 30 weeks.
**What is wrong.** 15–40K H100-h on one 8-GPU node is 80–210 days of wall clock, before environment engineering, which the plan itself calls the larger cost. The A3B precedent is one shared H200 for 13.7 GPU-hours per run; v5 is two to three orders of magnitude larger and says so nowhere.
**Fix.** State the training allocation (nodes × weeks) per stage, or scale T3 to what exists (e.g. AgentWorld-scale 100-step runs on 5–15K environments) and re-derive G3's gate from that; add "environment construction" as a line item with headcount.

#### M6 — The scheduler's cost model is the right shape but mis-specified in four places
**Where:** `dmc/scheduler.py:111–145` (ladder accuracy), `:167–200` (prefix discount), `:218–224` (hash); `experiments/07_system1_cascade_prereg.md`.
**What is wrong.**
1. *Correlated errors across rungs.* Ladder accuracy = Σ reach·coverage·acc + reach_last·acc_all uses each executor's *marginal* accuracy on the items the previous rung *abstained on*. Abstained items are harder, so the next rung's conditional accuracy is lower; the model systematically over-predicts. The 07 pre-registration will see this as "realized below the Wilson band" but attaches no consequence. The calibration split already runs every executor on every item, so the conditional accuracy P(correct_j | abstain_{<j}) is directly estimable.
2. *No input-token model for agent nodes.* `tokens_in` is a constant per executor; a node whose input is a tool output has unbounded input unless a cap is compiled. The "worst-case bound never exceeded" claim (I.1) therefore does not hold by construction for agent graphs.
3. *Prefix discount is applied per (lane, group), not per executor.* A ladder that escalates from the hub to the 27B still gets the hub's prefix discount on the spoke rung (`:181–186`), although the cache does not transfer across models (W2-02:415).
4. *Schedule hash omits the lock hash*, although R28 defines the schedule as a function of (goal IR, lock, cost models).
Also: no per-node difficulty (one ladder per op type), whereas W2-02:261–269 shows history-aware per-step routers beat static effort; the compile-time decision head (§I.2.3 item 4) could supply a difficulty bucket as a node attribute.
**Fix.** Conditional cost models per rung position (or difficulty bucket) estimated on the joint calibration set; compiled tool-output caps per node (enforced by truncation) so the bound is real; executor-keyed prefix groups; lock hash in the schedule hash; `Node.difficulty` from a compile-time probe. Amend 07 to compute and use conditional accuracies and to state the consequence of a calibration-transfer failure.

#### M7 — Guards and recompilation are an analogy, not a specification
**Where:** §I.2.1 "Guards instead of open-ended replanning"; nothing in `dmc/scheduler.py` or tests.
**What is wrong.** Data-dependent branches are the entire difficulty of agent compilation (W2-01:211, SGLang's tracer is "limited to programs without data-dependent control flow"). v5 says guards are typed predicates, variants are pre-compiled, recompiles are counted and budgeted — but defines no guard language, no variant set, no bound semantics (does the worst-case token bound cover all variants? the recompile budget?), no behaviour when the budget is exhausted (it should be D⊥ `Abstain`, not a silent loop), and no tests.
**Fix.** Specify in §I.2.1 and implement in the scheduler: `Guard(node, predicate over typed tool-output IR, on_fail → variant_id | recompile)`, variants compiled with their own bounds, schedule worst case = max over variants + recompile allowance, exhaustion → `Abstain` receipt; tests for bound monotonicity and for COBRA-style branch constraints (capabilities fixed per branch at compile time).

#### M8 — Pilot 06 cannot test what item (ii) uniquely claims, and it tests a different initialization than the plan prescribes
**Where:** `experiments/06_goal_gated_experts_prereg.md`; `dmc/expansion.py:72–76` (`router_new = zeros`, no noise); PLAN §I.2.3 item 2 ("router rows initialized from the parent row", "+ noise").
**What is wrong.**
1. *No baseline that isolates new capacity.* Both arms train new experts + rows + bias on 2,700 labelled items and are compared only with the untouched base. A hot-expert LoRA arm at matched trainable parameters (item i), or a router-only arm (bias + rows, experts frozen at the parent copy), would show whether *new capacity* adds anything over ordinary fine-tuning. Without them a +10-point gain is uninformative — the 03 partial results show the base near chance on several oracle families, so any supervised signal will clear +10.
2. *Init mismatch.* The plan prescribes parent-row init with symmetric noise (UMoE, W2-03:40–41, 53–54); the prototype zero-initializes `router_new` and adds no noise, so the routed variant starts with new-expert logits exactly 0 — the "one-sided" regime UMoE found starves copies.
3. *No wrong-goal test.* Compile-time goal assignment will sometimes be wrong (Parse errors); the pilot measures goal-on drift on wikitext but not goal-on accuracy on a *different* decision family.
4. *In-template generalization only*, acknowledged; fine for a mechanism test, but the prereg should say the result cannot speak to the production claim.
**Fix (amend before running).** Add arms: hot-expert LoRA r16 at matched parameters; router-only; random-parent warm start. Success = routed/additive ≥ LoRA arm with CI, plus identity and drift guards. Align init with the plan (parent row + σ = 0.05·std noise on both parent and copy) or change the plan. Add a wrong-goal accuracy measure.

#### M9 — Goal-gated experts need engine work the plan does not budget, and "active compute unchanged" is true for only one variant
**Where:** §I.2.3 item 2; §I.2.1 expert residency; W2-03:324–329, 371–379.
**What is wrong.** Serving needs (a) a per-token-row router-logit bias indexed by the request's goal (DeepSeek's `e_score_correction_bias` is static per model, not per request), (b) `num_experts = 256 + k` with EP divisibility and new FP8/NVFP4 scales, (c) EP hot-spotting on goal experts that receive concentrated traffic, (d) R3 replay masks over the extended router during T3. This is the same class of work as Part II's external-route entry point (R27), and larger. The *additive* variant also runs n_new extra experts for every token (like the shared expert): with one new expert that is ≈ +11% routed-FFN FLOPs per layer, so "active compute unchanged with top-8 fixed" holds only for *routed*.
**Fix.** Budget the engine work in G1 next to R27; state the FLOP cost per variant; pick k so 256 + k divides the EP degree; carry EP load-balancing (expert replication) as a risk.

#### M10 — Thinking control is scheduled before it is known to work on this base
**Where:** §I.2.3 item 3 ("the scheduler sets `enable_thinking` and a thinking budget enforced at decode time"), §I.6 G1 (compiler v1 weeks 2–8) vs T5 (G2).
**What is wrong.** Qwen3.6 has no `/think` soft switch; `enable_thinking` is a chat-template kwarg that changes the rendered prefix (W2-02:24–31), with an open question whether per-turn toggling keeps the KV prefix byte-identical (W2-02:286); the card gives no thinking-budget guidance, and Qwen3's budget forcing is an *emergent* property of its fusion training (W2-02:51–55), not a documented Qwen3.6 feature. Compiler v1 assumes both levers exist; T5, which would train them, comes later.
**Fix.** G0 measurement: accuracy-vs-budget curve for Qwen3.6-35B-A3B under hard truncation + "answer now" prompt, and the template/KV effect of per-turn `enable_thinking`. Until T5 lands, compiler v1 uses only on/off per node plus early-exit (DEER/CALM-style, training-free) and reports realized tokens.

#### M11 — Stage interactions are unmanaged: atlases, warm starts and System-1 heads all go stale
**Where:** §I.4 T0–T6 ordering; §II.4.3 ND3.
**What is wrong.** T1 trains attention/GDN/shared-expert LoRA (unless the experts-only ablation wins); even with the router frozen, that changes what the router sees — the A3B team measured hot experts *after* applying the attention adapter for exactly this reason (A3B-ch.2). So T0's hot-expert sets and super-expert profile are stale after T1, T2's "hottest base expert" warm start must be chosen on post-T1 weights, T3 RL moves experts again, and T6's decision heads and every executor cost model are calibrated on hub states that T5 then changes. Part II's ND3 atlas diff is a release gate, but Part I never says "re-atlas after T1/T3; re-calibrate T6 after any hub change".
**Fix.** Make re-atlasing (ND1) a required step after T1 and T3, warm-start selection on the weights that will be expanded, and T6 (heads + cost models) a per-hub-release step that runs *last* and is re-run on any hub change; add these to §I.4 as explicit edges.

#### M12 — Missing pieces for an agent that does real work
**Where:** §I.2.1 Goal IR, §I.5, §I.6.
**What is wrong.**
- *Memory and state.* The Goal IR has context references and the Part II session hash, but no durable memory model: what persists across nodes and sessions (files, repos, tickets, scratchpads), how memory *writes* are recorded side effects with idempotency keys (R22) and how replay (D0) treats them, and how "file system as memory" (Manus, W2-02:444) interacts with compaction and the pinned-rules region.
- *Tool safety.* An autonomy level exists, but there is no per-tool capability lattice (read / write / destructive / external-send), no dry-run or confirmation policy for irreversible actions, no secrets handling, and COBRA's per-branch capability constraints are cited but not placed in the compiler.
- *The goal suite.* I.1 and I.5 evaluate on "an internal goal suite" that is never defined (domains, size, verifiers, who labels, the product the five-rung ladder measures through). The A3B method worked because the product and the 144 test calls existed.
- *Classic-ML integration contract.* Item (v) is a table; nothing says how a GBDT/TabPFN executor is registered (feature schema compiled from IR slots, calibrated outputs, versioned cost model) or how its result re-enters the typed IR.
**Fix.** Add to the Goal IR and the scheduler: memory ops as typed, recorded activities; a capability lattice per node fixed at compile time; a G0 deliverable defining goal-suite v0 (≥ 100 goals across the goal families, each with an executable verifier or two-judge rubric) and the product surface; an executor-registration contract (schema in, calibrated probability out, cost model, contract D0/D1).

### MINOR

#### M13 — Part II is stale after the hub change. §II.4.3 S3 still says "Qwen-Scope SAEs, all 48 layers" (Qwen3-30B-A3B); the hub has 40 layers and Qwen-Scope covers Qwen3.5-35B-A3B-Base, not 3.6. R27 says "48 × 8 = 384 B/token on the hub" (now 40 × 8 = 320 B). R26 says "6 × ~61 GB" (72 GB bf16 for this base, A3B-ch.2). §II.7 P0 still lists "Qwen3.6-35B-A3B go/no-go spike" as a spike rather than the hub's audit. §II.8 "Base-model obsolescence (Qwen3-30B-A3B is a 2025 model)" is now about the wrong model.

#### M14 — Status contradictions: §I.0.4 and §I.6 say pilots 06/07 are "running"; Part III says "queued after 03" / "queued after 06". Pick one (Part III is right).

#### M15 — Misread number: "MoE-LPR: skipping [the review pass] dropped retention to 47%". 47.14 is the original-language *average score* without review vs 52.12 with it (retention ≈ 87% vs 96.6%, W2-03:126–138).

#### M16 — Secondary figures presented as fact: Qwen3-4B 86.7% BFCL Live (W2-04 marks it secondary), GLiNER2's 0.72 zero-shot classification (secondary), Flash-Next AA index 56 (unverified). Mark them.

#### M17 — The A3B result is cited without its own caveats. The report recommends "a canary, not a straight swap" because the 144 test calls were reused across rounds and two gates were missed (A3B-ch.9). §I.0.2 should carry that, since the plan reuses the same evaluation method.

#### M18 — The Groq mapping table stacks effects from unrelated settings (3.7× latency from LLMCompiler, 1.8–3.6× from KVFlow/Halo, −50% from plan caching, "36–97% of slow-path tokens" from S1-MAS's multi-agent coordination setting) as if additive. W2-01:295–300 says they decompose; it does not say they multiply. Label the column "upper bounds from different settings" and let I.1's ±15% audit be the claim.

#### M19 — MTP speedup on a GDN hybrid is unmeasured (state rollback on rejected drafts, W2-02:362). §I.2.3 item 5's 1.8× is borrowed from attention models; measure in G0 and keep MTP off in the D1 lane (R6) as already stated.

#### M20 — Thinking-off cost: "turning thinking off costs Qwen3-30B-A3B ~49 AIME points" is right (AIME'25 70.9 → 21.6); say which AIME, since AIME'24 is 47.6.

#### M21 — §I.2.3's "evidence-backed" base choice: the same-generation evidence is reassuring for reasoning only; say so (ties to M4).

---

## What v5 gets right — do not change

- The two-engine framing: capability transfer (OPD from the 27B, agentic RL with R3, decision-token DPO) **plus** compilation; and the plain statement that token efficiency cannot close an accuracy gap.
- The A3B methodology carried over verbatim: pre-registration with dated amendments, paired conversation-level bootstrap, null controls, checkpoint selection by behaviour probes, evaluation through the product, "every check must be able to fail", two-judge labels and "agreement is not truth".
- The mapping of the user's five items to five mechanisms, with the honest demotion of new *blocks* (latency; lost to MoE-LPR) and of expert *masks* (Phase-0 NMI ≤ 0.04).
- Executors declare contracts and nodes declare requirements (R28); contract-aware scheduling; `CompileError` on an over-budget graph; the schedule as a content-addressed artifact.
- Goal-off bit-identity for new experts (tested), the experts-only / ESFT-tight ablations, and the refusal to re-impose load balancing.
- Selection over fusion; Self-MoA over mixed MoA; same-model escalation by default and per-node spokes with their own prefix group.
- Cost-of-pass as the metric; raw-hub vs system reporting as mandatory; constraint pinning outside the compacted region; budgets from a cost model, never from the model's self-estimate.
- The CPU prototypes are real, tested code; the GPU runbook exists.

---

## VERDICT: APPROVE WITH CHANGES

Minimum changes required for approval (M1 and M2 before any pre-registration is frozen or any training GPU-hour is spent; without them the verdict would be REJECT):

1. **M1** — Replace I.1 with targets derived from a written feasibility model (cost basis, token ratio at parity per suite, escalation budget); per-benchmark δ scaled to item count and measured noise; predictability scoped to compiled caps + recompile allowance.
2. **M2** — Define the served artifact (per-family merged checkpoints pinned to replicas, or one goal-gated checkpoint with the engine work budgeted), and restate "one hub / one atlas", residency and cache claims accordingly.
3. **M3** — G0 gate for GDN prefix caching; remove KV-reuse gains from the expected effects until it passes.
4. **M5** — State the training allocation per stage or rescale T3; add environment construction as a line item.
5. **M6** — Conditional (escalated-subset) accuracies in the cost model and in 07; compiled tool-output caps; executor-keyed prefix groups; lock hash in the schedule hash.
6. **M7** — Guard/variant/recompile semantics specified and implemented with tests; exhaustion → `Abstain`.
7. **M8** — Pilot 06 amended before running: LoRA and router-only arms, plan-aligned initialization, wrong-goal measure.

M4, M9–M12 should be resolved before G1 begins; M13–M21 are text fixes for the next revision.

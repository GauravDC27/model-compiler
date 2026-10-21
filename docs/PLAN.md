# Goal-Compiled Agentic MoE (GCAM): Architecture, Training and Implementation Plan

**Version:** v6 (v5 revised after review round 4; v5 was the rethink after the user's 2026-10-21 brief and built on v4, which converged in three Fable 5.1 review rounds) · **Date:** 2026-10-21 · **Status:** under adversarial review (round 5)
**Review trail:** `docs/reviews/round-1…3-*` (v1–v4) → round 4 on v5 (`docs/reviews/round-4-fable-review.md`, response `docs/reviews/round-4-response.md`) → round 5 on this version.

> One sentence: a **goal compiler** turns a request into a typed task graph and a **static schedule** (Groq-style: every step's executor, token budget, escalation ladder, guards and worst-case cost decided before anything runs); the schedule is executed by an **assembly of executors** — System-1 classifiers and classic ML, compiled code, a **Qwen3.6-35B-A3B hub** whose goal families get discovered-expert adapters and **goal-gated new experts**, and specialist spokes led by Qwen3.8-27B — under the **compiler-grade trust layer** of Part II.

Evidence: `research_notes/Deterministic MoE model compiler/` (wave 1, notes 01–07), `research_notes/Goal compiled agentic MoE/` (wave 2, notes 01–05, cited as W2-NN), the user's A3B training report (`A3B_Training_Explained_2026-10-21.pdf`, cited as A3B-§ or A3B-ch.N), and our CPU experiments (Part III). Figures marked *(secondary)* come from a secondary source and have not been checked against the primary one.

---

## I.0 Executive summary

1. **The target is quality parity with Qwen3.8-27B at a fraction of its cost, and the gap to close is mainly agentic, not reasoning.** Model cards: GPQA 86.0 vs 89.2, LiveCodeBench v6 80.4 vs 90.3, HLE 21.4 vs 30.8, SWE-bench Pro 49.5 vs 61.7. Artificial Analysis: Intelligence Index 18 vs 34, AutomationBench 5.2% vs 48.2% (W2-05). On reasoning, same-generation dense-vs-MoE costs only ~1–4 points and about two-thirds of the gap comes from Qwen's newer post-training (3.6 → 3.8), which can be transferred. On agentic suites the same-generation MoE-vs-dense gap is the larger part (SkillsBench −19.5, Claw-Eval pass^3 −10.6, Terminal-Bench 2.0 −7.8, NL2Repo −6.8; W2-05), so the plan tests early whether training moves it (G0–G1 agentic spike, §I.6). Token efficiency alone cannot close an accuracy gap, and closing this one will probably need *more* hub tokens on agentic work, not fewer (W2-05).
2. **So the plan has two engines:** (a) **capability transfer** into the hub — on-policy distillation from Qwen3.8-27B (shared 248,320-token vocabulary makes full-logit distillation feasible), agentic RL with routing replay, decision-token DPO — on top of the user's A3B recipe (hot-expert LoRA, cached-prefix training, two-judge labels; v3c beat the 27B by +112 turn-equivalents through the product at 330 vs 785 ms; the report itself recommends a canary rather than a straight swap, because its 144 test calls were reused across rounds and two gates were missed, A3B-ch.9 — GCAM inherits the method and therefore the caveat); and (b) **cost reduction by compilation** — a goal compiler that assigns each step the cheapest calibrated executor, bounds worst-case tokens at compile time, and escalates only on calibrated abstention. Cost targets are derived from a written feasibility model (§I.1.1), not asserted.
3. **Groq's lesson, transferred:** Groq removes runtime arbitration and lets the compiler decide where and when every operation runs, so latency and cost are known in advance (W2-01). Cycle-exact timing does not transfer to agents (output lengths are data-dependent), but the *discipline* does: whole-plan analysis, resources reserved ahead of time, the plan as single source of truth, explicit guards where data-dependence is unavoidable (implemented, §I.2.1). Compile-time planning in LLM systems is well evidenced (LLMCompiler, Agentic Plan Caching, KVFlow/Halo, Dyserve; W2-01); their effects come from different settings and are upper bounds, not additive (§I.3).
4. **The user's five items map onto five mechanisms:** (i) *train discovered layers/experts* = goal-family hot-expert adapters from routing atlases (A3B recipe; ESFT-tight variants; `dmc/expert_lora.py`); (ii) *new layers + router* = **goal-gated new experts** whose router rows carry a per-goal bias set by the compiler (bit-identical to the base when the goal is off — tested; CPU pilot 06 queued, with LoRA, router-only and random-parent baselines so it can fail); (iii) *specialized models + ensemble* = certified spokes and verifier-selected ensembles (selection beats fusion: fused answers were never preferred in 0/42 tasks; Self-MoA beats mixed MoA by 6.6%; W2-04); (iv) *System-One models* = hidden-state decision heads, encoders, small SLMs and compiled code that answer without generating tokens and abstain to System-2 when unsure (encoders match zero-shot LLMs on fixed label sets at ~1000× lower cost; W2-04); (v) *classic ML/DL* = GBDTs and tabular foundation models for tables (LLMs lose to GBDTs on tabular data), Chronos-2/TimesFM for forecasting, embedding + kNN, rerankers (W2-04), all registered through one executor contract (§I.2.2).
5. **Goal-based MoE** means routing is conditioned twice: at compile time (the goal selects the family checkpoint or the family's goal-gated experts, and expert residency) and at token time (the learned router within that goal's capacity). The compiler, not the model, decides which capacity is in play — a static decision that is cheap, auditable and replayable. **What is served is defined in §I.2.3:** per-family merged checkpoints pinned to replicas from G1 (Artifact A), and one goal-gated checkpoint as the target if it passes a pre-registered comparison at G2 (Artifact B).
6. **Trust stays compiler-grade.** Every executor declares the determinism contract it delivers (D0 replay / D1 bitwise, Part II); nodes declare what they require; the scheduler only assigns compliant executors (`dmc/scheduler.py`, tested). Until Qwen3.6's Gated-DeltaNet kernels pass a batch-invariance audit, D1-requiring nodes run on D1-certified executors (System-1 heads, dense students, code), and the hub serves D0 + statistical D2.

---

## I.1 Product requirements, derived from a feasibility model

### I.1.1 Feasibility model (the formula is fixed now; G0 plugs in measured numbers by dated amendment, before any GCAM run)

Per suite, normalize every quantity by the 27B's tokens per task T₂₇ (input + output; cached input counted separately):

- τ_h = hub tokens per task ÷ T₂₇ (the hub's verbosity relative to the 27B);
- e = tokens the 27B spends on escalated nodes ÷ T₂₇ (the **escalation budget**);
- s = System-1 / code / classic-ML cost in 27B-token equivalents (≈ 0 at this precision);
- r = c_h ÷ c_t, the hub's serving cost per token over the 27B's, **measured in G0** on our node at the production batch, input and output separately. Priors: 1/3 from Artificial Analysis throughput (134 vs 44 tok/s, W2-05), 1/6 from the A3B report's GPU time per turn (0.075 vs 0.449 s, A3B-ch.9).

At accuracy parity, **cost-of-pass ratio = r·τ_h + e + s**, and with escalations run serially, **latency ratio ≈ τ_h·(v_t / v_h) + e** (v = decode throughput). No KV/prefix-reuse credit is taken until gate G0-3 passes (§I.3).

Evidence for τ_h: the hub is already less verbose than the 27B (160M vs 200M output tokens on the AA index, τ ≈ 0.8 before training), but closing the gap probably needs more tokens on agentic work (W2-05), and length-efficiency methods buy −25–50% tokens at a 0–2 point *loss* on hard sets (W2-02). The model therefore plans for τ_h ≈ 1.0 on reasoning suites and τ_h ≤ 1.2 on agentic suites.

| Scenario | r | τ_h | e | Cost-of-pass ratio |
|---|---|---|---|---|
| AA cost basis, no escalation | 1/3 | 1.0 | 0 | 0.33 |
| AA cost basis, 15% escalation | 1/3 | 1.0 | 0.15 | 0.48 |
| AA cost basis, agentic verbosity, 15% escalation | 1/3 | 1.2 | 0.15 | 0.55 |
| A3B cost basis, 15% escalation | 1/6 | 1.0 | 0.15 | 0.32 |

Consequences, used below: cost-of-pass ≤ 0.5× is reachable with an escalation budget e ≤ 0.15 on the AA basis when τ_h ≤ 1.05; ≤ 0.33× only without escalation (AA basis) or on the A3B basis; tokens per solved task ≤ 0.5× is not a raw-hub target at all, only a compiled-system target where System-1 executors and code take whole nodes off the hub. If G0 measures r > 0.4, the cost target is re-derived from the same formula by dated amendment before G1.

### I.1.2 Requirements

| Axis | Requirement | Measured as |
|---|---|---|
| **Reasoning parity** | GCAM (the compiled system) non-inferior to Qwen3.8-27B at `reasoning_effort=xhigh` on GPQA-Diamond, HLE (no tools), AIME 2024 + 2025 + 2026 pooled (90 problems), LiveCodeBench v6, MMLU-Pro: lower 95% CI of (GCAM − 27B) ≥ −δ_b on each benchmark b, with **δ_b = max(1.5 points, one item = 100/n_b points, 2.8·SE_b)**, where SE_b is the paired standard error of the 27B against itself (two independent avg@16 runs) measured in G0. The 2.8·SE_b term gives a system exactly as good as the 27B ≥ 80% power to pass; without it the gate is unpassable on small suites (AIME alone: one problem = 3.3 points) or demands superiority (GPQA-Diamond, 198 items). Every δ_b is published before any GCAM run. | avg@k per item with k ≥ 16 samples; paired item-level bootstrap; same harness and prompts; both models served by us. |
| **Agentic parity** | Non-inferior with δ_b defined the same way (floor 2 points) on SWE-bench Pro, Terminal-Bench 2.x, τ³-bench (pass^k, k = 4), BrowseComp, and goal-suite v0 (§I.5); AutomationBench/GDPval tracked. | pass^k and pass@1; null control = 27B vs itself (A3B-§5 method). |
| **Cost** | Compiled system: cost-of-pass ≤ 0.5× the 27B's on every suite, with escalation ≤ 15% of the 27B's token volume (e ≤ 0.15); ≤ 0.33× is a stretch reported for the no-escalation configuration. Raw hub: tokens per solved task ≤ 1.0× the 27B's on reasoning suites at parity (τ_h reported per suite). Compiled system on goal-suite v0: tokens per solved task ≤ 0.5×. | Measured cost basis r (G0-2); input / cached / output tokens; $ at a fixed price sheet. |
| **Latency** | Compiled system p50 task latency ≤ 0.6× the 27B's (from 0.33·τ_h + e at the AA throughput ratio). | Wall clock on the same node. |
| **Predictability** | (a) *By construction:* every compiled program has a worst-case token bound — main graph + most expensive pre-compiled variant + recompile allowance × per-recompile reserve, with every tool output truncated to its compiled cap (`dmc/scheduler.py::compile_program`); the runtime abstains rather than exceed it. (b) *Statistical:* realized mean tokens within ±15% of the schedule's prediction on suite means over ≥ 200 tasks; the per-task ratio distribution and the cap-hit rate (share of tasks in which any cap truncated) are reported, since agent runs of the same task vary up to 30× in tokens (W2-02). | Program bounds; experiment-07-style predicted-vs-realized audit with joint calibration. |
| **Trust** | Part II contract per node; D1 for nodes that require it; schedule determinism (same goal IR + lock + cost models → same schedule hash). | Part II conformance suite. |

Raw-hub vs system reporting is mandatory (A3B lesson 5: "measure where the user is").

---

## I.2 Architecture

```
Goal (NL) ─► Goal compiler ───────────────────────────────────────────────────────────────┐
              1. Parse → Goal IR (objective, deliverable, verifiers, constraints, budgets,  │
                 autonomy → capability ceiling, memory refs, context refs)  [Part II R12–R15]
              2. Plan → task DAG of IR ops (plan cache keyed by goal-IR template hash)     │
              3. Schedule → executor ladder, token budget, tool-output caps, lane, prefix   │
                 group, family replica, guards + variants + recompile allowance,           │
                 worst-case bound, schedule hash (includes the lock hash)                   │
                                                                                            ▼
Runtime (durable, recorded): lanes ▸ caps ▸ guards ▸ verifiers ▸ memory activities ▸ receipts
   │
   ├─ System-1 tier: compiled code · classic ML · encoders · decision heads on hub states · SLMs
   ├─ Hub (System-2): Qwen3.6-35B-A3B hub-base + goal-family capacity (Artifact A or B, §I.2.3)
   ├─ Spokes: Qwen3.8-27B (teacher / hardest-node escalation / judge) · reward model · verifiers
   └─ Ensemble policies per node: cascade · verifier-selected best-of-K · confidence-filtered parallel thinking
```

### I.2.1 Goal compiler

- **Goal IR** extends the Part II IR (`dmc/ir.py`) with an objective, a deliverable type, **success criteria as verifiers** (tests, checks, rubric + judge), budgets (tokens, latency, $), an autonomy level (which compiles to a capability ceiling, below), memory references and context references (snapshotted, R23). Ambiguity is a compile error (`Clarify`), as in Part II. The memory ops and Goal IR fields are IR v1; IR v0 has the twelve Part II ops.
- **Planner** = the hub's *Plan* profile (trained on decomposition traces; AgentFlow-style planner training reports +14–15% at 7B, Kimi's orchestrator-only RL BrowseComp 60.6 → 78.4; W2-05). Output: a DAG of typed IR ops. **Plans are cached** by goal-IR template hash (Agentic Plan Caching: −50% cost, −27% latency in its setting; W2-01) and reused with slot substitution.
- **Scheduler** (`dmc/scheduler.py::compile_schedule`, implemented and tested): for each node, the cheapest executor ladder (any cost-ordered subsequence of capable executors, up to `max_rungs`) whose expected accuracy meets the node's bar and whose determinism contract meets the node's requirement. **Ladders are priced on joint calibration data** when it exists: every executor is run on the same calibration items and the ladder is replayed item by item (`dmc/scheduler.py::Outcome`), so the accuracy of a later rung is its accuracy on the items earlier rungs passed up — which are harder — and the tokens are those actually spent on them; the independent product of marginal accuracies over-predicts (round-4 review, M6) and is used only where no joint data exists, with the estimate type recorded per assignment. Calibration may be keyed by a per-node **difficulty bucket** set at compile time by a decision head (`op#difficulty`). Per-call token budgets come from the cost model, never from the model's self-estimate (models predict their own token use poorly, r ≤ 0.39; W2-01/02). Tool outputs feeding a node are **capped** at a compiled `input_cap` that counts against the worst case; the runtime truncates deterministically (head + tail around a marker, `dmc/scheduler.py::cap_input`) and counts cap hits in `Receipt.cap_hits`. List scheduling by critical path across lanes; **prefix groups are keyed by (lane, group, executor)**, because a cache never transfers across models or family checkpoints; a **worst-case token bound** (compile error if over budget); a content hash over assignments, budget, lanes, executor versions and contracts, and the **lock hash** (R28).
- **Guards, variants and recompiles** (`dmc/scheduler.py::Guard`, `dmc/scheduler.py::Program`, `dmc/scheduler.py::GuardRuntime`; tested). Agent graphs have data-dependent branches, which is what defeats naive tracing compilers (W2-01). As in a JIT with deoptimization (and Dyserve's pre-built variants; W2-01), a program is a main graph plus **pre-compiled variants**, **guards** and a **recompile allowance** with a per-recompile token reserve. A guard names a node and a typed predicate over that node's IR-level output (e.g. `tests_pass`, `nonempty_result`), evaluated over data, never by free-text judgement (CaMeL discipline); on failure it selects a variant or spends one recompile. Semantics: (1) every variant is compiled with its own bound when the program is compiled; (2) the program's worst case = main + most expensive variant + allowance × reserve, checked against the budget at compile time; (3) each variant runs at most once and recompiles stop at the allowance; anything beyond yields an `Abstain` receipt (D⊥), never a loop; (4) the program hash covers main, variants, guards, allowance and ceiling. A recompile re-enters the compiler with the guard's typed failure as new context and the remaining budget as its budget.
- **Capabilities, fixed per branch at compile time.** Every node declares the capabilities it needs from `dmc/scheduler.py::CAPABILITIES` (read, write, external_send, destructive); the program's ceiling comes from the Goal IR's autonomy level; any node in the main graph or in a variant that exceeds the ceiling is a compile error. A guard failure therefore cannot steer execution into a branch with more authority than the goal was compiled with — the defence against branch-steering attacks on static plans (COBRA, Oct 2026). Irreversible effects (external_send, destructive) additionally need a dry-run node and, at autonomy levels that require it, a confirmation node before the effect node; secrets are referenced by handle in the IR and never rendered into a prompt.
- **Memory and state.** Memory is explicit and typed: `memory.read` and `memory.write` nodes over named stores (files, repositories, tickets, scratchpads). Reads are snapshotted by content hash (R23), so D0 replay returns the recorded value; writes are recorded side effects with idempotency key H(program hash, node id, value) (R22), so replay and kill/restart never re-apply them. File-system-as-memory (Manus; W2-02) maps to a content-addressed workspace whose snapshot hash enters `Context.session_state_hash`. Compaction never touches the pinned region (rules, Goal IR, capability ceiling) and records what it dropped.
- **Expert residency plan.** The goal determines which family checkpoints (Artifact A) or goal-gated experts (Artifact B) are needed; the schedule lists them so serving keeps them resident (and, on disaggregated hardware such as NVIDIA's Groq 3 LPX — GPUs for attention, LPUs for MoE experts, Mar 2026 — places them ahead of time; W2-01).

### I.2.2 Executor assembly (cheapest adequate executor per node)

| Node kind | System-1 (no/low tokens) | System-2 / spoke escalation | Evidence |
|---|---|---|---|
| Arithmetic, dates, lookups, rule checks | **Compiled code** from IR slots (exact; coverage = grammar coverage) | Hub reason profile | Exp. 07 (code covers its canonical template exactly, abstains elsewhere → escalation) |
| Fixed-label classification, intent, routing | Encoder classifier (ModernBERT/DeBERTa) or **decision head** on hub hidden states (one prefill, 0 generated tokens) | Hub Decide profile → 27B | RoBERTa 95.9 vs Haiku 84.1 on ATIS; probes recover tool-need and numeric comparison better than asking (W2-04) |
| Entity extraction | GLiNER2 (F1 0.590 vs GPT-4o 0.599, 2.6× faster on CPU) | Hub Call profile (constrained) | W2-04 |
| Tabular decisions | GBDT / TabPFN-3 / TabICLv2 | Hub with tool access to the model | LLM-on-rows 74.8 vs XGBoost 78.5 AP; TabPFN-3 ≫ tuned GBDT in TabArena (self-reported) (W2-04) |
| Forecasting | Chronos-2 / TimesFM-2.5 | — | GIFT-Eval: Chronos-2 79.8% win rate (W2-04) |
| Retrieval / ranking | Embedding + kNN, cross-encoder reranker | Hub | W2-04 |
| Tool calling | SLM specialist (Qwen3-4B: 86.7% BFCL Live, *secondary*) | Hub Call → 27B | W2-04 |
| Reasoning, coding, planning | Hub, thinking off or early exit | Hub thinking → verifier-selected best-of-K → 27B | W2-02, W2-05 |
| Judging / verification | Executable verifiers; scalar RM (Skywork-Reward-V2-Qwen3-8B) | Hub Judge (both orders) → 27B judge | Part II §5.3 |

**Executor registration contract** (what the scheduler may consume; identical for an LLM profile, an encoder, a GBDT or a code compiler):
1. *Input schema* compiled from IR slots (for classic ML: the feature schema, with each feature's IR source and unit).
2. *Output* = a typed IR value plus a calibrated probability with a conformal threshold (R19); no executor returns free text into the graph, so a GBDT's answer re-enters the IR exactly like an LLM's.
3. *Cost model* = the `dmc/scheduler.py::Executor` fields (coverage, accuracy on answered items, forced accuracy, tokens in/out as compiled bounds, `input_reads`, latency, version) **plus per-item `Outcome` records on the shared calibration split** for its ops, so it can be priced jointly with every other rung.
4. *Contract* (D0 or D1 on its certified stack) and lock.
5. *Calibration size:* at least as many items per (op, difficulty) bucket as the Wilson lower bound needs to clear the bars it is used for; fewer → the executor is registered but not schedulable for that bucket.
Cost models are per lock: any change to the hub or an executor invalidates its records and every schedule that used them (the schedule hash includes the lock hash).

### I.2.3 The hub: Qwen3.6-35B-A3B and the served artifact

**Why this base.** The user's team has a working, reviewed training stack for it (A3B report); it serves at ~3B active parameters with native MTP. The same-generation evidence is reassuring for *reasoning only* (dense-vs-MoE ~1–4 points); for agentic work the same-generation gap is large (§I.0.1), which is why the agentic spike runs in G0–G1 and the switch criteria below are fixed now. **Alternative:** Qwen3.8-Flash-Next (125B total, 6B active, plus 51B n-gram embedding parameters held in host RAM; beats Qwen3.8-27B on every benchmark both vendor cards report; license to be confirmed from the card before any use; W2-05). **Switch criteria (decided at the end of G1):** switch the hub if (a) the agentic spike's gain has an upper 95% bound below +4 points on its terminal eval (RL does not move this class on this base), or (b) the escalation share the scheduler needs for system parity on the agentic suites, measured on G1 calibration data, exceeds e = 0.30 (cost-of-pass ≤ 0.5× is then infeasible: 1/3 × 1.2 + 0.30 = 0.70). A switch re-measures r (6B active doubles the hub's per-token compute) and the host-RAM footprint before any training. Qwen3-30B-A3B remains the discovery-methodology and D1 comparison MoE (Part II §4.2).

**The served artifact (decided now; round-4 M2).** Item (i) merges adapters into weights and item (ii) gates experts by goal; both cannot hold in one checkpoint unless item (i) is re-expressed. The plan therefore fixes:

- **Hub-base** = Qwen3.6-35B-A3B + T1 (general capability transfer from the 27B), merged bf16 → block-FP8 with the A3B survival check. T1 is general, so merging it creates no conflict between families. All family capacity is trained on hub-base.
- **Artifact A — served from G1.** Per goal family f, a checkpoint C_f = hub-base + f's merged hot-expert LoRA (item i). The scheduler treats each C_f as its own executor (own lock, cost model, joint calibration records, atlas), pins nodes to C_f replicas, and pays full prefill on cross-family edges (prefix keys are per executor). Memory: ~72 GB bf16 / ~36 GB FP8 per checkpoint (A3B-ch.2); one 8×H200 (141 GB) node holds up to six family replicas, a hub-base replica for family-less nodes and the 27B spoke (~54 GB bf16). No engine work. In A, item (ii)'s experts for f are added to C_f once engine item E2 lands (G2), with the goal bias fixed on; "goal off" then means "the node is not routed to C_f". The bit-identity property of `dmc/expansion.py` is used in training (on hub-base, off-family loss is unchanged while f's experts train) and as a test that C_f with its new experts gated off equals its merged-LoRA parent.
- **Artifact B — the target, decided at the end of G2 by a pre-registered comparison.** One checkpoint: hub-base frozen, plus, for every family f, a small set of goal-gated experts (routed variant) warm-started as copies of f's hottest experts on hub-base (ESFT-tight: 2–8 per layer) and trained on f; item (i) is then *expressed as* item (ii) — discovered experts are trained as goal-gated copies, never in place. Goal-off bit-identity then holds in serving, there is one resident model and one lock, and cross-family prefix reuse is possible once G0-3 passes. Size: ≈ 126M parameters per added expert slot across 40 layers; at k = 48 slots per layer (six families × 8) that is ~6.0B parameters (~12 GB bf16, ~6 GB FP8), and 256 + 48 = 304 experts divides an EP degree of 8. **Decision rule:** adopt B if, on each of the first three families, B is non-inferior to A (δ_b on the family's suite), off-family outputs equal hub-base's bit for bit in serving, and engine items E1–E4 pass the conformance matrix; otherwise A stays and B remains a research arm.
- **Rejected — C:** multi-LoRA serving of expert adapters (fragile in vLLM/TRT-LLM/SGLang; W2-03).

**Engine work** (budgeted in G1 next to R27's external-route entry point; round-4 M9): **E1** a per-request goal-bias vector added to router logits before top-k (static per request, so it does not break batch invariance; DeepSeek's `e_score_correction_bias` is per model, not per request); **E2** `num_experts = 256 + k` with EP re-sharding and FP8 block scales for new experts (k chosen so 256 + k divides the EP degree); **E3** EP load balance under concentrated goal traffic (redundant replicas of goal experts, EPLB-style) — carried as a risk; **E4** R3 routing-replay masks over the extended router for T3. Artifact A needs none of these in G1, E2 (and a static per-checkpoint bias) from G2; Artifact B needs all four.

**Compute per variant.** *Routed* goal experts compete inside top-8: active compute is unchanged. *Additive* goal experts run for every token on top of the 8 routed + 1 shared experts (all with intermediate size 512): one additive expert costs ≈ +11% MoE-FFN FLOPs per token per layer. The additive variant is kept only if pilot 06 and T2 show a gain worth that cost.

**Goal-family capacity inside the hub (the five kinds):**

1. **Hot-expert adapters (item i).** Per goal family (reason-math, code, agentic-tool, research, decide, create, voice…): a routing atlas on that family's traffic *on the weights that will be adapted* (hub-base) → hot-expert set by selection counts (A3B rule: overall 80% mass ∪ per-slice 50% mass; ~97/256 experts per layer for voice) → rank-16 LoRA on those experts (`dmc/expert_lora.py::HotExpertLoRA`; B zero-initialized, so attaching is an identity). Two ablations pre-registered from the literature: (a) **experts-only** (no attention/shared-expert LoRA), because a Qwen3-30B-A3B PEFT study found experts-only best and attention-only worst for retention, and ESFT attributes most forgetting to shared parameters; (b) **ESFT-tight sets** (5–15% of experts) vs the A3B 80/50 rule (W2-03). Merged for serving (Artifact A) and exported bf16-merge → block-FP8 with the A3B survival check.
2. **Goal-gated new experts (item ii)** (`dmc/expansion.py`). One or more new experts per layer per goal family, warm-started as copies of that family's hottest base expert with router rows initialized from the parent row, plus symmetric noise (σ = 0.05·std) on the copy and the new row only — the parent is never perturbed, so goal-off identity holds (UMoE; W2-03); a **per-goal bias** on the frozen router logits, soft-anchored to the base router (Router-Prior-Bias style; re-imposing load-balancing loss hurt: 31.9 vs 45.8; W2-03); a router-only **review pass** on <1% replay (MoE-LPR: without it, original-language retention fell from 96.6% to ≈ 87%; W2-03). Two variants until evidence picks one: *routed* and *additive* (compute above). New *blocks* (LLaMA-Pro depth expansion) are a fallback only: they add per-token latency and lost to MoE-LPR at equal parameters (W2-03).
3. **Thinking control — measured before it is scheduled.** Qwen3.6 has no `/think` soft switch: `enable_thinking` is a chat-template argument that changes the rendered prefix, and whether per-turn toggling keeps the KV prefix byte-identical is open (W2-02); budget forcing is an emergent property of Qwen3's training, not a documented Qwen3.6 feature. G0-4 measures the accuracy-vs-budget curve under hard truncation + "answer now" and the template/KV effect of per-turn toggling. Until T5 lands, **compiler v1 uses only `enable_thinking` on/off per node plus training-free early exit** (DEER −19% to −80% tokens at +0.3 to +5 points; hidden-state probe −24%; W2-02) and reports realized tokens. Trained difficulty-aware length policies (When2Think +10.0 points Pass@3 AIME24 at −27.9% tokens; TRAAC +8.4 at −36.8%; W2-02) are T5 candidates. The fast path is conservative on math and code: turning thinking off costs Qwen3-30B-A3B ~49 points on AIME 2025 (70.9 → 21.6; W2-02).
4. **Decision heads.** Small probes on intermediate hub states for node-level decisions (tool needed? escalate? which family? difficulty bucket?) — one prefill, no generated tokens, calibrated with conformal thresholds; System-1 executors that live inside the hub, re-trained whenever the hub changes (T6).
5. **MTP speculative decoding** (native head; lossless under greedy up to numerics). Its speedup on a Gated-DeltaNet hybrid is **unmeasured** — rejected drafts need linear-attention state rollback (W2-02) — so the 1.8–2.0× reported for attention models (DeepSeek-V3 MTP, EAGLE-3 on Qwen3; MoE verification erodes gains by activating the union of experts; W2-02) is not assumed; G0-5 measures acceptance and speedup. Off in the D1 lane (R6). Lossy acceptance is excluded (1.54× longer completions on reasoning; W2-02).

**Context management** for long agent runs: compaction trained with RL (CompactionRL on a 30B-A3B: +5.5 SWE-bench Verified, +6.7 Terminal-Bench 2.0; context editing −84% tokens on a 100-turn eval; W2-02/05), with **safety rules pinned outside the compacted region** (compaction raised violations from 0% to 30–59%; W2-02).

### I.2.4 Ensemble policies (item iii)

- **Cascade** (default): System-1 → hub → 27B, ladders chosen by the scheduler on joint calibration data.
- **Verifier-selected best-of-K** for hard reasoning/coding nodes: K samples from the hub (Self-MoA: one strong model sampled K times beats mixing models by 6.6%), selection by executable verifier or judge (selection beats fusion; fusion was never preferred in 0/42 tasks; the verifier is the bottleneck; W2-04).
- **Confidence-filtered parallel thinking** (DeepConf: 99.9% AIME 2025 at up to −84.7% tokens vs full parallel thinking; W2-02) where verifiers are unavailable.
- **Spoke escalation** to Qwen3.8-27B only for nodes whose calibrated hub accuracy is below the node's bar, within the escalation budget e (§I.1.1); switching models mid-session loses the prompt cache (W2-02), so escalation is per node, with its own prefix key.
- No runtime answer fusion by default; weight merging (soups/TIES) only among same-base adapters, evaluated like any executor.

---

## I.3 Token economics: Groq principles → mechanisms → reported effects

The right-hand column lists effects **reported in each source's own setting**. They are upper bounds from different settings, they decompose rather than multiply (W2-01), and none of them is credited in the feasibility model (§I.1.1); the ±15% predicted-vs-realized audit is the claim.

| Groq principle (W2-01) | GCAM mechanism | Reported effect in its own setting (upper bound) |
|---|---|---|
| Compiler knows every operation's place and time | Static schedule: executor, budget, cap, lane per node; worst-case bound | Predictable cost; overruns impossible by construction given caps and the recompile allowance |
| No runtime arbitration | Batch-by-profile scheduling, pinned family replicas, static escalation budgets | Fewer model/adapter swaps; bounded cascades (forced-deferral attacks bounded) |
| Data movement planned ahead (stream registers) | Prefix groups as a **placement preference**; workflow-planned KV reuse **only after gate G0-3** (prefix caching is disabled on this hybrid in production today: vLLM restored corrupted linear-attention state after ~8.6k cached tokens, and disables prefix-cache reads when prompt log-probs are requested; A3B-ch.2/4) | 1.8–3.6× (KVFlow, Halo); cached input ~10× cheaper (Manus) — **not credited until G0-3 passes** |
| Whole-program dataflow | DAG planning with parallel independent nodes | 3.7× latency, 6.7× cost (LLMCompiler); 5× tokens (ReWOO) |
| Deterministic, replayable execution | Plan cache + result store (Part II D0) | −50% cost (Agentic Plan Caching) |
| Specialized functional units | System-1 tier + compiled code | 36–97% of slow-path tokens avoided (System-1 gates in a multi-agent coordination setting, W2-02); ~1000× cheaper classification (W2-04) |
| Pipeline across chips | Lanes across executors; MTP inside the hub | MTP speedup on GDN unmeasured (G0-5) |

Agent token cost is dominated by **input** (Manus reports 100:1 input:output; SWE-bench agent runs use ~1000× chat-coding tokens and vary up to 30× between runs of the same task; W2-02), so context policy (compaction, compiled tool-output caps, prefix placement) is a first-class part of every schedule.

---

## I.4 Training program (builds on the A3B recipe)

| Stage | What | Recipe and guardrails | Indicative compute |
|---|---|---|---|
| **T0 Discovery** | Routing atlas per goal family on Qwen3.6-35B-A3B; hot-expert sets; fast/slow expert analysis (TempoMoE found "fast" and "slow" experts in VLM MoEs; W2-02); super-expert profiling; Part II ND1. | Selection counts, not router affinity (A3B: affinity entropy 0.98, selection concentrated). | < 50 GPU-h |
| **T1 Capability transfer** | Close the 3.6 → 3.8 post-training gap: SFT with top-k KD (A3B: top-20, w = 0.3) on 27B traces, then **on-policy distillation** (student samples, teacher scores per token, reverse KL; full logits — shared vocabulary; top-k KD is biased, W2-05). Reasoning + agentic + goal-family data. Output: **hub-base**. | Hot-expert LoRA (experts-only ablation), cached-prefix + 512-token grad tail, lr 1e-4, merge bf16 → FP8 with survival check, paired val + holdout, pre-registration. On-policy distillation: 74.4 AIME'24 in 1.8K GPU-h vs 67.6 for RL in 17.9K on Qwen3-8B (W2-05). | 1–3K H100-h (W2-05 extrapolation) |
| **Agentic spike** (G0–G1) | ~100 RL steps of terminal RL on the base with R3, AgentWorld-style (terminal-only RL for 100 steps moved a Qwen3.5-35B-A3B SFT model Terminal 32.8 → 47.0, SWE 52.0 → 63.5; W2-05), on the Agent Lightning + verl stack that ran R3 on Qwen3.5-35B-A3B with 4×B200, EP = 4 (W2-03). | Pre-registered gate: terminal-eval gain ≥ +4 points with paired CI lower bound > 0, and no reasoning-suite loss beyond δ_b. Feeds the switch criteria (§I.2.3). | One node, ~1 week |
| **T2 Goal experts** | Goal-gated new experts per family on hub-base; router rows + goal biases; review pass. | Pre-registered per family: gain on that family vs a matched-parameter LoRA arm, identity off-goal, traffic share, drift on other goals with the goal on, wrong-goal accuracy (pilot 06 design). | 10–100 GPU-h per family |
| **T3 Agentic RL** | GSPO/CISPO-style sequence-level objectives + **R3 routing replay** + exact token handoff; bitwise train/infer path if available for this architecture (vLLM reports one for Qwen3.5-35B-A3B's family; W2-05); environments: SWE (SWE-smith/SWE-Universe style), terminal, tool simulators, browsing; rule-based reward shaping. | Stale-sample dropping; KL monitoring. Scale set by the allocation tier below. | Tier 1: up to ~25K H100-h; Tier 0: ≤ 10K as a sequence of AgentWorld-scale runs |
| **T4 Decision-token DPO** | Agent decisions (call tool / which tool / stop / escalate / ask). | A3B round-2/3 recipe: two independent judges must agree; preference margin only on decision tokens (weight 4, β 0.5) + anchors; agreement between models is never a label. | < 100 GPU-h per round |
| **T5 Token efficiency** | Difficulty-aware length RL; learned think/no-think per node; CompactionRL; early-exit probes. | Guard: hard-set non-inferiority (uniform length pressure costs 0–2 points on hard sets; think/no-think RL can collapse; efficiency training reduces CoT faithfulness — monitored; W2-02). | 1–2K H100-h |
| **T6 System-1 distillation and calibration** | Decision heads, encoders, SLMs, GBDTs trained on hub/27B labels and logged traces; conformal calibration; joint `Outcome` records for every executor → cost models. | Each executor certified with coverage/accuracy/forced-accuracy/contract before the scheduler may use it. | CPU–small GPU |

**Stage dependencies (explicit edges; round-4 M11).** Training changes what the router sees even with the router frozen — the A3B team measured hot experts *after* applying the attention adapter for this reason (A3B-ch.2). Therefore: T0 → T1 → **re-atlas on hub-base** (ND1) → T2, with warm-start parents and hot sets chosen on hub-base, the weights that are expanded; T3 → **re-atlas** and re-check of T2's traffic shares and identity tests, since RL moves experts; T4 and T5 after T3; **T6 runs last on every hub release and re-runs after any hub change**, because decision heads read hub states and every cost model and joint calibration record is per lock (the schedule hash includes the lock hash, so stale schedules cannot be served).

**Training allocation (round-4 M5).** The plan states what it needs rather than assuming it. One H200-hour is counted as one H100-hour (conservative for training). One 8-GPU node at 70% utilization gives ≈ 940 GPU-h per week.

| Tier | Training nodes (8×H200) | T1 | Spike | T2 | T3 | T5 | Timeline |
|---|---|---|---|---|---|---|---|
| **Tier 1** (requested) | 2, weeks 2–24, plus the evaluation node | 2 nodes × 1–2 weeks | 1 node × 1 week | < 0.5 node-week | 2 nodes × weeks 10–24 (≈ 26K GPU-h ceiling) | 2 nodes × 1 week | §I.6 as written (30 weeks) |
| **Tier 0** | 1, weeks 2–30 | 1 node × 2–3 weeks | 1 node × 1 week | < 0.5 node-week | AgentWorld-scale runs (100–300 steps per domain), ≤ 10K GPU-h | 1 node × 2 weeks | G3 ends week 30, G4 week 36; G3's raw-hub expectation lowered, system gate unchanged |

Rollouts for RL share the training nodes (colocated, verl-style). **Environment construction is a line item:** 3 engineers × 12 weeks (weeks 2–14) for SWE (SWE-smith-style tasks from ≥ 200 repositories), terminal tasks, MCP/tool simulators and a browsing sandbox with snapshots; 1 engineer × 6 weeks (weeks 0–6) for goal-suite v0 and its verifiers; 1–2 engineers × 1 quarter for engine items E1–E4 and R27.

Methodology carried over verbatim from the A3B report: pre-registration with dated amendments; paired comparisons with conversation-level bootstrap; null controls (model vs its own replay); checkpoint selection by **behaviour probes** (greedy-action accuracy), not likelihood ranking; evaluation through the product; every check must be able to fail (A3B lessons 1–7).

---

## I.5 Evaluation

Five-rung ladder, generalized from the A3B report (A3B-§5): (1) held-out loss as served; (2) gold answer/action accuracy on curated suites; (3) replay of real goal traffic at temperature 0, with a self-disagreement baseline (A3B measured ~2% tool-decision self-disagreement under greedy GPU decoding); (4) blind two-judge comparison on divergent outputs with shuffled labels and a null control; (5) through the product: the compiled system as users receive it. Public suites per I.1 with per-benchmark δ_b; token accounting per task (input/cached/output), cost-of-pass at the measured cost basis, escalation share e, latency; predicted-vs-realized schedule audits with cap-hit rates; Part II parity metrics for nodes with determinism requirements.

**Goal-suite v0 (G0 deliverable; round-4 M12).** ≥ 100 goals across the goal families (reason-math, code, agentic-tool, research, decide, create), each with an executable verifier (tests, checks, state assertions in a sandbox) or a two-judge rubric with a null control; the product surface is the goal compiler's API as users call it (goal in; deliverable, receipts and cost out). Labels come from two independent judges with human adjudication of disagreements; agreement is never a label by itself (A3B lesson 4). 20% of goals are held out until G4 and topped up with goals recorded after the training-data cutoff, so the final comparison does not reuse the development set (the A3B canary caveat).

---

## I.6 Roadmap

| Phase | Weeks | Deliverables | Exit gate |
|---|---|---|---|
| **G0 Baselines, harness, de-risking** | 0–3 | Eval harness with token accounting for hub and 27B on all I.1 suites; null controls and **δ_b per benchmark** (G0-1); **measured cost basis r** and the filled-in feasibility model (G0-2); **GDN prefix-caching gate** (G0-3): state checkpointing at chunk boundaries validated on the pinned vLLM version, with the A3B 40-example cached / uncached / repeat probe extended to ≥ 8.6k-token prefixes; **thinking-control measurement** (G0-4); **MTP-on-GDN measurement** (G0-5); Part II GDN batch-invariance audit (G0-6); **goal-suite v0** (G0-7); routing atlases per goal family (T0); CPU prototypes (scheduler with joint calibration, guards and capabilities ✔; expansion ✔; expert LoRA ✔; pilots 03 running, 06/07 queued). | Baseline table with CIs and δ_b; feasibility model amended with measured r; G0-3…G0-6 outcomes recorded (a failed G0-3 keeps KV reuse uncredited); executor registration contract frozen. |
| **G1 Capability transfer + compiler v1** | 2–8 | T1 → hub-base; **agentic spike**; Artifact A for the first families; System-1 tier v1 (decision heads, encoders, code compilers, SLM caller); goal compiler v1 (Goal IR, planner profile, scheduler with joint calibration, plan cache, guards/variants/recompiles, capability ceilings, tool-output caps; thinking on/off + early exit only); engine items E1–E4 started next to R27. | Hub reasoning closes ≥ half of the original gap on ≥ 3 reasoning suites; spike gate decided; **switch criteria evaluated** (escalation share for agentic parity on G1 calibration data reported); predicted-vs-realized within ±15% on goal-suite v0 means. |
| **G2 Goal capacity + decisions** | 6–14 | Re-atlas on hub-base; T2 goal-gated experts (first 3 families); T4 decision-token DPO; ensemble policies; T5 token-efficiency v1; **A-vs-B served-artifact comparison** (pre-registered, §I.2.3). | Per-family gains vs matched LoRA with off-goal identity; A/B decision recorded; raw-hub tokens per solved task ≤ 1.0× on reasoning suites at parity. |
| **G3 Agentic RL** | 10–24 (Tier 0: 10–30) | T3 with R3 at the allocation tier; environments; CompactionRL; planner training; re-atlas after T3. | Agentic suites within δ_b of the 27B for the compiled system with e ≤ 0.15 (raw hub reported). |
| **G4 Integration & certification** | 20–30 (Tier 0: 26–36) | T6 on the final hub; product-level evaluation; Part II certification of D1 lanes; canary on real goal traffic recorded after the training-data cutoff. | All I.1 requirements met, or the gap and its cost reported against the pre-registered gates. |

---

## I.7 Risks (Part I)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Agentic gap does not close at 3B active (AA index 18 vs 34; same-generation agentic gap large) | Medium–High | Agentic spike in G0–G1, not G3; escalation within the budget e; switch criteria to Qwen3.8-Flash-Next fixed now (§I.2.3), with its memory (125B + 51B host-RAM n-gram tables), cost basis and license checked before a switch. |
| Cost target infeasible at the measured cost basis | Medium | The feasibility model is a formula; G0-2 plugs in r and the target is re-derived by dated amendment before G1, never after seeing GCAM results. |
| Prefix caching stays broken on the GDN hybrid | Medium–High | No KV-reuse credit anywhere until G0-3 passes; prefix groups remain a placement preference. |
| Distillation copies teacher mistakes; models agreeing ≠ correct (A3B lesson 4) | Medium | Two-judge labels for decision training; verifiers for reasoning; on-policy distillation only on verifiable or judged data. |
| Goal-gated experts starve or drift | Medium | Parent-row init with symmetric noise + review pass (W2-03); traffic-share, drift and wrong-goal metrics per family; off-goal identity guaranteed by construction on hub-base. |
| Artifact B does not match Artifact A | Medium | A is served from G1 and remains the default; B is adopted only on the pre-registered rule. |
| Engine work (E1–E4, R27) slips | Medium–High | Artifact A needs none in G1; budgeted with headcount; E3 (EP hot-spotting) carried with redundant-expert replicas. |
| LoRA-on-experts serving fragility | High | Merge adapters (Artifact A) or express them as goal-gated experts (Artifact B); multi-LoRA serving rejected. |
| Thinking control or MTP behaves differently on Qwen3.6 than on Qwen3 | Medium | Measured in G0 (G0-4, G0-5) before the compiler relies on them; compiler v1 uses on/off + early exit only. |
| Atlases, warm starts and cost models go stale after training | High | Explicit stage edges (§I.4): re-atlas after T1 and T3, T6 last and per hub release, schedule hash includes the lock. |
| Static plans attacked via branch steering (COBRA) | Medium | Typed guards over data, capability ceilings fixed per branch at compile time, static escalation budgets. |
| Compaction drops safety rules | Medium | Pin rules outside compaction; safety evals after every compaction policy change. |
| Token-efficiency training lowers hard-set accuracy or CoT faithfulness | Medium | Non-inferiority guards on hard sets; faithfulness probes; per-node thinking decisions instead of global pressure. |
| Training allocation smaller than requested | Medium | Tier 0 plan with lowered raw-hub expectations and a later G3/G4; the system-level gates do not move. |
| Benchmark conflicts (AA index versions, harness differences) | High | Run every comparison ourselves, paired, same harness. |
| GDN determinism unknown | High | Contract-aware scheduling (D1 nodes avoid the hub until the audit passes); G0-6. |

---

# Part II — Trust layer: the determinism contract and its requirements

Part II is the v4 plan (sections 1–8), which converged over adversarial review rounds 1–3. It is the contract every Part I executor and schedule runs under. Changes in v5: the hub is Qwen3.6-35B-A3B with D1 gated on the Gated-DeltaNet audit (§II.4.2); the six profiles of §II.5 are the node types the goal compiler schedules; executors declare their contract and nodes their requirement (R28 below). Changes in v6: hub numbers updated for Qwen3.6-35B-A3B (40 layers, ~72 GB bf16); R26's merged-checkpoint fallback is the served Artifact A of §I.2.3; R28 covers guards, variants, caps and the lock hash.

## II.1 The determinism contract

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

## II.2 Architectural requirements for consistent output parity

"MUST" items gate release. Each has an acceptance test.

### II.2.1 Toolchain identity and reproducibility

| ID | Requirement | Acceptance test |
|---|---|---|
| R1 | MUST pin every output-affecting artifact in a **lockfile**: weights (HF repo + revision), tokenizer files, chat template, adapters (merged or unmerged), expert masks; engine + version; `tokenizers`/`transformers`/torch/CUDA/driver versions; kernel set and determinism flags/env; **CPU/GPU thread or SM partitioning where it changes reduction order** (Phase 0: BF16 at 4 vs 3 CPU threads changes logits by up to ~4 and flips ~4.5% of routes); **grammar engine + version + grammar hash**; `async_scheduling`, speculative/MTP, prefix-cache and CUDA-graph settings; **scheduler policy** (e.g. batch-by-adapter); SKU; TP/EP/PP/DP; IR schema version; decoding policy; dispatch constants (`max_tier`, `max_repair`, `max_escalations`). | Lock hash on every receipt; replays refuse lock mismatch (`dmc/store.py::ResultStore`). |
| R2 | MUST canonicalize every hashed artifact (RFC 8785-style JSON, NFC text). | Property tests (`tests/test_canonical.py`). |
| R3 | MUST store every response content-addressed by `H(lock, canonical IR incl. session_state_hash, profile version, seed)` and support replay; a recompute that differs under an identical key is a hard error. | Replay byte-identical; divergence raises (`tests/test_store.py`). |
| R4 | MUST version and shadow-test routing tables, thresholds, templates and grammars as code. | Release diff lists every changed decision on the golden set. |

### II.2.2 Numerical determinism of execution (D1)

| ID | Requirement | Acceptance test |
|---|---|---|
| R5 | MUST run deterministic-lane traffic on **batch- and position-invariant kernels** (matmul, RMSNorm, attention incl. split-KV with fixed split *size*, MoE grouped GEMM, LoRA shrink/expand, sampler). | `unique@1000 == 1` per golden prompt under randomized concurrent load. |
| R6 | MUST make execution invariant to chunked-prefill boundaries, prefill-vs-decode path and preemption/recompute; v1 lock **disables** prefix caching, speculative decoding/MTP and async scheduling in the deterministic lane until each passes the matrix. | Conformance matrix (§II.6): identical logits at every position per variation. |
| R7 | MUST compute router logits with an **FP32 GEMM** (not a model-dtype GEMM cast afterwards, which is what stock BF16 checkpoints do) and fixed reduction order, and use a **stable top-k tie-break** (lowest expert index). Precedents: vLLM's merged DeepSeek-V4-Flash consistent path (FP32 gate GEMM; position tie-break in the DSA indexer) and draft PR #54955 (lowest-index MoE tie-break). Implemented for HF models in `dmc/routing.py::harden_routers`. | Router-decision equality across load variations; exact-tie unit tests (`tests/test_routing.py`); value of R7 measured in Phase 0 (stock vs hardened flip rate). |
| R8 | MUST NOT use per-batch dynamic quantization scales or token-count-dependent kernel selection (incl. expert-mask-induced token-count changes) in the deterministic lane. | Batch-mate + adapter-mix + mask fuzzing on MoE reproducers. |
| R9 | MUST sample with a **counter-based RNG keyed by (request seed, position)**; any top-k/top-p truncation MUST use a stable tie order (prob desc, token id asc). Current code: Gumbel-max over the full distribution with `dmc/runtime.py::position_seed`; truncation is not implemented yet. | Same seed → same sample regardless of batch-mates. |
| R10 | MUST fix TP/EP/PP per certified lock (or use TP-invariant kernels, TBIK). | Topology in lock; cross-topology runs are D2 only. |
| R11 | SHOULD offer **selective determinism** (per-request SLA via verify-and-rollback, LLM-42 / MarginGate) so throughput traffic does not pay the tax. | Deterministic requests meet R5 while co-scheduled with fast-lane traffic. |
| R26 | MUST certify D1 with **≥3 co-resident adapters and expert masks on/off**; if multi-LoRA kernels fail, fall back to (1) **batch-by-adapter scheduling** (serving default; no extra memory), then (2) **merged per-profile (per-family) checkpoints** — the certification reference, because only merged weights run on the kernel set vLLM has validated, and the served Artifact A of §I.2.3: on the hub ~72 GB bf16 / ~36 GB FP8 each (A3B-ch.2), one per GPU on an 8×141 GB H200 node. Merged ≠ base+adapter at ULP level, so a merged checkpoint is re-certified under its own weights hash. | Mixed-adapter fuzzing passes; for fallback 1, **{same-adapter batch-mates 0..k} × {adapter on/off} → identical logits** (vLLM pins only the LoRA shrink reduction); scheduler policy recorded in the lock. |

### II.2.3 Semantic determinism (D3) and defined failure (D⊥)

| ID | Requirement | Acceptance test |
|---|---|---|
| R12 | MUST compile every request to a **closed, typed IR** (discriminated union, required nullable slots) whose `Context` carries time, locale, timezone and a **`session_state_hash`** (prior turns, prior IRs, snapshotted tool results); parsers keep only the ambient fields a request depends on (**context minimization**: `now` is null unless the request refers to time), so time-independent requests keep a stable hash; no re-dispatch mid-program. Implemented in `dmc/ir.py::Context`. | No free-text path to a backend; cache key includes session state (`tests/test_compiler.py`). |
| R13 | MUST give the IR an explicit **ambiguity** channel; `Clarify` when Parse's decision margin is below its calibrated threshold. | Ambiguity suites: clarification precision/recall targets. |
| R13a | MUST render prompts only from explicit IR context: no template may read the clock, environment or host. Mechanism is template-specific: granite-3.1 inserts `strftime_now` only when no system turn is given (an explicit system turn suppresses it); Llama-3.x emits a date in the system header regardless. So every template's date hook (`strftime_now`, `date_string`) is bound to `Context.now`, or to a fixed epoch when the context carries no time (`dmc/runtime.py::template_clock`), and the **rendered prompt bytes are hashed into the receipt** (`Receipt.rendered_prompt_hash`, written by the compiler from the profile result). Phase 0: shifting only the injected date by one day changed 1 of 25 greedy outputs. | "Date-shift" test: the rendered prompt depends only on `Context.now` (`tests/test_runtime.py`); rendered-prompt hash on every receipt (`tests/test_compiler.py`). |
| R14 | MUST measure **paraphrase invariance** of NL→IR on curated paraphrase/format suites and gate releases on it, **always paired with accuracy**: report PIR together with the **consistent-correct rate** (CCR, share of paraphrase groups answered correctly in every variant), because a constant answer scores PIR = 1 (Phase 0: greedy text had PIR 0.8 while answering "no" to 13 of 15 decisions, "yes" to 1, and one unparseable). SHOULD use **paraphrase ensembling** (k paraphrases must agree on the IR hash; disagreement → Clarify) offline and optionally online. | PIR and CCR ≥ targets at stated coverage. |
| R15 | SHOULD render the IR deterministically for user confirmation in high-stakes modes and run IR→NL→IR round-trip checks offline. | Round-trip agreement tracked per release. |

### II.2.4 Spec conformance (D4) and decision extraction

| ID | Requirement | Acceptance test |
|---|---|---|
| R16 | MUST generate structured output under **grammar-constrained decoding** compiled from the IR/tool schema with the engine pinned in the lock (llguidance or XGrammar-2; the other engine as offline cross-check); allow a free reasoning span before the constrained span where reasoning is needed (CRANE). | 100% schema validity on golden + fuzzed inputs. |
| R17 | MUST extract decision-bearing values by **scoring candidates in one forward pass**, averaged over a fixed set of candidate orderings, with the label's probability **marginalized over its surface forms** (e.g. "no"/"No"/"NO": scoring only one casing misreads a model that prefers another); closed enums use single-token labels or length-normalized scores; never sample text. | Decision = argmax of order-averaged, case-marginalized scores; margin logged (`dmc/profiles/decide.py::DecideProfile`). |
| R18 | MUST run profile verifiers and allow only **bounded deterministic repair** (`max_repair` in lock; repair prompt is a pure function of the validator error). | Repair terminates; repair outputs are D1-reproducible. |
| R19 | MUST calibrate confidence and set abstention thresholds by **split-conformal calibration** on a frozen set; thresholds versioned with a dead band. | ECE ≤ target; coverage/risk curve in the release report. |

### II.2.5 Verification, certification, operations

| ID | Requirement | Acceptance test |
|---|---|---|
| R20 | MUST maintain golden corpora, metamorphic relations and a **synthetic-oracle generator** ("Csmith for prompts"). | Nightly differential runs. |
| R21 | MUST report the parity metric suite every release (§II.6), including **paraphrase pick-flip rate with dead band** and **adversarial-suffix / keyword-injection flip rate** for Parse. | Release blocked on any gate failure. |
| R22 | MUST run multi-op programs on a durable executor with recorded activities and idempotent tools. | Kill/restart mid-program → identical result. |
| R23 | MUST snapshot external inputs (retrieval, web, tool results) by content hash. | Re-run with snapshot → identical output. |
| R24 | MUST compute the **fragile-routing share** (share of routed decisions with margin < ε, ε calibrated to measured cross-stack noise) for every deterministic-lane request (`dmc/routing.py::RouteTracer`), and use it in confidence/abstention **only if it adds value beyond the decision margin**. | Incremental test: at fixed coverage C₀, parity with (decision margin + fragile share) exceeds parity with decision margin alone, with a CI excluding zero, on a held-out cross-stack set (Phase-0 test: `experiments/03_route_pinning.py`, H4). Three outcomes, pre-registered: CI lower bound > 0 → MUST; CI upper bound < +0.5 points → diagnostic only; otherwise inconclusive → confirmatory test at hub scale (n ≥ 2,000). The fragile share's ε is the measured p95 of cross-stack router-logit noise, not a fixed constant. |
| R25 | MUST ship the **non-optional discovery deliverables** ND1–ND3 (§II.4.3) with every release, and block release on an unexplained atlas diff. | Atlas diff report attached to release. |
| R27 | MUST record a **routing trace** (per-token top-k per layer: 40 × 8 = 320 B/token on the Qwen3.6-35B-A3B hub at one byte per index — 256 experts fit in a byte, 256 + k goal experts need two — more if gate weights are stored) plus the token ids it was recorded on and the sampling-mask summary, and support **token-verified** `pin_routes` re-execution: a token mismatch or an exhausted trace is an error, never a silent fall-back to live routing (`dmc/routing.py::RoutePinner`, strict by default). Receipts carry `pinned: full | partial | none`. Production needs an external-route entry point in the fused-MoE kernel path (none exists in vLLM serving; budgeted in P1). | D2-replay parity measured across certified stacks; strict-mode tests (`tests/test_routing.py`). |
| R28 | MUST give every executor a declared determinism contract (D0 replay or D1 bitwise on its certified stack) and every node a required contract; the scheduler may only assign executors that meet it; a schedule — including its guards, pre-compiled variants, recompile allowance, tool-output caps and capability ceiling — is a pure function of (goal IR, lock, executor cost models) whose content hash includes the lock hash; exhausting variants or recompiles yields `Abstain`. | `tests/test_scheduler.py` (contract filtering, schedule hash invariance and lock sensitivity, program bounds, guard exhaustion, capability ceilings). |

---

## II.3 System architecture: the compiler pipeline

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

### II.3.1 IR design (dmc-IR)

Discriminated union on `op` (`decide.classify`, `decide.select`, `tool.call`, `tool.plan`, `evaluate.pairwise`, `evaluate.rubric`, `evaluate.verify`, `create.text`, `analyze.reason`, `analyze.data`, `research.report`, `meta.clarify`); all slots required and typed (nullable instead of optional); closed enums; explicit units; `Context {now, locale, timezone, session_state_hash}` with context minimization (R12); `ambiguities` (must be empty to execute); `seed` and `parent_seed` only on `create.*`; static budgets (`tool.plan.max_steps`) inside the IR so they are hashed. Flat schemas sized to hosted grammar limits (Anthropic: 20 strict tools, 24 optional / 16 union params) so the same IR can target external models. Versioned; migrations are deterministic functions. All twelve ops exist as types in `dmc/ir.py` (v0); only `decide.classify` has an executing profile in Phase 0. Dispatch table: `dmc/compiler.py::DISPATCH_TABLE`.

---

## II.4 The Deterministic MoE hub and its spokes

### II.4.1 Hub + certified spokes

- **Hub:** one MoE base (hub-base, §I.2.3) with per-family capacity (LoRA matches full fine-tuning on small/medium SFT and at rank 1 for RL, must cover MLP/MoE layers; "LoRA Without Regret", Sep 2025). Under Artifact A each family checkpoint has its own lock, atlas and certification; under Artifact B there is one lock and one atlas with per-goal slices.
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

### II.4.2 Hub selection and controls

| Role | Model | Status |
|---|---|---|
| **Hub (v5)** | **Qwen3.6-35B-A3B** (256 experts, 8 routed + 1 shared, 40 layers: 30 Gated-DeltaNet + 10 gated attention, MTP; Apache-2.0), text-only, HF revision pinned | User decision in v5 and the base of the user's proven A3B training stack (Part I §I.2.3). **D1 is gated on the Gated-DeltaNet batch-invariance audit** (go/no-go below); until it passes, the hub delivers D0 + statistical D2 and the scheduler routes D1-requiring nodes to D1-certified executors (`dmc/scheduler.py::compile_schedule`). Qwen-Scope SAEs exist for Qwen3.5-35B-A3B-Base (re-validate or re-fit). Production also disables prefix caching on this hybrid (vLLM bug restoring corrupted linear-attention state; A3B report); gate G0-3 (§I.6) must pass before any prefix-cache use, in either lane. |
| Discovery & D1 comparison MoE | Qwen3-30B-A3B-Instruct-2507 (128 experts, top-8, 48 layers) | The MoE with all five discovery assets and vLLM batch-invariance validation of its family; P0 must still validate the pinned Instruct-2507 checkpoint itself and re-profile super experts. |
| GDN audit (P0) | Qwen3.6-35B-A3B Gated-DeltaNet kernels | **Pass** if GDN kernels pass batch-mate, chunk-boundary and prefix warm/cold fuzzing with MTP off (or MTP verified bitwise) and vision off; then the hub can be certified D1. **Fail**: hub stays D0/D2; D1 nodes use other executors. |
| Dense control | Qwen3-32B (BI 22/22 on GB300), Qwen3-8B for cheap runs | Same generation, same attention as the hub. |
| Capability reference | Qwen3.8-27B (the model the user named; dense hybrid) | Capability comparison and Create spoke, not a tax control. |
| Scale-up | DeepSeek-V4-Flash (284B/13B, MIT; hash-routed first 3 layers; bitwise train/infer path in vLLM) | P6+. |
| CPU lab | granite-3.1-1b-a400m (32 experts, top-8), OLMoE-1B-7B | Probes, pilots, vertical slice. |

**MoE determinism tax** is reported as a tuple per pair: (unique@N in default mode, first-divergence index, fragile-token share, D2-fresh parity at C₀).

### II.4.3 Deep discovery program (every layer, circuit, router)

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
| S3 Dictionaries | Qwen-Scope residual SAEs (published for Qwen3.5-35B-A3B-Base, 40 layers, and for Qwen3-30B-A3B, 48 layers): **re-validate or re-fit for Qwen3.6 and hub-base, and after each family checkpoint or goal-expert release**. No CLTs in v1 (no 30B-MoE budget exists); circuits use EAP-IG over coarse nodes {heads, experts, shared expert, router logits}. | Feature catalog per profile. | Fidelity thresholds; features validated by intervention. |
| S4 Behaviour circuits | Contrastive sets per profile (yes↔no, tool A↔B, prefer A↔B); EAP-IG localization; expert-level attribution. | Circuit cards. | Patching recovers ≥ X% of logit diff; stable across seeds. |
| S5 Interventions (pre-registered) | Margin regularization; StableMoE router freeze/distill; expert masks/router bias; decision-feature steering. Controls: size-matched random-expert nulls, prompting baseline, off-target ≤ ⅓ on-target, capability budget. | Intervention cards. | D2-fresh / D3 gains within capability budget, or a negative result that removes the lever. |

### II.4.4 Router hardening toolbox

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
- **Batch-invariant train/infer numerics** — stretch only (see §II.4.5).

### II.4.5 Determinism-hardening training (per profile)

| Method | Applies to | Effect (literature) | Guardrail |
|---|---|---|---|
| Invariance/consistency training (BCT, ACT, Flip-Flop Consistency) on paraphrase + format augmentation | Parse, Decide, Call, Judge | +11.6% cross-variant agreement (F²C); −86% biased reasoning (BCT) | Held-out paraphrase families |
| Mode-seeking on-policy distillation; majority-vote targets (ScPO/TTRL) | Decide, Call, Judge, Reason final answers; dense students | Collapses onto one answer | Entropy–capability coupling: monitor pass@k; never on Create |
| Calibration + abstention (RLCR Brier reward; ~1k graded examples) | Parse, Decide, Judge, Reason | Explicit abstention instead of guessing | ECE, abstention precision |
| Router margin regularization / StableMoE | Hub adapters (not Create) | Wider margins (hypothesis) | P0 pilot gate, capability budget |
| RL baseline: R3 + FP32 router on rollout and trainer + FP16 or TIS/MIS correction with sampling mask exported | Any RL stage | Stable MoE RL without bitwise kernels | KL monitoring |
| Bitwise RL (batch-invariant backward passes for grouped GEMM/top-k/combine) | Stretch, or at DeepSeek-V4-Flash scale-up where it ships | KL = 0 | Separate engineering estimate |

---

## II.5 The six profiles

Each profile = `(adapter, expert mask?, template, grammar, decoding policy, verifier, thresholds)`, locked and versioned. All decision-bearing values are scored (R17); all thresholds are conformal (R19).

### II.5.0 Parse (front end) — strictest D3 targets
Dense parser spoke; closed intent scoring with dead band; constrained slot filling; paraphrase ensembling; clarification; invariance + calibration training; certified on paraphrase and adversarial-flip suites before any other profile.

### II.5.1 Decide
- Optional greedy reasoning span → **order-averaged label scoring** → conformal abstention band. Thinking off by default (reasoning-tuned models abstain ~24% less, AbstentionBench).
- Contenders: hub adapter; **dense student**; linear-probe head on frozen hub states; gpt-oss-safeguard-20b; Bradley-Terry scalar head.
- Training: classification SFT + calibration (~1k graded examples) + abstention + paraphrase invariance + label-permutation augmentation.
- Targets: D1 TARa = 1; D2-fresh parity ≥ 99.5% at coverage ≥ C₀ = 0.85; PIR ≥ 97% with CCR reported, at C₀; ECE ≤ 0.03.

### II.5.2 Call (function calling only)
- (1) **Compile from IR first** (all arguments present as typed slots ⇒ deterministic code, no model call); (2) tool selection = scoring over {tools, `no_tool`, `clarify`}; (3) arguments under strict JSON-Schema grammar, greedy; (4) RFC 8785 canonical call; (5) validation; (6) idempotency key = call hash; recorded activity.
- Data: APIGen-MT / ToolACE / xLAM-style trajectories + IR-derived synthetic calls. Benchmarks: τ²-bench pass^k (every model loses 10–30 points pass^1→pass^4), BFCL v4 with care (Epoch AI: 48% of sampled tasks defective).
- Fixed interface is the main lever: BFCL format-sensitivity deltas reach 81.5 points for small specialists vs 8.0 for Qwen3-235B. Call stays narrow; planning goes to `tool.plan` + durable executor. Dense-student arm in the bake-off.

### II.5.3 Judge (contrastive cross-evaluation)
- Executable verifiers first; scalar RM (Skywork-Reward-V2-Qwen3-8B) for ranking; generative verifier profile on the hub (maj@k over fixed seeds) for out-of-domain cases.
- **Both-order** pairwise scoring is a MUST (position bias grows with the quality gap); verdict = mean log-odds across orders; tie band.
- **Style normalization is conditional on the criterion type** (content criteria: render to canonical plain form, since style/markdown bias dominates 2026 audits at 0.10–0.76 and SOTA RMs fall to 46.6% under style interference; formatting/instruction-following criteria: no normalization). Mode recorded in the receipt.
- Rubric scores = expected value of the score-token distribution. Cross-family panels with agreement statistics. Dense-student arm.

### II.5.4 Create
- Output is a pure function of `(IR, seed)`; default seed = H(IR); **`parent_seed` inheritance** so a revision keeps the base draft's seed schedule (documented: any IR edit without `parent_seed` yields an unrelated draft).
- Plan → draft → critique → revise; N candidates per stage with seeds `H(IR, stage, i)` on counter-based RNG; ranking by a locked RM + Judge with stable tie-break. Diversity from an explicit enumerated latent (Verbalized Sampling recovers 1.6–2.1×), recorded in the IR.
- Spoke: Qwen3.8-27B (best small open writer, Apache-2.0), **D0 only** (record/replay) until its Gated-DeltaNet kernels pass a batch-invariance audit; the P3 bake-off includes a plain-attention writer (hub or Qwen3-32B) that can run on the D1 lane. Kimi K3 / GLM-5.3 are stronger but custom-licensed. No mode-seeking on this profile.

### II.5.5 Reason
- Thinking-mode generation on the deterministic lane (BF16 diverges ~token 65 vs ~1,826 in FP32 across stacks per LayerCast; long chains need D1 kernels, and FP32 accumulation is an option for D2). Answer-level determinism via K fixed-seed chains + majority vote with deterministic tie-break; final answer by grammar. Pinned data sandbox; snapshotted retrieval (R23).
- Verifiers: code execution, numeric checks, PRM/Judge on steps, citation-to-snapshot checks.
- Hub reasoning adapter; Tongyi DeepResearch-30B-A3B spoke for `research.report` (own super-expert/fragility profile; no assumed atlas transfer); RLVR per §II.4.5 baseline.

---

## II.6 Verification and certification

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

## II.7 Trust-track roadmap (runs alongside Part I §I.6)

| Phase | Weeks | Scope | Exit gate |
|---|---|---|---|
| **P0 Foundations & de-risking** | 0–6 | *Done on CPU (this repo):* lockfile, hashing, metrics, IR v0 (12 op types), store with replay, receipts, R7 router hardening, token-verified route tracing/pinning, synthetic-oracle generator, `decide.classify` vertical slice; router atlas on granite (§III). *Running / queued on CPU:* route-pinning pilot (03, running), goal-gated experts (06), System-1 cascade (07), margin regularization (04), all pre-registered; conformance (02) done. *Pending GPU:* Qwen3-30B-A3B-Instruct-2507 vs Qwen3-32B/8B, default vs batch-invariant vLLM; the hub's Gated-DeltaNet audit (§II.4.2, G0-6) and prefix-cache gate (G0-3). | Baseline report; hub confirmed; tax tuple measured; pilot outcomes recorded against their pre-registrations. |
| **P1 Deterministic runtime** | 3–10 | Deterministic lane: vLLM BI + FP32 router GEMM + stable tie-break + static scales + pinned MoE kernel configs; full conformance matrix (covering array) incl. **adapter mix, same-adapter batch-mates and masks**; counter-based RNG; routing traces; **external-route entry point in vLLM's fused-MoE path** for R27 and pinned prefix routes (engineering estimate in P1 planning; until it lands, D2-replay is certified on the HF reference stack); selective-determinism evaluation (LLM-42 / MarginGate). | D1 on hub + ≥3 co-resident adapters on one SKU, or fallback in lock. |
| **P2 Compiler front end** | 6–14 | IR v1; dense Parse spoke; constrained decoding; clarification; paraphrase ensembling; caches; round-trip checks; adversarial-flip suite. | Parse PIR, flip and clarification targets met. |
| **P3 Profiles v1 + bake-offs** | 10–20 | Decide, Call, Judge, Create, Reason; spokes locked; **dense-student arms**; durable executor. | Each profile meets D1/D4 gates and benchmark floors; bake-off decisions recorded. |
| **P4 Discovery program** | 6–24 | S1–S5 on the shipping hub; ND1–ND3 operational; dictionaries re-validated per adapter. | ND1–ND3 in the release pipeline (mandatory); S5 effect sizes or negative results recorded. |
| **P5 Hardening training** | 14–32 | Invariance, mode-seeking (non-Create), calibration/abstention; margin regularization only if the P0 pilot survived; stack-independent router (pre-registered); RL baseline R3 + FP32 router + TIS. | D2-fresh/D3 improve at fixed C₀ without capability regression beyond budget. |
| **P6 Certification & ops** | 24–36 | Release process; differential testing at scale; SKU certification matrix; **formal batch-invariance analysis of MoE kernels** (grouped GEMM, top-k, permute/combine): extend Vosti's Triton analyzer if its code is released, otherwise budget a re-implementation (the notes do not confirm a release); drift monitoring. | First certified release with D0–D4 report and atlas diff. |

**Indicative compute:** P0–P2: one 8×H100/H200 node. P4: +1 node; residual-SAE work streams activations rather than caching (~20 TB per 100M tokens if cached naively). P5: 2–4 nodes for LoRA/RL on the hub. Fallback memory: merged checkpoints cost ~61 GB BF16 each on Qwen3-30B-A3B and ~72 GB bf16 / ~36 GB FP8 on the Qwen3.6-35B-A3B hub (one per GPU on 8×141 GB). Engine work: external-route fused-MoE entry point (P1, estimate pending); formal MoE-kernel analysis ~1 engineer-quarter if Vosti's analyzer is available, more if re-implemented.

---

## II.8 Trust-track risks

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
| Base-model obsolescence (Qwen3.6 was overtaken by Qwen3.8 within months; Qwen3-30B-A3B, the discovery comparator, is a 2025 model) | Medium | Methods are hub-agnostic; switch criteria to Qwen3.8-Flash-Next fixed in §I.2.3; DeepSeek-V4-Flash scale-up. |

---

---

# Part III — Phase-0 evidence (CPU, granite-3.1-1b-a400m-instruct)

| Experiment | Status | Key numbers |
|---|---|---|
| 00 quick probe (`experiments/00_granite_router_probe_quick.md`) | done | Single prompt; stock BF16 router (R7-violating). Superseded by 01. |
| 01 router atlas (`experiments/results/01_router_atlas.json`), 75 prompts, stock routers | done | Margin < 0.01: 7.8% of routing decisions (< 0.001: 1.0%); FP32-vs-BF16 flips 9.0% of decisions and touches 79% of tokens, by layer 2–4% early → 16–17% late; margin→flip AUC 0.87; batch of 4 vs alone: 0 routing flips, router-logit drift ≤ 3.4e-5, but 24% of greedy 24-token continuations differ; same-shape repeat: unique@3 = 1; task NMI 0.013–0.040 vs shuffled null p95 ≤ 0.0095; Decide PIR: label scoring 0.40 (accuracy 11/15) vs greedy text 0.80 (accuracy 10/15, "no" on 14/15); flipped label-scored items all had margins 0.23–0.54, so an abstain band of ~0.6 gives PIR 1.0 at 67% coverage, with 8/10 decided items correct and CCR 4/5 groups; one-day date shift changed 1/25 greedy outputs. |
| Proxy calibration, informal (treatment-blind, 10 prompts; no artifact) | superseded by 05 | R7-hardened routers, BF16, 4 vs 1/2/3 threads: max abs Δlogit 3.8–4.8, routing flips 3.5–4.6%. Motivated the thread-count proxy; replaced by `experiments/05_proxy_calibration.py`. |
| 05 proxy calibration (`experiments/results/05_proxy_calibration.json`), 75 prompts, hardened routers | done | REF = BF16/4 threads vs ALT = BF16/3 threads: router-logit |Δ| median 0.0048, p95 0.033 (→ ε_H4 = 0.033); per-position max output-logit |Δ| median 0.63, p95 2.25; per-position KL mean 0.0205 (median 0.0028, p95 0.096); routing flips 4.7%. REF vs STRESS (FP32): KL mean 0.0205, flips 5.8%. **Regime: stress** (mean KL 13.7× the published cross-engine 1.5e-3), so emulated-BF16 thread variation on this CPU is about as disruptive as a precision change; CPU pilot results can remove levers or validate mechanisms but cannot set production C₀ (pre-registered). BF16 forward ≈ 1.0–1.1 s vs FP32 0.62 s (no native BF16 on this CPU). |
| 02 CPU conformance (`experiments/results/02_cpu_conformance.json`), 25 generations + 30 decisions | done | Started before R7 hardening became the backend default: FP32 rows are R7-compliant (an FP32 model's router GEMM is FP32), BF16 rows use the stock BF16 router GEMM (R7-violating); decisions used the case-only scorer. Same-shape repeats: greedy 32-token text identical in 25/25 prompts × 3 runs (D1 on a fixed stack). FP32 at 4/2/1 threads: text 100% and decisions 100% identical (FP32 CPU kernels here are thread-count invariant). FP32 vs BF16: greedy text identical in only 68% of prompts, but decisions agree on 96.7% (29/30); the single flip had FP32 margin 0.026 (median 1.02), and an abstain band at margin 1.0 gives 100% parity at 47% coverage. Decision extraction by scoring (R17) is far more robust to a numerics change than generated text. |
| 03 route pinning, D2-fresh vs D2-replay (pre-registered + amendments 1, 1b) | running | — |
| 04 margin regularization (pre-registered, amendments 1–2) | queued after 06/07 | — |
| 06 goal-gated experts on granite (pre-registered + amendment 1) | queued after 03 | Tests item (ii): routed vs additive new experts against matched-rank hot-expert LoRA, router-only and random-parent arms; off-goal bit-identity, drift, traffic share, wrong-goal accuracy. A mechanism test on in-template data: it cannot speak to the production claim. |
| 07 code / System-1 probe / System-2 cascade compiled by the scheduler (pre-registered + amendment 1) | queued after 06 | Tests items (iv)/(v) and predictability: joint-calibration predicted vs realized accuracy and tokens per decision, with compiled input caps; independent estimate reported beside it. |

Caveat: the 01 FP32-vs-BF16 numbers were measured with granite's stock router, which runs the router GEMM in BF16 (R7-violating); they overstate what the deterministic lane would see. The hardened-router numbers above are the relevant baseline.

---

# Part IV — What exists in this repo now

- `dmc/canonical.py` (RFC 8785-style hashing), `dmc/lockfile.py` (toolchain identity), `dmc/ir.py` (typed IR v0: twelve op types, `Context` with `session_state_hash`, `parent_seed`), `dmc/store.py` (content-addressed replay; a divergent recompute raises `NondeterminismError`; `Receipt` fields per §3 step 9), `dmc/metrics.py` (TARr/TARa, unique@N, first divergence, PIR, flips, ECE), `dmc/moe_probe.py` (router margins, flips, stable top-k, AUC, task specificity), `dmc/routing.py` (R7 `harden_routers`; token-verified `RouteTracer`/`RoutePinner`; fragile-routing share), `dmc/runtime.py` (deterministic CPU backend: R7 router hardening **on by default** and recorded in the lock, explicit system turn, template clock bound to `Context.now`, one request per pass, counter-based sampling; table backend for tests), `dmc/http_backend.py` (OpenAI-compatible backend for GPU Phase 0; its `score` relies on the server returning echoed prompt log-probs, which must be verified under the batch-invariant flag before use), `dmc/profiles/decide.py` (order-averaged label scoring, marginalized over case and leading-space surface forms by default, with abstention and a rendered-prompt hash), `dmc/compiler.py` (pipeline, rule parser with context minimization, `DISPATCH_TABLE`), `dmc/conformance.py` (variation harness), `dmc/oracle.py` (synthetic-oracle generator), `dmc/scheduler.py` (compile-time scheduler: executor ladders priced on joint calibration records or, without them, on independent summaries; contract filtering; compiled tool-output caps and input reads; worst-case token bounds; lanes; executor-keyed prefix groups; schedule hash including the lock hash; programs with guards, pre-compiled variants, a recompile allowance and a capability ceiling, abstaining on exhaustion), `dmc/expansion.py` (goal-gated new experts: routed and additive variants, parent-row router init with noise on the copy only, bit-identical to the base when the goal is off), `dmc/expert_lora.py` (hot-expert LoRA on fused expert weights, identity at init: item i).
- `tests/`: unit tests for all of the above, plus `tests/test_plan_citations.py`, which fails if this document cites a `dmc/...` file or symbol that does not exist.
- `experiments/`: CPU experiments on granite-3.1-1b-a400m with pre-registrations; JSON results in `experiments/results/`. `experiments/gpu/`: engine-conformance script and runbook for GPU Phase 0 (not run here).

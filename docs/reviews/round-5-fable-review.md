# Round 5 — Fable 5.1 adversarial review of PLAN v6

**Reviewed:** `docs/PLAN.md` v6 at HEAD `d323b58` (plan text unchanged since `d198647`, which added
`dmc/scheduler.py::cap_input` and `Receipt.cap_hits`), `docs/reviews/round-4-response.md`,
`dmc/scheduler.py`, `dmc/expansion.py`, `dmc/expert_lora.py`, `dmc/store.py`, `tests/test_scheduler.py`,
`experiments/06_goal_gated_experts_prereg.md` + script, `experiments/07_system1_cascade_prereg.md` +
script, the pilot-03 intermediate snapshot (`experiments/results/03_route_pinning.json`, 151/300 rows),
`experiments/results/01_router_atlas.json`, wave-1 note `02_moe_routing_and_models.md`.
**Date:** 2026-10-21.

**What I ran.** `.venv/bin/python -m pytest -q`: **80 passed** (the response says 79; the `cap_input`
test is the 80th). A four-line program with two variants and chained guards through
`compile_program` + `GuardRuntime` (N3). A recount of `E_decide_groups_greedy_text` in the 01 results
(N13). Line checks of every number cited below. No new web research this round; everything is checked
against the repo and the notes.

**Standard.** As before: a finding is resolved when the plan text, the code and the pre-registration
say the same thing and the number or mechanism survives a check, not when the response says
"accepted".

---

## 1. Round-4 findings — are they actually resolved?

| ID | Status | Evidence |
|---|---|---|
| M1 infeasible targets | **PARTIAL** | §I.1.1 now has a real model (r·τ_h + e + s) and a scenario table; good. But the requirements table (§I.1.2 line 53) demands cost-of-pass ≤ 0.5× **on every suite** with e ≤ 0.15, while §I.1.1 line 36 plans for τ_h ≤ 1.2 on agentic suites and its own table (line 42) prices that scenario at **0.55**. Line 45 even says 0.5× needs τ_h ≤ 1.05. The requirement contradicts the model that derives it → N1. |
| M2 served artifact | **RESOLVED** | §I.2.3 A/B arithmetic checks: 126M × 48 ≈ 6.05B; 304/8 = 38; six 36 GB FP8 family replicas + hub-base + 27B (54 GB) on 8 × 141 GB fits. One residual on A's cost basis → N7 (minor). |
| M3 GDN prefix gate | **RESOLVED** | G0-3 (line 206), §I.3 "not credited", hub row says both lanes (line 375). |
| M4 agentic spike + switch | **PARTIAL** | Spike in G0–G1 with a gate and switch criteria fixed now (lines 115, 172). But the gate is under-resolved for its threshold and has an undefined inconclusive band → N6; the e band between the requirement (0.15) and the switch trigger (0.30) is undefined → N1(c); the response's statement that the notes do not record the Flash-Next license is false → N12. |
| M5 allocation | **RESOLVED** | Tier arithmetic adds up: Tier 1 = 44 node-weeks ≈ 41K GPU-h available vs ≈ 33–36 node-weeks used (T3 = 28 node-weeks ≈ 26.3K); Tier 0 = 28 node-weeks ≈ 26K vs ≈ 16–17 used. Trivial: T3 row says "~25K" (line 174), tier row "≈ 26K" (line 185). |
| M6 joint calibration | **RESOLVED** | `Outcome`, `_estimate_joint` replays item by item, tokens are recorded spend, estimate type recorded per assignment, `input_cap`/`input_reads`, executor-keyed prefixes, lock hash; tests cover correlated errors and tokens. Two small residuals → N10. |
| M7 guards/variants | **PARTIAL** | `Guard`/`Program`/`compile_program`/`GuardRuntime` exist and are tested for **one** variant. The program bound covers only the most expensive variant while the runtime lets every variant run once, so the "by construction" claim is false for ≥ 2 variants (demonstrated) → N3. Dry-run/confirmation nodes and the recompile re-entry with the remaining budget are prose only. |
| M8 pilot 06 | **RESOLVED** | Arms, matched rank (`params_per_expert`), noise on copy and new row only (`expansion.py` lines 72–82; plan line 131 states the deviation), wrong-goal test on held-out families, "cannot speak to production" all verified in prereg and script. One prereg wording point → N14. |
| M9 engine items | **RESOLVED** | E1–E4 at line 124 with artifact mapping; FLOPs per variant. |
| M10 thinking control | **RESOLVED** | G0-4; compiler v1 on/off + early exit only (line 132). |
| M11 stage edges | **PARTIAL** | Edges are stated (line 179) and the lock hash is in the schedule hash (verified). The §I.6 roadmap violates the edges it sits under → N4. |
| M12 memory/caps/registration/goal-suite | **PARTIAL** | Prose is in place (§I.2.1, §I.2.2, §I.5). Goal-suite v0 is a G0 deliverable at line 198 but a weeks-0–6 line item at line 188, and it is too small for the ±15% audit it is supposed to feed → N5. |
| M13 Part II numbers | **PARTIAL** | R27 320 B/token, R26/§II.7 memory, S3 Qwen-Scope, P0 updated. Stale text remains: R26 still *mandates* ≥ 3 co-resident adapters although §I.2.3 rejects multi-LoRA serving; §II.7 "P5: 2–4 nodes"; "Hub (v5)" → N11. |
| M14 status consistency | **PARTIAL** | 03 running / 06–07 queued is consistent everywhere. The Part III 01 row still says "no" on 14/15; the JSON says 13/15 → N13. |
| M15 MoE-LPR | RESOLVED | Line 131. |
| M16 attribution | RESOLVED | Vendor-card attribution present. |
| M17 canary / hold-out | RESOLVED | §I.0.2, §I.5 (20% held out, post-cutoff top-up). |
| M18 upper bounds | RESOLVED | Line 150. |
| M19 MTP on GDN | RESOLVED | G0-5. |
| M20 AIME figure | RESOLVED | Line 132 (70.9 → 21.6). |
| M21 | RESOLVED | Under M4. |

Net: 14 resolved, 7 partial, 0 unresolved. The partials are the findings of this round.

---

## 2. Findings

### BLOCKER

None. v6 has no gap of the "target with no model" or "artifact undefined" kind. What remains is the
plan contradicting its own new machinery in six places, each fixable in a paragraph or a few lines of
code.

### MAJOR

**N1 — The cost requirement is infeasible under the plan's own agentic assumption, and the e band
0.15–0.30 has no rule.**
(a) §I.1.2 line 53: "cost-of-pass ≤ 0.5× the 27B's **on every suite**, with e ≤ 0.15". §I.1.1 line 36:
"the model therefore plans for … τ_h ≤ 1.2 on agentic suites". Line 42 prices exactly that
(1/3 · 1.2 + 0.15) at **0.55**; line 45 says 0.5× needs τ_h ≤ 1.05. So on agentic suites the
requirement fails under the planning assumption with s = 0, before any System-1 credit, and the only
re-derivation clause (r > 0.4) does not cover it. The plan must choose: per-suite targets (reasoning
≤ 0.5×; agentic ≤ 0.6× at τ_h ≤ 1.2, e ≤ 0.15), or e ≤ 0.10 on agentic, or own τ_h ≤ 1.05 on agentic
and drop the "needs more tokens on agentic work" sentence. Any of these is fine; the current text is
not. (b) The latency row (line 54) is derived from the same τ_h = 1.2 and comes out at 0.55 ≤ 0.6 —
consistent, but note it is a *mean* formula while the requirement is p50; the p50 of a mixture with
15% serial escalations is roughly the unescalated latency, so either say "mean" or require p50 and
p90. (c) §I.2.3 line 115 switches the hub if the measured escalation share exceeds 0.30 (0.70 ratio).
The requirement is e ≤ 0.15. For a measured e in (0.15, 0.30] the plan neither passes its cost gate
nor switches; state what happens (stay and report the miss with the re-derived target, or lower the
trigger to the point where ≤ 0.5× becomes infeasible: e > 0.5 − r·τ_h ≈ 0.10–0.17 at the planning
values).

**N2 — The δ_b power claim is wrong as stated; SE_b is the wrong standard error.**
Line 51: δ_b = max(1.5, 100/n_b, 2.8·SE_b), "SE_b is the paired standard error of the 27B against
itself (two independent avg@16 runs)", and "2.8·SE_b gives a system exactly as good as the 27B ≥ 80%
power". The arithmetic behind 2.8 is right *for the SE of the tested difference*: pass iff
D̂ − 1.96·SE_D ≥ −δ; with true D = 0, P(pass) = P(Z ≥ 1.96 − δ/SE_D) ≥ 0.80 iff δ ≥ 2.80·SE_D. But the
tested difference is GCAM − 27B, and SE_D ≥ SE_b with strict inequality whenever the two systems have
different per-item success profiles: Var(D̂) = Var_items(p_G(i) − p_27(i))/n + sampling noise, and the
first term is zero only for a model against itself. Two different models at equal mean accuracy
disagree on many items (the 01 atlas and the A3B §5 null controls both show this is the usual case),
so with δ_b = 2.8·SE_b the power is below 80% — by how much depends on the between-model item
variance the plan does not measure. Fix: define SE_b as the paired SE of (hub-base − 27B) measured in
G0 (both systems are evaluated there anyway; GCAM's profile will lie between hub-base's and the
27B's, so this is the natural conservative input), or keep SE_b as the self-SE and state the power
claim as conditional on SE_D ≈ SE_b. The floor terms are fine (with avg@16 the resolution is
100/(16·n_b), so "one item" is a floor, not a resolution).

**N3 — Predictability "by construction" is false for programs with more than one variant; two
further §I.2.1 policies are prose only.**
Line 55 and §I.2.1 line 86: worst case = main + most expensive variant + allowance × reserve;
"the runtime abstains rather than exceed it". Code: `compile_program` line 388 adds
`max(v.worst_case_tokens)`; `GuardRuntime.on_failure` lines 427–431 refuses only a *repeat* of the
same variant. Guards may watch nodes inside variants (line 376 allows it), so variant X's guard can
select variant Y. Demonstration (main = 2 nodes × 200 tokens, variants X and Y of 200 each, guards
a→X, x1→Y, b→Y): `compile_program` returns bound **600**; the runtime answers `ga → X`, `gx → Y`,
`gb → abstain`; realized worst case **800 > 600**. The comment at line 386 ("counting them on top is
conservative") is true for one variant only. The test (`test_program_bounds_guards_variants_and_capabilities`)
has one variant, so it cannot catch this. Fix, pick one: (i) bound = main + Σ variants + reserve
(trivially correct; variants are small), (ii) runtime allows one variant per program execution, then
`abstain`, (iii) bound = max over guard-reachable variant chains. Add a two-variant test. Also mark as
*not implemented* in §I.2.1: the dry-run/confirmation-node requirement for `external_send`/`destructive`
(nothing in `compile_program` checks it; a `destructive` node within the ceiling compiles with no
dry-run node), the recompile's "remaining budget as its budget" (`on_failure` returns "recompile" and
tracks nothing), and the runtime truncation (see N9: `cap_input` exists, no caller).

**N4 — §I.6 schedules work in the order §I.4's stage edges forbid.**
Line 179: "T0 → T1 → re-atlas on hub-base → T2, with warm-start parents and hot sets chosen on
hub-base"; "T4 and T5 after T3". Line 207: G1 (weeks 2–8) builds "Artifact A for the first families"
— per-family hot-expert LoRA whose hot sets come from an atlas — while "Re-atlas on hub-base" is the
first item of G2 (line 208, weeks 6–14). Either A's hot sets are chosen on the pre-T1 atlas (the exact
mistake M11 was about and the A3B team's ch.2 lesson) or the roadmap is wrong. Line 208 also puts T4
(decision-token DPO) and T5 (token-efficiency v1) in G2, before T3 in G3 (weeks 10–24), against
"T4 and T5 after T3". Fix: move "re-atlas on hub-base" into G1 after T1 (T1 is 1–2 weeks, so weeks
~4–5), and either move T4/T5 after T3 or amend the edge to "T4/T5 v1 may run on hub-base before T3
and are re-run after T3" — the second is probably what was meant, but the plan must say one thing.

**N5 — Goal-suite v0 is too small and too late for the gates that depend on it.**
Line 198: v0 = "≥ 100 goals … 20% held out until G4" (≈ 80 development goals), "G0 deliverable".
Line 188: "1 engineer × 6 weeks (weeks 0–6) for goal-suite v0" while G0 is weeks 0–3 (line 206) and
G0-7 is a G0 exit item. Line 55: the ±15% predicted-vs-realized audit is "on suite means over ≥ 200
tasks"; line 207 makes that audit "on goal-suite v0 means" the G1 exit criterion. 80 development
goals cannot give ≥ 200 tasks unless a task is a (goal, run) pair, which the plan does not say and
which would turn the audit into a within-goal repeatability measure rather than a suite-level one.
Fix: size v0 at ≥ 250 goals (≥ 200 development), or define "task" as a goal run and require ≥ 200
runs over ≥ 50 distinct goals, and make the week-0–6 line item and the G0-7 deliverable agree (v0 at
G0 exit with a stated minimum, full v0 by week 6).

**N6 — The agentic spike gate cannot resolve its own threshold and has no inconclusive rule.**
Line 172: gate = "terminal-eval gain ≥ +4 points with paired CI lower bound > 0"; line 115: switch
if "the spike's gain has an upper 95% bound below +4". The terminal evals in the notes have ~90 tasks
(Terminal-Bench 2.x: 89). A paired pass@1 difference over 89 tasks has SE ≈ 3–4 points, so a CI half-width
of ≈ 7: *pass* needs a point gain above ≈ 7, *switch* needs a point gain below ≈ −3, and everything
between −3 and +7 — the whole plausible range around the +4 threshold — is neither, with no stated
consequence. The AgentWorld move the plan cites (+14) would pass; a real +4 would be a coin flip. Fix:
state the spike's eval (which suite, n), use k ≥ 4 samples per task with item-level paired bootstrap
(as the parity rows already do), compute the n needed to resolve +4 at 80% power, and write the
inconclusive rule (stay, re-run with more steps, or decide on criterion (b) alone). Also name the
environment source for a spike at weeks 2–4: the in-house environments are weeks 2–14 (line 188), so
the spike must run on public environments (AgentWorld's released ones, or a Terminal-Bench train split),
and the plan should say so.

### MINOR

**N7 — Artifact A's cost basis ignores replica utilization.** Line 120 pins six family checkpoints
one per GPU. The cost model's r is per hub token at the measured throughput (G0-2); with family-pinned
single-GPU replicas under a skewed goal mix, some replicas idle and the realized $ per token is r/u
for utilization u < 1. Either measure cost-of-pass at fleet level in G1 or carry a utilization factor
for A (B has the same issue as E3, already a risk row).

**N8 — Tier/T3 figure mismatch and Tier-0 timeline.** "~25K" (line 174) vs "≈ 26K" (line 185). Under
Tier 0 the training node ends at week 30 while G4 (weeks 26–36) includes "T6 on the final hub" and
"Part II certification of D1 lanes" — certification runs need the serving SKU; say the evaluation node
covers it.

**N9 — `cap_input` is provided but not wired; the 07 pilot truncates differently.** `cap_input` has no
caller in `dmc/runtime.py`, `dmc/compiler.py` or the profiles, and `Receipt.cap_hits` has no writer;
§I.2.1 line 85 says "the runtime truncates deterministically (… `cap_input`) and counts cap hits in
`Receipt.cap_hits`". Say "function provided, wiring pending". Pilot 07 truncates head-only
(`ids[:Q_CAP]`, script line 108) and re-tokenizes the decoded text, so the s1.probe bound (1 × 128)
assumes `encode(decode(ids[:128]))` ≤ 128 tokens; the 4-token slack covers only s2. Moot in practice
(no cap hits are expected), but the pilot should call the same function the plan cites so check 4
exercises it.

**N10 — Scheduler residuals.** (a) `_ladder` compares joint expected tokens (recorded spend) against
independent costs (bounds) when some candidate ladders have records for every executor and others do
not; recorded spend ≤ bound, so selection is biased toward ladders with records. The §I.2.2 contract
requires records for every registered executor, which makes this all-or-none in practice; assert it
(raise if calibration exists for the op but misses an executor) rather than fall back silently.
(b) Line 300 subtracts `reach × (call_tokens(cap) − call_tokens(cap, discount))` from joint expected
tokens; the discount is capped by the bound, not by the recorded spend, so a per-assignment expected
value can in principle go negative; clamp at 0.

**N11 — Stale Part II text.** R26 (line 283) still begins "MUST certify D1 with ≥3 co-resident
adapters", and P1's exit criterion (line 494) is "D1 on hub + ≥3 co-resident adapters on one SKU, or
fallback in lock", while §I.2.3 line 122 rejects multi-LoRA serving and A is merged checkpoints. Make
the merged checkpoint the certification reference (one lock per C_f) and the adapter-mix fuzzing
conditional on batch-by-adapter ever being used. §II.7 line 501 "P5: 2–4 nodes" vs the 1–2-node tiers.
Line 375 "Hub (v5)".

**N12 — Flash-Next license is in the notes.** The response says "our notes do not record the license
name". Wave-1 note `02_moe_routing_and_models.md` lines 326 and 362 record `qwen-community-1.0`. The
plan's "to be confirmed from the card" hedge is fine; cite the recorded name and check the terms
(community licenses commonly carry usage conditions) before the switch criteria can be exercised.

**N13 — Part III 01 row.** "\"no\" on 14/15" (line 529). `E_decide_groups_greedy_text` =
[[no,no,no],[no,no,no],[no,no,no],[no,no,no],[no,other,yes]] → "no" on **13/15**.

**N14 — Pilot 06 prereg wording.** Two primary comparisons (routed vs lora, additive vs lora) at 95%
each with no multiplicity statement; say both must clear, or Holm. State the interpretation rule for
the controls now: if `random_parent` also clears the LoRA bar, the result supports "new capacity
helps" but not "the discovered parent matters"; if `router_only` does, it supports neither.

**N15 — 03 intermediate snapshot.** Committed at `d323b58`, labelled "not final, not cited" — correct,
and the plan cites no number from it (verified). Keep it that way until 300/300 and the pre-registered
analysis; do not update the Part III row from a partial file.

---

## 3. Code-matches-plan check

| Plan claim | Code | Verdict |
|---|---|---|
| Ladders priced on joint records, estimate type recorded (line 85) | `_estimate_joint`, `Assignment.estimate` | matches |
| `op#difficulty` calibration keys | `_ladder` line 223 | matches |
| `input_cap` counted in worst case; `input_reads` per executor | `call_tokens`, line 301 | matches |
| Prefix groups keyed (lane, group, executor) | `seen_prefix` line 295 | matches |
| Schedule hash includes lock | line 319–322 | matches |
| Program worst case = main + most expensive variant + reserve | line 388 | matches the text — and the text is wrong (N3) |
| Each variant at most once; recompiles stop at allowance; then Abstain | `GuardRuntime` | matches |
| Capability ceiling over main and every variant | lines 366–373 | matches |
| Program hash covers main, variants, guards, allowance, ceiling | lines 392–395 | matches (also budget) |
| Dry-run/confirmation nodes before irreversible effects | none | **prose only** (N3) |
| Runtime truncation + `Receipt.cap_hits` | `cap_input` defined, never called; field never set | **provided, not wired** (N9) |
| Goal-off bit identity (line 131) | `forward` returns `self.base(...)` when no goal | matches; 06 tests it empirically |
| Noise on copy and new row only | `expansion.py` lines 72–82 | matches |
| Matched-rank LoRA baseline | `params_per_expert`, script lines 193–196 | matches |
| 07 executor sizes are bounds; expected tokens from records | script lines 178–185; `Outcome` list | matches |

---

## 4. Fidelity to the user's brief

Unchanged from round 4 and still intact: Groq-style compile-time scheduling is a real artifact with
static ladders, worst-case bounds, lanes and residency; goal-based MoE is defined at compile time
(family checkpoint / goal-gated experts) and token time; the base is Qwen3.6-35B-A3B with the user's
A3B recipe; (i) hot-expert LoRA/Artifact A, (ii) goal-gated experts + E1–E4 (new *blocks* demoted to a
fallback with a stated reason, line 131 — a reasoned deviation, not an omission), (iii) spokes and
verifier-selected ensembles, (iv) decision heads/probes/T6 and pilot 07, (v) GBDT/forecasting/kNN
executors under one contract; token economics now has a model. The parity target (27B at xhigh) is
stated with a non-inferiority rule — whose power claim needs N2. The one place the plan over-promises
the user is "predictability by construction" (N3) and "≤ 0.5× on every suite" (N1); both are fixable
without changing the design.

---

## 5. What v6 gets right

- A feasibility model with named quantities (r, τ_h, e, s), measured in G0, with a scenario table
  and a re-derivation clause; KV reuse credited nowhere; "effects decompose, none credited".
- The served artifact is defined: A from G1 with no engine work, B as the target under a
  pre-registered A-vs-B rule; sizes and EP divisibility check; multi-LoRA serving rejected in writing.
- Allocation tiers whose node-week arithmetic adds up, with environment construction, goal-suite and
  engine work as headcount line items.
- Joint calibration in code: per-item `Outcome` records, ladder replay, recorded spend for tokens, the
  estimate type on every assignment, a test that shows the independent estimate over-predicting.
- Compiled input caps with `input_reads`, executor-keyed prefix groups, lock hash in the schedule hash,
  capability ceiling enforced over every variant, bounded recompiles that end in Abstain.
- Pilot 06 amendment: matched-rank LoRA baseline, router-only and random-parent controls, noise on
  the copy only with the deviation stated in the plan, wrong-goal test on held-out families, honest
  scope statement.
- Pilot 07 amendment: the dry run caught mean-sized executors before any real run; compiled 128-token
  cap and template bounds; joint vs independent reported side by side; a stated consequence for a
  calibration-transfer failure.
- Pre-registration hygiene held: dated amendments before runs; the 03 snapshot committed as
  intermediate and not cited.
- GDN-specific gates (G0-3 prefix caching, G0-5 MTP, G0-6 batch invariance) and thinking control
  measured before the compiler relies on it.
- The agentic risk is faced early (spike in G0–G1) with switch criteria written now, and the
  same-generation evidence is scoped to reasoning only.
- The goal-suite hold-out and post-cutoff top-up carry the A3B canary lesson into the gates.

---

## 6. Verdict

**VERDICT: APPROVE WITH CHANGES**

Minimum changes before v7 is the working plan:

1. **N1** — Make the cost requirement consistent with §I.1.1: per-suite targets (or a tighter e or τ_h
   on agentic suites, chosen explicitly), and a rule for measured e in (0.15, 0.30].
2. **N2** — Define SE_b as the paired SE of (hub-base − 27B) measured in G0, or state the 80% power
   claim as conditional on SE_D ≈ SE_b; publish δ_b from that.
3. **N3** — Fix the program bound (Σ variants, or one variant per execution, or the max reachable
   chain) with a two-variant test; mark dry-run/confirmation nodes, recompile re-entry and runtime
   truncation as not yet implemented in §I.2.1.
4. **N4** — Re-order §I.6 to obey §I.4: re-atlas on hub-base inside G1 before Artifact A's hot sets;
   T4/T5 after T3 or an amended edge that says v1 runs before and is re-run after.
5. **N5** — Size goal-suite v0 for the ≥ 200-task audit (or define "task" as a run and say so) and make
   its week-0–6 line item agree with the G0-7 deliverable.
6. **N6** — Give the spike gate an eval, an n and a k that resolve +4 at 80% power, and an
   inconclusive rule; name the spike's environment source.

Recommended (not required): N7–N15.

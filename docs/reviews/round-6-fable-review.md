# Round 6 — Fable 5.1 adversarial review of PLAN v6.1 and the architecture blueprint

**Reviewed:** `docs/PLAN.md` v6.1 at HEAD `34f8f1e` (diff against v6 read in full), `docs/reviews/round-5-response.md`,
`dmc/scheduler.py` and `tests/test_scheduler.py` (diffs and the three new tests), `experiments/06_goal_gated_experts_prereg.md`
amendment 2 + script comparison block, `experiments/07_system1_cascade_prereg.md` amendment 2 + `s1.probe` slack,
`docs/blueprint.html` (all 11 sheets, read in full, every number and coordinate checked against the plan, the code
and the tests). **Date:** 2026-10-21.

**What I ran.** `.venv/bin/python -m pytest -q`: **83 passed**. Pixel arithmetic for Sheets 02, 04, 08 and 10
recomputed from the stated formulas and week ranges. The round-5 two-variant case is now the test
`test_program_bound_covers_a_chain_through_several_variants`, and it passes with the sum bound.

---

## 1. Round-5 findings — resolved?

| ID | Status | Evidence |
|---|---|---|
| N1 cost target vs agentic τ_h | **RESOLVED** | §I.1.1 consequences and §I.1.2 Cost row: ≤ 0.5× reasoning + goal-suite (τ_h ≤ 1.05), ≤ 0.6× agentic (0.4 + 0.15 = 0.55); e band (0.15, 0.20] stay + report, > 0.20 switch (b); latency is a mean with p50/p90 reported. Arithmetic consistent everywhere (plan, Sheet 08 chart, Sheet 10 text). One residual on how (b) is applied → P3. |
| N2 δ_b power claim | **RESOLVED as asked** | SE_b = paired SE of (hub-base − 27B), provisional from base in G0, fixed at G1; power stated conditionally; self-SE reported as the floor. The fix has a consequence the plan does not state → P2. |
| N3 program bound / prose-only policies | **RESOLVED** | `compile_program` sums every variant (line ~410) with the two-variant chain test; `Program.require_confirmation` + `IRREVERSIBLE`/`CONFIRM_OPS` tested; §I.2.1 marks `cap_input` wiring and recompile remaining-budget accounting as pending. One semantics gap in what the confirmation dependency actually guarantees → P4. |
| N4 roadmap vs stage edges | **RESOLVED** | G1: T1 → hub-base → re-atlas (weeks ~4–5) → Artifact A from that atlas; G2: T2 from the G1 atlas, T4/T5 v1 "recipes only"; G3: re-run after T3; edge text amended to match. Sheet 07 and Sheet 10 say the same. |
| N5 goal-suite size/timing | **RESOLVED** | ≥ 100 at G0 exit, ≥ 250 by week 6 (≥ 200 development, ≥ 50 held out); audit = one run per development goal; week-0–6 line item now says "≥ 100 by week 3, ≥ 250 by week 6". Staffing realism noted → P6. |
| N6 spike gate | **PARTIAL** | Public training tasks, ≥ 300 held-out tasks, k = 4, n fixed from a G0 SD estimate for SE ≤ 1.4, pass / (a) / inconclusive rule. The pass rule's power is not what the text says and the k/n example does not reach its own SE target → P1. |
| N7 fleet-level cost under A | RESOLVED | Cost row: utilization u divides r. |
| N8 25K/26K, Tier-0 G4 node | **RESOLVED in the plan; stale in the blueprint** | Plan says ≈ 26K in both places; Sheet 07's table still says "Tier 1 ≤ ~25K" → P7. |
| N9 `cap_input` wiring; 07 truncation | RESOLVED | Plan: "wiring pending (P1)"; 07 amendment 2 adds `TEMPLATE_SLACK` to `s1.probe` (script line 183) and says check 4 does not exercise `cap_input`. |
| N10 scheduler residuals | RESOLVED | Partial records for an op raise `CompileError` (tested); per-assignment expected tokens clamped at 0. Small robustness note → P4(c). |
| N11 stale Part II text | RESOLVED | R26 rewritten around merged checkpoints and B's goal-bias on/off; P1 exit criterion and matrix updated; P5 shares the Part I allocation; "Hub". |
| N12 Flash-Next license | RESOLVED | `qwen-community-1.0` cited with "terms to be checked". |
| N13 Part III 01 row | RESOLVED | "no" on 13/15, "yes" on 1, one unparseable — matches the JSON. |
| N14 06 multiplicity + control rule | RESOLVED | Amendment 2: Bonferroni 97.5% CIs for the two primaries (script computes 95% and 97.5%, `boot_diff(level=0.975)` indices 12 and 986 of 1000 — correct); `CLEARS_LORA_BAR` for all four arms; `SUCCESS` only for `routed`/`additive`; interpretation rule fixed before any run. |
| N15 03 snapshot uncited | RESOLVED | Still uncited. |

Net: 13 resolved, 1 partial (N6), 1 resolved in the plan but not in the blueprint (N8).

---

## 2. Findings

### BLOCKER

None.

### MAJOR

All three are gate or decision-rule definitions, each fixable in one to three sentences. None requires a
design change. I list them as MAJOR because each is a pre-registered rule the plan will be held to, and
each, as written, does not do what the surrounding text says it does.

**P1 — The spike gate's pass rule has 50% power at the effect it claims to resolve, and the k/n example
does not reach its own SE target.**
§I.4 spike row (line 172): "n is fixed … so that SE ≤ 1.4 points (+4 resolvable at 80% power). Gate:
pass = point gain ≥ +4 **and** CI lower bound > 0". With SE = 1.4 and a true gain of exactly +4, the
lower-bound condition alone passes with probability P(D̂ > 2.74) ≈ 0.82 — that is the 80% figure. The
conjunct "point gain ≥ +4" passes with probability 0.50 at a true +4, and since point ≥ 4 implies lower
bound > 0, the conjunction's power is **50% at +4** and reaches 80% only at a true gain of ≈ +5.2. Either
drop the point conjunct (pass = CI lower bound > 0; (a) = upper bound < +4; else inconclusive), or keep it
and state the power honestly. Second, the example "≥ 300 held-out tasks, k = 4" cannot give SE ≤ 1.4 on
the sampling term alone: a paired per-task difference of two k-sample means has variance
[p_b(1−p_b) + p_a(1−p_a)]/k; at p ≈ 0.33 → 0.47 (the AgentWorld move the plan cites) and k = 4 that is
0.118, SD 0.34, SE at n = 300 ≈ **2.0 points** before any between-task heterogeneity. SE ≤ 1.4 needs
n ≈ 600 at k = 4 — more disjoint held-out terminal tasks than exist publicly (Terminal-Bench 2.x has 89) — or
**k = 8 at n = 300** (variance 0.059, SE ≈ 1.4 exactly). The plan's procedure (fix n from a G0 SD
estimate) would discover this, but the pre-registration should not start from an example that its own
arithmetic rules out: write k = 8 (or "k chosen so that SE ≤ 1.4 at n = 300"), and say what happens if the
required n exceeds the available tasks (pool more public sets, or widen the inconclusive band and decide on
(b) alone). Sheet 07 and Sheet 10 carry the same "≥ 300 tasks, k = 4" text and need the same fix.

**P2 — After the N2 fix, δ_b is unbounded on small suites and the plan does not say what it will be or
what happens when it is large; the blueprint asserts a rule the plan does not contain.**
§I.1.2 now defines δ_b = max(1.5, 100/n_b, 2.8·SE_b) with SE_b the paired SE of (hub-base − 27B). That is
the right SE for the power argument, and it has a consequence: the between-model term dominates and does
not shrink with more samples per item, only with more items. Rough sizes at avg@16: AIME pooled (90
items) with 10–20% of items where one model solves and the other does not gives SE_b ≈ 3–4 points →
**δ_b ≈ 9–12 points**; GPQA-Diamond (198) ≈ 6–8; LiveCodeBench v6 ≈ 2–3; HLE and MMLU-Pro ≈ 1–2. A
non-inferiority margin of ten points on AIME is not "parity" in the sense §I.0.1 or the user's brief
means it, and a reader of the requirements table cannot see this from the formula. The plan needs one of:
(i) a cap on δ_b (e.g. 4 points), with benchmarks whose 2.8·SE_b exceeds the cap moved out of the gate
into "reported, resolution-limited" or into a pooled reasoning composite (AIME is already pooled across
three years; a composite over AIME + GPQA-Diamond + LCB raises n_b); or (ii) an explicit statement of the
expected δ_b per benchmark from G0 numbers, with the small-suite ones labelled as resolution-limited.
`docs/blueprint.html` Sheet 10 (G0-1, "If it fails: Benchmark leaves the gate or is pooled") already
describes mechanism (i) — but the plan does not contain it. The blueprint must not promise a rule the plan
lacks; put the rule in §I.1.2 and keep the sheet.

**P3 — Switch criterion (b) is under-specified in three ways that matter for the most consequential
decision in the plan.**
§I.2.3: "(b) the escalation share the scheduler needs for system parity on the agentic suites, measured on
G1 calibration data, exceeds e = 0.20 (… 1/3 × 1.2 + 0.20 = 0.60)". (i) *Which checkpoint?* G1 calibration
data come from hub-base (post-T1, pre-RL) or from the spike checkpoint; the plan does not say. Measured on
a pre-RL hub, the needed escalation is exactly what T3 is meant to reduce, so (b) would be evaluated on
the hub the plan does not intend to ship. State that (b) is measured on the spike checkpoint (the best
agentic hub available at G1), and that a *passing* spike with (b) > 0.20 is a "stay with RL headroom"
outcome unless the spike-adjusted share still exceeds the bound — as written, "(a) or (b)" switches even
when the spike passes. (ii) *0.20 is a derived constant, not the rule.* It equals 0.6 − r·τ_h at the
planning values r = 1/3, τ_h = 1.2. At G1 both are measured (G0-2 for r, per-suite τ_h), so the criterion
should be written as "measured r·τ_h + e > 0.6 on the agentic suites" with 0.20 as the planning value;
otherwise a measured r of 0.25 (A3B basis is 0.17) would trigger a switch at an escalation share the model
could afford. (iii) *The (0.15, 0.20] band* in §I.1.2 is "stay and report"; say for how long — the G3 exit
gate requires e ≤ 0.15, so a hub that sits in the band through G3 has failed the gate and the plan should
say what the G3 outcome is (report the miss against the re-derived target, which §I.1.1 already allows).

### MINOR

**P4 — Confirmation dependency orders; it does not gate. And two code residuals.**
(a) `require_confirmation` (scheduler line ~386) checks that an irreversible node *depends on* a
`meta.confirm`/`tool.dry_run` node. A dependency fixes execution order; nothing in `Guard`, `Program` or
`GuardRuntime` makes a refused confirmation stop the effect node — `on_fail` may only name a variant or
"recompile" (line 399), there is no "abstain" target, and a confirm node's negative value has no compiled
consequence. As implemented this is a lint, which is fine for v1, but §I.2.1 says "compile only if … (implemented
and tested)" in a paragraph about not steering execution into unauthorised branches. Add `on_fail: "abstain"`
as a legal guard target and require that every confirm/dry-run node that gates an irreversible node carries a
guard whose failure abstains or selects a variant without the irreversible capability; until then say
"ordering enforced; gating is IR v1". (b) `_ladder` checks for missing executor records against `items[0]`
only; a later item missing an executor raises `KeyError` inside `_estimate_joint` instead of `CompileError`.
(c) `GuardRuntime` still tracks no tokens, which the plan now states; fine.

**P5 — Artifact A's per-family adapter training has no compute line and no time box.**
§I.4's tier table has columns T1, Spike, T2, T3, T5. Item (i) — rank-16 LoRA on ~97/256 experts per layer per
family, merged into C_f — is neither T1 (one hub-base) nor T2 (item ii, 10–100 GPU-h per family). G1 says
"Artifact A for the first families" without a count or a GPU-hour figure. At A3B-report scale this is a few
hundred GPU-h per family; six families fit inside Tier 1's ≈ 5K GPU-h slack and Tier 0's ≈ 9K, so it is a
bookkeeping omission, not an infeasibility — but under Tier 0 it shares weeks 5–8 of one node with the spike
and the tail of T1, so the G1 time box is tight. Add a row (item (i): n families × GPU-h) and name the
"first families".

**P6 — Goal-suite assumptions.** (a) §I.1.1 assumes τ_h ≤ 1.05 for goal-suite v0, but v0 spans agentic-tool
and research families whose planning value is 1.2; the ≤ 0.5× goal-suite target therefore relies on
System-1/code taking nodes off the hub (s ≈ 0 is then not the right assumption either). Say that explicitly
or report goal-suite cost per family. (b) 1 engineer × 6 weeks for ≥ 250 goals with executable verifiers or
two-judge rubrics is ≈ 8 verified goals per working day; plausible only with templated families. Note it.

**P7 — Blueprint drift from v6.1 (everything else checks out, see §3).** Footer still says "plan v6"
(line 1039). Sheet 07 table: "Tier 1 ≤ ~25K" (line 815) vs the plan's ≈ 26K. Sheet 02: the "eight nodes"
example gives n8 `external_send` "after a confirmation node", but no confirm node is in the table or the
Gantt — with `require_confirmation` on by default this program would not compile; add the ninth node. Sheet
05 claim text: "the bias is −∞, so the layer computes exactly what the base layer computes" — in the code the
bit-identity path is `active_goal is None → base(h)`; the −∞-bias path (304-wide FP32 GEMM, stable sort)
is what Artifact B's serving test checks, not a by-construction identity; the figcaption has it right, the
claim sentence does not. Sheet 08 claim: formula written without the s term; harmless, but it is the plan's
formula minus a symbol.

---

## 3. Blueprint verification (what I checked)

| Sheet | Check | Result |
|---|---|---|
| 01 | 40 × 8 routes/token; ~3B active, 40 layers, 256 experts; hub D0 + D2 until audit; five-item mapping | matches plan (R27, §I.2.3, §I.0.1) |
| 02 | Table sum 0 + 3,000 + 4,600 + 16,000 + 5,500 + 0 + 2,800 + 0 = 31,900; n3 = 600 + 2 × 2,000 matches `input_reads=2` test arithmetic | matches; confirm node missing (P7) |
| 03 | 10 items, S1 coverage 0.5, hub 0.9 marginal / 4 of 5 hard, hub bound 100, hard items 200 tokens; independent 0.95 / 50, joint 0.90 / 100; 0.93 bar → CompileError | matches `test_joint_calibration_accounts_for_correlated_errors` line for line |
| 04 | 31,900 + 14,500 + 8,000 = 54,400 ≤ 60,000; bar widths 149 : 68 : 37 px at 280 px / 60,000 | matches; every-variant wording matches code |
| 05 | k = 6 × 8 = 48; noise on copy only; identity path cited to `tests/test_expansion.py` | matches code; claim sentence overstates (P7) |
| 06 | 6 FP8 family replicas + hub-base + 27B on 8 GPUs; 304 = 8 × 38; ~6B / ~6 GB FP8; E2 for A from G2 | matches §I.2.3 |
| 07 | Edge order T0 → T1 → hub-base → re-atlas → T2; T3 → re-atlas → T4 → T5 → T6; T4/T5 v1 note; spike dashed | matches §I.4 v6.1; T3 figure stale (P7) |
| 08 | Axes: x = 90 + 1500·e, y = 370 − 400·ratio. Lines (90,210)→(690,50) = 0.4 + e; (90,236.7)→(690,76.7) = 0.333 + e; (90,303.3)→(690,143.3) = 0.167 + e. Targets y = 170 (0.5), 130 (0.6), 236.7 (0.33). Points at x = 315 (e = 0.15): y = 150 / 176.7 / 243.3 = 0.55 / 0.48 / 0.32. Switch line x = 390 (e = 0.20) crosses the agentic line at y = 130 (0.6). r priors 44/134 = 0.33, 0.075/0.449 = 0.17 | all correct to the pixel; matches the plan's table and §I.1.2 |
| 09 | D0–D4/D⊥ promises; 27B "D0 until its GDN audit" | matches Part II (27B is a dense GDN hybrid, §II.4.1) |
| 10 | 23.9 px/week: G0 0–3, G1 2–8, G2 6–14, G3 10–24 (+6 Tier 0), G4 20–30 (+6); gates at weeks 3, 8, 14, 24, 30; G0-7 sizes; switch text at 0.20 | matches §I.6 v6.1; G0-1 failure rule not in plan (P2) |
| 11 | 83 tests; pilot statuses and numbers (7.8%, AUC 0.87, 68%/96.7%, 13.7×) | matches Part III |

---

## 4. Code-matches-plan (v6.1 deltas)

| Plan claim | Code | Verdict |
|---|---|---|
| Bound = main + every variant + reserve; two-variant chain tested | `compile_program` sum; `test_program_bound_covers_a_chain_through_several_variants` | matches |
| Irreversible nodes need a direct confirm/dry-run dependency, default on | `IRREVERSIBLE`, `CONFIRM_OPS`, `Program.require_confirmation`, test | matches (ordering only, P4) |
| Partial calibration refused | `_ladder` missing-records check, test | matches (P4b edge) |
| Expected tokens clamped at 0 | line ~312 | matches |
| `cap_input` wiring pending; recompile budget untracked | no callers; `GuardRuntime` | matches the plan's new wording |
| 06: Bonferroni 97.5%, controls compared, SUCCESS only for primaries | script lines 262–276 | matches prereg amendment 2 |
| 07: `s1.probe` fixed size = 4-token slack | script line 183 | matches prereg amendment 2 |

---

## 5. Fidelity to the user's brief

Unchanged and intact. Groq-style compile-time scheduling is implemented machinery with a now-correct
worst-case bound; goal-based MoE is defined at compile time (family checkpoint / goal-gated experts) and
token time; the base is Qwen3.6-35B-A3B with the user's A3B recipe; items (i)–(v) map to §I.2.3–§I.2.4
mechanisms and the blueprint's Sheet 01 list; parity is a non-inferiority rule against Qwen3.8-27B; token
economics has a model whose chart is now correct to the pixel. The one place the user could be surprised
is P2: on the two smallest headline suites "parity" will be a wide margin, and the plan should say so
before G0 publishes δ_b.

---

## 6. What v6.1 gets right

- The round-5 arithmetic contradictions are gone: per-suite cost targets derived from the one formula, an
  e band with a rule, a mean latency requirement, a sum bound that the reviewer's own counter-example now
  tests.
- The δ_b definition uses the right standard error and states its power claim conditionally — the honest
  version, whose consequence P2 asks the plan to spell out.
- The roadmap obeys the stage edges it sits under, and the edge for T4/T5 says what "v1 before T3" means.
- Goal-suite v0 is sized for the audit it feeds, with the hold-out counted.
- The spike has a training-data source, a held-out eval, an SD-driven n, and a three-outcome rule — the
  structure is right; P1 is arithmetic.
- `require_confirmation` is a real compile-time check with a test, and the plan now distinguishes
  "implemented" from "specified" for truncation wiring and recompile accounting.
- Pilot amendments 2 (06, 07) are dated, before any run, and the 06 interpretation rule for the controls
  is fixed in advance — so a positive result cannot be re-read after the fact.
- Part II is cleaned up around the served artifact (R26, P1, P5) rather than carrying v4 text that
  contradicted Part I.
- The blueprint is a faithful drawing of the plan: Sheet 03 is the unit test, Sheet 04 is the bound
  arithmetic, Sheet 08 is the formula, Sheet 10 is §I.6 — all verified. The drift in P7 is cosmetic.
- The 03 snapshot remains uncited while the run completes.

---

## 7. Verdict

**VERDICT: APPROVE WITH CHANGES**

Structurally the plan has converged: no blocker, no design change, no new contradiction between the
feasibility model, the artifact story, the allocation and the roadmap. The three MAJOR items are
gate-definition fixes that must land before the pre-registrations they govern are published:

1. **P1** — Spike gate: state the power of the pass rule as written (50% at +4) or drop the point-estimate
   conjunct; replace the "≥ 300 tasks, k = 4" example with one that meets SE ≤ 1.4 (k = 8 at n = 300, or n
   from the SD estimate), and say what happens if the required n exceeds the available held-out tasks. Same
   text in Sheets 07 and 10.
2. **P2** — δ_b: add a cap or a pooling/"resolution-limited" rule for benchmarks where 2.8·SE_b is large,
   and show the expected δ_b per benchmark; put the rule the blueprint's Sheet 10 already describes into
   §I.1.2.
3. **P3** — Switch criterion (b): name the checkpoint it is measured on, define the outcome when the spike
   passes but (b) exceeds the bound, and write (b) as the formula at measured r and τ_h with 0.20 as the
   planning value.

Recommended, not required: P4–P7 (P7 is a five-minute blueprint sync).

# Round 7 — Fable 5.1 adversarial review of PLAN v6.2, the blueprint and its PDF

**Reviewed:** `docs/PLAN.md` v6.2 at HEAD `d333981` (diff against v6.1 read in full), `docs/reviews/round-6-response.md`,
`dmc/scheduler.py` and `tests/test_scheduler.py` (diffs and the amended tests), `docs/blueprint.html` (diff),
`docs/GCAM_Blueprint.pdf` (22 pages, text extracted with `pdftotext`; cover, contents, executive summary, the two
specification tables, Appendices A–C read in full) and `docs/pdf/build_blueprint_pdf.py` (read in full, since the
added PDF content is generated there). **Date:** 2026-10-21.

**What I ran.** `.venv/bin/python -m pytest -q`: **83 passed**. A four-line program in which a variant reuses the
id of the main graph's confirmation node (Q1). `pdfinfo` on the PDF (22 pages) and on the A3B report (52 pages).
Risk-register row count: plan §I.7 has 16 rows; Appendix A has 16, verbatim. Power arithmetic for the spike gate
re-done from the plan's own numbers.

---

## 1. Round-6 findings — resolved?

| ID | Status | Evidence |
|---|---|---|
| P1 spike gate power and k/n | **RESOLVED** | §I.4 spike row: pass = CI lower bound > 0 (P(D̂ > 2.74 | μ = 4, σ = 1.4) ≈ 0.82, correctly "≈ 80%"); the point estimate is reported and the 50% figure for the conjunction is stated; k = 8 with the sampling arithmetic written out (≈ 2.0 at k = 4, ≈ 1.4 at k = 8); n and k fixed from the G0 SD estimate; what happens if n is unavailable (pool public sets, then accept and state a wider inconclusive band). Sheets 07 and 10 carry the same text; the PDF takes the sheets verbatim. |
| P2 δ_b unbounded on small suites | **RESOLVED** | §I.1.2: cap at 4 points; above the cap → resolution-limited, pooled into an item-weighted reasoning composite under the same rule; composite above the cap → reported, not gated, gate rests on the rest; expected sizes written (AIME ≈ 9–12, GPQA-D ≈ 6–8, LCB ≈ 2–3, HLE/MMLU-Pro at the floor). Blueprint G0-1 row now states this rule and nothing more. |
| P3 switch criterion (b) | **RESOLVED** | §I.2.3: (b) is measured on the spike checkpoint as r·τ_h + e > 0.6 at measured r, τ_h (0.20 = planning value); rule table: switch on (a); switch on inconclusive + (b); pass + (b) → stay and re-evaluate on the T3 checkpoint at week 17; (0.15, 0.20] fails G3, is reported, does not switch. G1 and G3 rows updated; Sheet 10 text matches. One consequence left unstated → Q2. |
| P4 confirmation gates; code residuals | **RESOLVED** | `on_fail = "abstain"` legal and handled in `GuardRuntime`; `compile_program` requires, for every irreversible node in every graph, a confirm/dry-run dependency that carries a guard abstaining or falling back to a variant with no irreversible node; tested including the leaky "retry that re-sends" variant. All-items record check tested. One robustness hole → Q1. |
| P5 item (i) adapter line | **RESOLVED** | §I.4: first three families, ≈ 15–300 GPU-h each, anchored to the A3B run's 13.7 GPU-h on one H200 (report p. ~539 of the text, "full run took 13.7 hours" — verified), weeks 5–8 after the G1 re-atlas, Tier 0 ordering stated. |
| P6 goal-suite assumptions | **RESOLVED** | §I.5: templated generators with executable checks explain ≈ 8 goals/day; per-family cost reported; the ≤ 0.5× goal-suite target is stated to rely on System-1/code taking nodes off the hub on agentic-tool and research goals at τ_h = 1.2. |
| P7 blueprint drift | **RESOLVED** | Footer/header v6.2; T3 ≈ 26K; Sheet 02 has nine nodes with `meta.confirm` n8 ("abstains if refused") before n9; Sheet 05 claim separates goal-off identity (by construction) from the −∞ path; Sheet 08 formula includes s. Gantt geometry re-scaled consistently (43 px/unit, axis ticks at 0/5/10/15). |

All seven resolved. 83 tests pass.

---

## 2. PDF added content — checked against the plan and the code

| PDF element | Check | Result |
|---|---|---|
| Cover | Plan v6.2; hub 35B total / ~3B active; reference "Qwen3.8-27B (dense)" (plan: dense GDN hybrid — "dense" is the FFN, fine); review status; 22 pages; "Compiler core · 83 tests"; revision history rounds 4/5/6 → v6/v6.1/v6.2 | matches |
| Contents note | "Numbers marked illustrative are worked examples" — Sheet 02 is the only illustrative set and is marked so in the drawing | matches |
| Executive summary | "trails the 27B by 3 points on GPQA-Diamond but by 12 on SWE-bench Pro" = 89.2 − 86.0, 61.7 − 49.5 (§I.0.1); "three done, one running, three queued" = 01/02/05, 03, 06/07/04; targets table = §I.1.2; decided/open table = §I.2.3, §I.6 | matches; one omission → Q4(a) |
| Component specification (Sheet 01 extra) | Parse → dense parser spoke (P2) + rule parser v0; Planner = hub Plan profile (G1), D0; Scheduler pure function of IR, lock, cost models; registry = T6 output; Hub D0 + D2 until the GDN audit; Spokes 27B + Skywork-Reward-V2-Qwen3-8B; trust files | matches, except the durable executor's phase → Q4(b) |
| Hub specification (Sheet 05 extra) | 40 = 10 × (3 GDN + 1 GA), hidden 2048, 256 + 1 experts, FFN width 512 (W2-03 line 334, W2-05 line 15); one expert 3 × 2048 × 512 = 3.15M, × 40 = 126M = plan §I.2.3's "≈ 126M per slot"; rank-16 LoRA 0.09M (fused gate-up) to 0.12M (separate) per expert, 97 of them ≈ 3–4 experts; k = 48 → 6.0B, 304 = 8 × 38; vocab 248,320 (plan line 15, W2-05 line 96); 262,144 native (W2-05 line 16); ~72 / ~36 GB (A3B-ch.2) | all correct |
| Appendix A risk register | 16 rows parsed from §I.7, verbatim; parser slice `## I.7 Risks` → first `# Part II` (line 214 → 237) is the right block | matches |
| Appendix B glossary | δ_b with cap and pooling; e ≤ 0.15; GDN in 30 of 40 layers; MTP "off in the bitwise lane" (= GDN audit go/no-go "MTP off or verified bitwise"); A3B voice agents; Artifact A/B | matches; two trivial drifts → Q4(c) |
| Appendix C review trail | Rounds 1–6: 21 / 19 / 15 / 21 / 15 / 7 findings = F1–F21, G1–G19, K1–K15, M1–M21, N1–N15, P1–P7; all "approve with changes"; round 4 reviewed v5 → v6; A3B report 52 pages (verified); "a test that every file and symbol the plan cites exists" = `tests/test_plan_citations.py` (exists) | matches |
| Sheets 01–11 | Taken verbatim from the HTML by `extract()`; the HTML was verified sheet by sheet in round 6 and its v6.2 diff here | matches |

---

## 3. Findings

### BLOCKER

None.

### MAJOR

None. I looked for one in the places most likely to hold it — the new gate texts, the switch rule, the gating
check, the PDF's hand-written sections — and did not find one. The plan, the code, the pre-registrations and
both renderings of the blueprint now say the same thing.

### MINOR (none required for approval)

**Q1 — Node-id collisions across main and variants bypass the confirmation-gating check.**
`compile_program` builds `gated = {g.node …}` as a set of names over all guards, then checks each graph's
confirm dependencies by name. Node ids are checked for uniqueness only within a graph (`_topo_order`). A variant
that reuses the main graph's confirm-node id compiles with an ungated send: main = (`ok` confirm, guard → abstain;
`send`), variant V = (`ok` confirm with **no** guard; `send2`) → compiles, bound 400. The same name-keyed lookup
feeds the guard-existence check and `GuardRuntime`. Fix: raise `CompileError` on any node id shared between main
and a variant (or between variants), or compute `gated` per graph. One line plus a test. This needs an author
to reuse ids, which the IR's content-addressed node ids will not do by accident, so it is robustness, not a hole
in the plan's claim.

**Q2 — A week-17 switch has no stated schedule consequence.**
§I.2.3: if the spike passes and (b) holds, (b) is re-evaluated on the T3 checkpoint at week 17, "switching only
if it still holds then". A switch at week 17 means T1 again on Flash-Next (6B active, r re-measured), re-atlas,
item (i) adapters and T3 from scratch, inside a 30-week Tier 1 plan whose G3 ends at week 24. Say what the
timeline becomes (G3/G4 slip of roughly the G1–G2 duration, ≈ 8–10 weeks, and Tier 0 later still) so the
rule's cost is visible when it is exercised; the decision itself is sound.

**Q3 — §I.1.2's Cost row still phrases the band in e alone.**
"(0.15, 0.20] … above 0.20 the switch criterion (b)" — §I.2.3 now defines (b) at measured r and τ_h with 0.20
as the planning value. One clause in §I.1.2 ("above the e at which r·τ_h + e > 0.6; 0.20 at planning values")
keeps the two sections from drifting when G0 measures r.

**Q4 — PDF hand-written sections: three small drifts.**
(a) The executive summary lists all five reasoning suites under the parity target without the §I.1.2 note that
AIME and GPQA-Diamond are expected to be resolution-limited and pooled; a reader of the summary alone would expect
per-benchmark gates on them. One clause. (b) Component specification: "Runtime … Durable executor (P1,
Temporal-class)" — the plan places the durable executor in P3 (§II.7 P3 row; Part II architecture item 10); P1 is
the deterministic serving lane. (c) Glossary: "Cost-of-pass … r·τ_h + e" omits the s that Sheet 08 now carries;
"τ_h planned 1.0 on reasoning" vs the plan's "≤ 1.05". Both harmless.

**Q5 — `tests/test_plan_citations.py` covers `dmc/` paths only.**
The plan also cites `experiments/…` scripts, prereg files and result JSONs, and the blueprint cites
`tests/test_expansion.py`. Extending the citation test to those paths would keep the renderings honest as files
move. Optional.

---

## 4. Fidelity to the user's brief

Unchanged and intact: Groq-style compile-time scheduling as working, tested machinery with a correct bound;
goal-conditioned MoE on Qwen3.6-35B-A3B with the user's A3B recipe; items (i)–(v) each mapped to a mechanism and a
test; parity against Qwen3.8-27B defined as non-inferiority with margins that are now capped and, where a suite
cannot resolve them, reported rather than gamed; token economics from a formula whose chart is correct. The PDF's
executive summary is a fair one-page statement of what the user asked for and what will be measured, with the one
clause in Q4(a) to add.

---

## 5. What v6.2 gets right

- The spike gate is now a test with stated power, stated sample arithmetic and a stated fallback — the three
  things a pre-registration needs.
- δ_b has a cap and an explicit, honest consequence for the suites that cannot resolve parity; the expected
  margins are written down before G0 measures them.
- The switch rule names its checkpoint, its formula and its interaction with the spike, and the (0.15, 0.20]
  band has an outcome.
- Confirmation is a gate, not an ordering: refusing it has a compiled consequence, and a fallback that would
  re-send is refused at compile time (tested with exactly that case).
- Item (i) training is a budget line grounded in the A3B report's measured run.
- Goal-suite production is explained rather than asserted.
- The blueprint and its PDF are generated from one source (sheets extracted verbatim; risk register parsed
  from §I.7), so they cannot drift from each other, and the hand-written PDF sections check out against the
  plan and the notes.
- The 03 snapshot (now two commits of intermediate results) remains uncited.

---

## 6. Verdict

**VERDICT: APPROVE**

Nothing at MAJOR or above remains. The plan has converged across seven rounds: its targets derive from a stated
model, its served artifact and allocation are defined and consistent with its roadmap, its gates are
pre-registered tests with known power, and its compile-time machinery is implemented with tests that encode the
review's counter-examples. Q1–Q5 are recommended housekeeping (Q1 one line plus a test; Q2–Q4 a clause each) and
can land in the next ordinary revision without re-review.

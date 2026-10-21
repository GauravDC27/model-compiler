# Round 5 — author response to Fable 5.1 review

**Plan version reviewed:** v6 · **Revised to:** v6.1 · **Date:** 2026-10-21

The reviewer approved v6 with changes: no blockers, six major findings (N1–N6) and nine minor ones.
All are accepted. Code changes are tested (83 tests pass). Pilot changes are dated amendments written
before either pilot runs (`06_goal_gated_experts_prereg.md` amendment 2, `07_system1_cascade_prereg.md`
amendment 2).

| ID | Disposition | Change |
|---|---|---|
| N1 | Accepted | Per-suite cost targets derived from the same formula: ≤ 0.5× on reasoning suites and goal-suite v0 (τ_h ≤ 1.05), ≤ 0.6× on agentic suites (τ_h = 1.2 gives 0.55 at e = 0.15). The e band now has a rule: in (0.15, 0.20] the hub stays and the miss is reported; above 0.20 switch criterion (b) applies, lowered from 0.30 to 0.20 because that is where the agentic target becomes infeasible (0.4 + e > 0.6). Latency is a mean requirement (≤ 0.6×), with p50 and p90 reported. |
| N2 | Accepted | SE_b is now the paired SE of (hub-base − 27B) at avg@16: provisional in G0 from (base − 27B), fixed at G1 before any GCAM gate run. The power claim is stated for a system whose per-item profile differs from the 27B's about as much as hub-base's does; the 27B's self-SE is reported as the resolution floor. |
| N3 | Accepted, implemented | `compile_program` now bounds main + **every** variant + reserve (each variant runs at most once), with a two-variant chain test reproducing the reviewer's case. Irreversible nodes (external_send, destructive) compile only with a direct `meta.confirm` / `tool.dry_run` dependency (`Program.require_confirmation`, default on; tested). The plan marks what is still prose: runtime wiring of `cap_input` and recompile remaining-budget accounting (P1). |
| N4 | Accepted | Roadmap reordered to obey the stage edges: G1 runs T1 → hub-base → re-atlas (weeks ~4–5) → Artifact A from that atlas. The edge now says T4/T5 v1 may run on hub-base in G2 to test recipes and are re-run after T3 in G3; only post-T3 versions ship. |
| N5 | Accepted | Goal-suite v0: ≥ 100 goals at G0 exit, ≥ 250 by week 6 (≥ 200 development, ≥ 50 held out). The G1 audit runs on the ≥ 200 development goals, one run per goal; "task" is defined as a distinct goal or item. |
| N6 | Accepted | Spike: trains on public terminal-task sets (in-house environments are not ready before week 14); evaluates on ≥ 300 held-out terminal tasks with k = 4 and a paired item bootstrap, n fixed in the pre-registration from a G0 estimate of the per-task difference SD so that SE ≤ 1.4 points. Pass = gain ≥ +4 with CI lower bound > 0; criterion (a) = CI upper bound < +4; otherwise inconclusive, and the decision rests on criterion (b) alone. |
| N7 | Accepted | Cost under Artifact A is measured at fleet level; utilization u divides r. |
| N8 | Accepted | T3 Tier 1 is "≈ 26K" in both places; under Tier 0, T6 and certification after week 30 run on the evaluation node. |
| N9 | Accepted | Plan says `cap_input` and `Receipt.cap_hits` exist and runtime wiring is pending. Pilot 07 amendment 2 adds the 4-token re-tokenization slack to `s1.probe` and states that the pilot's head-only text truncation does not exercise `cap_input`. |
| N10 | Accepted, implemented | Partial calibration records for an op raise `CompileError` (records are all-or-none per op); per-assignment expected tokens are clamped at 0. Tested. |
| N11 | Accepted | R26 rewritten: merged family checkpoints are the certification reference; B is certified with goal bias on/off; adapter-mix fuzzing only if adapter serving is ever introduced. P1 exit criterion and conformance matrix updated; P5 shares the Part I allocation; "Hub (v5)" → "Hub". |
| N12 | Accepted | The round-4 response was wrong: wave-1 note 02 records `qwen-community-1.0`. The plan cites it and keeps "terms to be checked before any use". |
| N13 | Accepted | Part III 01 row: "no" on 13/15, "yes" on 1, one unparseable. |
| N14 | Accepted, implemented | Pilot 06 amendment 2: Bonferroni-adjusted 97.5% CIs for the two primary comparisons (script computes both 95% and 97.5%); controls get the same comparison and a fixed interpretation rule (random_parent clearing → "new capacity helps", not "the parent matters"; router_only clearing → routing alone explains it). |
| N15 | Noted | The 03 snapshot stays uncited until the run completes. |

The architecture blueprint (`docs/blueprint.html`) is updated to v6.1 alongside the plan.

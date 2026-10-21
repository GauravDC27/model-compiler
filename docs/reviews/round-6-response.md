# Round 6 — author response to Fable 5.1 review

**Plan version reviewed:** v6.1 · **Revised to:** v6.2 · **Date:** 2026-10-21

The reviewer approved v6.1 with changes and judged it structurally converged: no blocker, three major
gate-definition findings (P1–P3), four minor (P4–P7). All are accepted. Code changes are tested (83 tests
pass). The blueprint (`docs/blueprint.html`) and its PDF (`docs/GCAM_Blueprint.pdf`, built by
`docs/pdf/build_blueprint_pdf.py`) are updated with the plan.

| ID | Disposition | Change |
|---|---|---|
| P1 | Accepted | Spike gate: pass = CI lower bound > 0 (≈ 80% power at a true +4 with SE 1.4); the point estimate is reported, not required to be ≥ +4. k = 8 samples per task at n ≥ 300 (k = 4 gives SE ≈ 2.0 on the sampling term alone). If too few disjoint tasks exist, more public sets are pooled; failing that, the wider inconclusive band is stated in the pre-registration. Sheets 07 and 10 carry the same text. |
| P2 | Accepted | δ_b is capped at 4 points. A benchmark whose 2.8·SE_b exceeds the cap is reported as resolution-limited and pooled into an item-weighted reasoning composite under the same rule; if the composite also exceeds the cap it is reported, not gated. Expected sizes are written down: AIME pooled ≈ 9–12 and GPQA-Diamond ≈ 6–8 (pooled), LiveCodeBench v6 ≈ 2–3, HLE and MMLU-Pro at the floor. The blueprint's G0-1 row now states the rule the plan contains. |
| P3 | Accepted | Criterion (b) is measured on the spike checkpoint and written as the formula r·τ_h + e > 0.6 at measured r and τ_h (0.20 is the planning value). Rule: switch if (a); switch if the spike is inconclusive and (b) holds; if the spike passes and (b) holds, stay and re-evaluate (b) on the T3 checkpoint at week 17. A hub in the (0.15, 0.20] band fails the G3 gate, is reported against the re-derived target, and does not switch. Roadmap rows updated. |
| P4 | Accepted, implemented | Guards may now target `abstain`. A confirmation gates only if it carries a guard that abstains or falls back to a variant with no irreversible node; otherwise the program does not compile (tested, including a fallback variant that would re-send). The partial-record check covers every calibration item, not only the first (tested). |
| P5 | Accepted | Item (i) adapters for Artifact A get their own line: first three families, ≈ 15–300 GPU-h each, weeks 5–8 after the G1 re-atlas, inside both tiers' slack; on Tier 0 the spike runs first. |
| P6 | Accepted | Goal-suite cost is reported per family, and the plan says the ≤ 0.5× goal-suite target relies on System-1 and code taking nodes off the hub on agentic and research goals. Goals come mostly from templated generators with executable checks; that is what makes ≈ 8 verified goals a day plausible. |
| P7 | Accepted | Blueprint: footer and header at v6.2; T3 "≈ 26K"; Sheet 02 now has nine nodes, with the confirm node n8 and its abstaining guard before n9 sends; Sheet 05 claim distinguishes goal-off identity (by construction) from the −∞-bias path (a serving test); Sheet 08 formula includes s. |

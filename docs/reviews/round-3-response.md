# Round 3 — author response to Fable 5.1 review

**Plan version reviewed:** v3 · **Revised to:** v4 · **Date:** 2026-10-21

The reviewer judged the architecture converged and asked for pilot-hygiene changes before the
pilots run. All are accepted; every pilot change is recorded as a dated pre-registration
amendment written before any run (`03_route_pinning_prereg.md` amendment 1,
`04_margin_regularization_prereg.md` amendment 2).

| ID | Disposition | Change |
|---|---|---|
| K1 | Accepted | 03 records each stack's own fresh margin and fragile share; the primary D2-fresh curve ranks by the executing stack's margin, REF-ranked curve secondary; pinned curves rank by REF's margin. PLAN §6 item 2 states the definition. |
| K2 | Accepted | New `experiments/05_proxy_calibration.py` (75 prompts, hardened routers): router- and output-logit Δ distributions with/without the first position, per-position KL, flip rate, JSON artifact; regime rule (faithful iff mean KL ≤ 10 × 1.5e-3) and ε for H4 = measured p95 router-logit Δ, both fixed in the amendments. It runs before 03/04. |
| K3 | Accepted | `DecideProfile` now marginalizes by default over case and leading-space surface forms (the granite tokenizer gives `no`, `No`, ` no`, ` No`, `NO` distinct ids); both pre-registrations name this scorer. |
| K4 | Accepted | `HFBackend(harden=True)` by default, permanent for the backend's life, recorded as `router` in `lock_fields()`. 02 was already running with the old default; it is relabelled in §10 (FP32 rows R7-compliant, BF16 rows R7-violating) rather than killed after ~50 minutes of CPU time. |
| K5 | Accepted | H1 is a paired repaired-vs-introduced sign test (uninformative if < 5 fresh flips); H4 has three pre-registered branches (support / demote / inconclusive → hub-scale retest); content-only fragile share and ε = 0.01 are secondaries; H4 at n ≈ 300 is labelled exploratory. R24 in the plan carries the three branches. |
| K6 | Accepted | Pinning failures are caught per item and counted as pinned disagreements; intermediate JSON is written after each part. |
| K7 | Accepted | The informal 10-prompt calibration row is relabelled "informal; no artifact; superseded by 05". |
| K8 | Accepted, implemented | `dmc/runtime.py::template_clock` binds `strftime_now` and `date_string` to `Context.now` (fixed epoch when absent); verified on the granite template. `DecisionResult.rendered_prompt_hash` is written into `Receipt.rendered_prompt_hash` by the compiler. Tests: `tests/test_runtime.py` (template cannot read the host clock), `tests/test_compiler.py`. |
| K9 | Accepted | §4.2: P0 must run batch-invariance validation on the pinned Instruct-2507 checkpoint itself and re-profile its super experts. |
| K10 | Accepted | Numbers corrected: 13 "no" + 1 "yes" + 1 unparseable; NMI ratio 5.0–7.6×; abstain-band claim now reports 8/10 correct and CCR 4/5. |
| K11 | Accepted | 03 reports REF single-pass vs incremental-decode agreement as a baseline. |
| K12 | Accepted | §0.3 carries the stock-router caveat inline. |
| K13 | Accepted | `dmc/http_backend.py` is committed, listed in §9, with the echo-log-prob caveat. |
| K14 | Accepted | Amendment 2 says the oracle-accuracy bound is a guard, not a test of "no capability loss". |
| K15 | Accepted | `tests/test_plan_citations.py` now also checks `Receipt.*` / `Context.*` attribute claims and the twelve-op count. (It immediately caught a premature citation of the 05 results file in §0.3, fixed before commit.) |

**Pilot-size cuts adopted (within the reviewer's acceptable list):** 03: 15 oracle groups per family (300 decisions), no free-running pinned STRESS. 04: 4 grid arms + control, 45 probe prompts + 16 held-out chunks, 90 oracle-agreement decisions, 30 secondary parity decisions. Nothing from the "not acceptable" list was cut.

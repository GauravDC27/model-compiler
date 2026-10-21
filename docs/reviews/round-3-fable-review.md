# Adversarial review, round 3 — `docs/PLAN.md` v3 (DMC)

**Reviewer:** Fable 5.1 · **Date:** 2026-10-21 · **Materials:** PLAN.md v3; `docs/reviews/round-2-response.md`; every module in `dmc/` (including the untracked `dmc/http_backend.py`); `tests/` (51 passing, run by me); `experiments/01–04` with both pre-registrations; `experiments/results/01_router_atlas.json`; installed `transformers 5.20.2` router sources. Citations are `NN:line` into the research notes or `file:line` into the repo. Every G-item below was checked in code, not prose.

---

## Overall assessment

v3 has converged architecturally. The contract table, the hub/spoke decision, the fallback ladder, the coverage floors, the two fresh-execution route levers and the non-optional discovery deliverables are now consistent with each other and with the evidence, and the status claims I objected to in round 2 are true in code (I verified each). What remains is confined to the two pilots and the Phase-0 measurement path, and all of it is cheap. Three things would still make the pilots mislead if they ran today: (1) the D2-fresh risk–coverage curve ranks ALT's fresh decisions by the *reference* stack's margin, which the executing stack does not have, and ALT's own margin is never recorded; (2) both pilots score labels with `marginalize_case=False`, contradicting R17's MUST and the atlas lesson that motivated it, and the pre-registrations do not say which scorer is used, so switching afterwards would be a forking path; (3) the primary proxy's noise magnitude is unexplained ("max |Δlogit| 3.8–4.8" from a thread-count change) and has no artifact, while ε = 0.01 for the fragile share is fixed rather than calibrated to that noise, so H4 may be testing the wrong threshold. Separately, the deterministic CPU backend does not enforce R7 by default and `02_cpu_conformance.py` — listed as "running" — uses stock routers, so its numbers will be R7-violating like the ones §10 already caveats. None of these is an architecture problem. Fix them, amend the pre-registrations with dated notes, and I would approve this plan regardless of how the pilots come out, because every outcome already has a pre-registered consequence.

---

## 1. Round-2 findings: resolution status (verified in code)

| ID | Status | Reason |
|---|---|---|
| G1 | **RESOLVED** | `RouteTracer` records token ids via an input-embedding pre-hook (`routing.py:120–137,152`); `RoutePinner` is strict by default and raises `TraceMismatch`/`TraceOverrun` (`routing.py:215–238`); 03 does teacher-forced per-position agreement, free-running only to first divergence, risk–coverage curves, 390 decisions with Wilson CIs. Residuals are new findings K1, K6. |
| G2 | **RESOLVED** | Amendment 1 has the λ = 0 control, a 3×2 grid, second-seed replication, sole primary = BF16 4-vs-3-thread flip rate ≤ ½ control, CE/entropy/agreement/accuracy guards (`04:161–172`). Residual: K3 (scorer). |
| G3 | **RESOLVED** | `Context.session_state_hash` (`ir.py:38`), `CreateText.parent_seed` (`ir.py:118`), twelve op types (`ir.py:134–139`), `Receipt` fields (`store.py:19–38`), `DISPATCH_TABLE` (`compiler.py:72–79`), `experiments/results/01_router_atlas.json` exists, `tests/test_plan_citations.py` guards citations. One field is still write-less (K8). |
| G4 | **RESOLVED** | Create spoke is D0-only pending a GDN audit under the Qwen3.6 go/no-go; plain-attention writer in the P3 bake-off. |
| G5 | **PARTIAL** | `harden_routers` (FP32 GEMM + lowest-index ties) exists and is tested; BF16-vs-BF16 thread variation is the primary proxy in 03/04. But `HFBackend` does not apply it (R7 is opt-in), `02_cpu_conformance.py` never calls it, the calibration behind "~4.5% / Δlogit ~4" has no script or JSON, and the magnitude is unexplained (K2, K4, K7). |
| G6 | **RESOLVED** | 384 B/token stated; external-route fused-MoE entry point is a P1 item; D2-replay certified on the HF stack meanwhile. |
| G7 | **RESOLVED** (plan) | R24/ND2 is the incremental test with a CI; demotion path defined. The pilot's consequence logic needs an "inconclusive" branch (K5). |
| G8 | **RESOLVED** | Floors pre-registered in §6 (Parse 0.90, Decide 0.85, Call 0.90, Judge 0.80, Reason 0.80); 03 uses C₀ = 0.85. |
| G9 | **RESOLVED** | §0.6 reworded correctly. |
| G10 | **RESOLVED** | Covering array + full product on ~20 prompts; unique@1000 base config only. |
| G11 | **RESOLVED** | R7 cites the V4-Flash FP32 gate GEMM and draft PR #54955 separately. |
| G12 | **PARTIAL** | Template-specific mechanism stated and the date-shift test kept; but `render_chat` does not bind `date_string`/`strftime_now` to `Context.now` (`runtime.py:59–66`), and `Receipt.rendered_prompt_hash` has no writer (K8). |
| G13 | **PARTIAL** | R17 text correct; `DecideProfile(marginalize_case=True)` exists (`decide.py:62–80`), but the default is `False`, both pilots use the default, and length normalization is not implemented (K3). |
| G14 | **RESOLVED** (plan) | Hub = Qwen3-30B-A3B-Instruct-2507, revision pinned. Nuance: that exact checkpoint is not on vLLM's tested list (K9). |
| G15 | **RESOLVED** | P6 conditions the analyzer extension on its release. |
| G16 | **RESOLVED** | R9 says truncation is not implemented. |
| G17 | **RESOLVED** | `DISPATCH_TABLE["research"] == "reason"`, tested. |
| G18 | **RESOLVED** | BF16 own-trace identity, strict mismatch/overrun and tie-break tests exist (`tests/test_routing.py`). |
| G19 | **RESOLVED** | 00 probe labelled superseded/R7-violating in §10. |

---

## 2. Findings

### BLOCKER (for running the pilots; not for the architecture)

#### K1 — The D2-fresh risk–coverage curve is ranked by the reference stack's margin, which a fresh execution does not have
**Where:** `experiments/03_route_pinning.py:51–54,105–117,128–133`; `03_route_pinning_prereg.md` measure 2 ("abstaining on REF's decision margin").
**What is wrong.** `at_coverage` sorts rows by `r["margin"]`, which is `ref.margin` (`03:105`). The row never stores `fresh.margin` for ALT or STRESS. In production, D2-fresh abstention is decided by the *executing* stack from its own scores; the reference's margin is unavailable. Ranking by REF's margin tells you which items sit near REF's boundary, which is a different (and for parity-at-C₀, optimistic) selection than ranking by ALT's own margin. The curve as coded is "parity of ALT at REF-chosen coverage", not "D2-fresh at C₀" as §6 defines it. The pinned curve, by contrast, legitimately uses REF's margin (replay carries REF's receipt).
**Fix (before running).** Record `fresh.margin` per stack; report fresh curves ranked by the executing stack's margin (primary) and by REF's margin (secondary, for comparison); amend the pre-registration accordingly. Also record ALT's own fragile share so H4 can be stated for the executing stack (see K5).

#### K2 — The primary proxy's noise is unexplained, unsourced, and ε is not calibrated to it
**Where:** PLAN §0.3, R1, §10 "Proxy calibration"; `03:31` (`EPS = 0.01`); `04_margin_regularization_prereg.md` (primary metric).
**What is wrong.**
1. "BF16 at 4 vs 3 CPU threads moves logits by up to ~4" is a very large number for a reduction-order change: published cross-engine disagreement on Qwen3-30B-A3B is KL ≈ 1.5e-3 (02:148); the atlas's own FP32 batch-vs-alone router-logit drift is 3.4e-5 (`01_router_atlas.json: C_batch_router_logit_maxdiff_max`). A max |Δ| of ~4 with only ~4.5% route flips (median margin 0.087) means the Δ distribution is heavily outlier-driven (plausibly the sink/first token with massive activations, 03:133–146) — or that the no-native-BF16 emulation on this CPU rounds *partial sums* to BF16, which would make this proxy far noisier than a GPU BF16 stack (FP32 accumulate, one rounding at output). Either way, the plan cannot currently say whether ALT is a faithful proxy or a stress test, and both pilots' primary metrics depend on it.
2. There is no script and no JSON for the calibration; §10 cites numbers that exist only in prose (the only `experiments/results/` file is the atlas).
3. ε = 0.01 is fixed by fiat. R24 says ε is "calibrated to measured cross-stack noise", and the atlas gives the tool (margin→flip AUC 0.87), but the fragile share in H4 uses 0.01 regardless of what the REF-vs-ALT router-logit Δ distribution is. If that Δ has p95 ≈ 0.05, a 0.01 threshold measures the wrong thing and H4's negative result would be uninformative.
**Fix (before running).** Add `experiments/05_proxy_calibration.py` that, on the 75 probe prompts with hardened routers, reports for REF-vs-ALT (and REF-vs-STRESS): the distribution of |Δ router logit| and |Δ output logit| (median, p95, p99, max; with and without the first token), next-token KL per position (median/p95), and route-flip rate — and writes JSON. State in both pre-registrations which regime the proxy is in (faithful: KL within ~10× of 1.5e-3; stress otherwise). Pre-register ε for H4 as the p95 of |Δ router logit| on the calibration set, with 0.01 as a secondary.

#### K3 — The pilots use the scorer R17 forbids, and the pre-registrations do not say which scorer they use
**Where:** `03:101,109`, `04:97,101,104` (`DecideProfile(b, abstain_margin=0.0)` → `marginalize_case=False`); R17 ("MUST … marginalized over its surface forms"); `03_route_pinning_prereg.md`, `04_margin_regularization_prereg.md` (silent on the scorer).
**What is wrong.** The atlas found that scoring only one casing misreads a model that prefers another, and v3 made case marginalization a MUST. The pilots then run the un-marginalized scorer. Decisions, margins, parity and the H4 signal all change with the scorer, so choosing it after seeing results is exactly the forking path the pre-registrations were meant to close. Also, `surface_forms` covers case only; the leading-space variant (" no" vs "no" after `assistant\n`) is the other verbalizer axis and is not handled.
**Fix (before running).** Pre-register the scorer (recommended: `marginalize_case=True`, matching R17) and add the space-prefixed forms to `surface_forms` if the tokenizer distinguishes them (one line; test it on the granite tokenizer). Make `marginalize_case=True` the profile default so the compiler slice and the pilots agree with R17.

### MAJOR

#### K4 — The deterministic CPU backend does not enforce R7, and `02_cpu_conformance.py` ("running") uses stock routers
**Where:** `dmc/runtime.py:41–57` (no `harden_routers`); `experiments/02_cpu_conformance.py` (no reference to `harden_routers`); PLAN §9 ("deterministic CPU backend"), §10 row "02 CPU conformance | running".
**What is wrong.** R7 is a MUST for the deterministic lane; `HFBackend` is described as that lane on CPU, yet it computes the router GEMM in the model dtype unless a caller wraps it. 02's D1/D2 numbers will therefore carry BF16 router rounding exactly like the 00/01 numbers §10 now caveats. `lock_fields()` does not record whether hardening is on, so the lock cannot distinguish the two.
**Fix.** `HFBackend(..., harden=True)` applies `harden_routers` permanently at construction (not as a context manager) and records `"router": "fp32-gemm,lowest-index"` in `lock_fields()`; re-run 02 hardened (or label its results R7-violating in §10 like 00/01).

#### K5 — H1 and H4 as stated cannot distinguish "false" from "underpowered", and the consequences are attached to the wrong branch
**Where:** `03_route_pinning_prereg.md` H1, H4; `03:123–134,147–181`; PLAN R24, ND2.
**What is wrong.**
- *H1 ("pinned parity ≥ fresh parity") is unfalsifiable at equality.* If ALT flips 0–3 decisions out of 390, pinned = fresh = ~100% satisfies H1 with no information about the routing amplifier. The informative statistic is **paired**: among items where fresh ≠ REF, how many does pinning repair, and how many new disagreements does pinning introduce (exact McNemar / sign test). Two marginal Wilson CIs at n = 390 cannot resolve a 1–2 pt difference; a paired count can. Pre-register: H1 supported iff repaired > introduced with a one-sided exact p < 0.05; "uninformative" iff fresh flips < 5.
- *H4's consequence ("if false, ND2 is downgraded") treats an inconclusive CI as false.* Half B has ~195 decisions, ~166 kept at C₀; parity differences of 1–2 items are inside the bootstrap CI almost regardless of truth. ND2 should be demoted only if the CI's *upper* bound is below a minimal useful effect (e.g. +0.5 pt), kept as a MUST only if the lower bound exceeds 0, and otherwise marked "inconclusive at this n; confirmatory test at hub scale with n ≥ 2,000". The plan's R24 already allows the diagnostic outcome; add the third branch.
- *Fragile share is diluted.* It is averaged over every routing decision in every scoring pass, including the identical system/template prefix, so between-item variance comes only from the few question tokens. Pre-register a secondary computed over content and label positions only (the tracer already has per-call margins; keep the token offsets).
**Fix.** Amend the pre-registration with the paired H1 statistic, the three-branch H4 consequence, and the content-only fragile share.

#### K6 — A strict-mode failure or a late crash loses the whole run; code and pre-registration disagree on what happens
**Where:** `03:113–115` (`assert pin.status == "full"` after a strict pinner that raises first), `03:228–240` (single write at the end); pre-registration measure 1 ("any TraceMismatch/TraceOverrun is reported as a failure, never silently routed live").
**What is wrong.** The pre-registration says a mismatch is *reported as a failure*; the code *raises* and aborts, after hours of BF16-emulated forwards, discarding Parts 1–2. The same applies to any exception in Part 3.
**Fix.** Catch `TraceMismatch`/`TraceOverrun` per item, record `pinned_status` and count it as a pinned failure; write intermediate JSON after Parts 4, 1–2 and 3.

#### K7 — §10 cites measurements that have no artifact
**Where:** PLAN §10 "Proxy calibration (treatment-blind, 10 prompts) | done"; R1; §0.3.
**What is wrong.** The only file in `experiments/results/` is `01_router_atlas.json`. "done" with numbers and no script/JSON is the pattern G3 was about. (Overlaps K2; listed separately because it is a status-honesty item, not a design item.)
**Fix.** Script + JSON (K2), or relabel the row "informal; to be replaced by 05".

#### K8 — Two R13a claims describe code that is not there
**Where:** R13a ("date hook … bound to `Context.now`"; "rendered prompt bytes are hashed into the receipt (`Receipt.rendered_prompt_hash`)"); `runtime.py:59–66`; grep: no writer of `rendered_prompt_hash` outside `store.py`.
**What is wrong.** `render_chat` passes no `date_string`; nothing populates `rendered_prompt_hash`; the "rendered-prompt hash stable" acceptance test does not exist. §3 step 9's honesty note lists what Phase 0 populates and does not include this field, but R13a presents it as done.
**Fix.** Have the profile return the rendered prompt (or its hash) so the compiler can set the field; pass `date_string=context.now` where the template supports it; add the date-shift test to `tests/`. Or mark both as P2 in R13a.

#### K9 — The pinned hub checkpoint is not the one on vLLM's tested list
**Where:** §0.6, §4.2 ("Qwen3-30B-A3B-Instruct-2507"); 01:152 (tested: `Qwen/Qwen3-30B-A3B`, `Qwen3-30B-A3B-Thinking-2507-FP8`).
**What is wrong.** Batch-invariance validation is per checkpoint on vLLM's list; Instruct-2507 BF16 is the same architecture but not listed. R3 data and the super-expert list are for the original/thinking checkpoints (02:141, 03:140). The plan says "re-validation" for SAEs but presents the BI validation as an asset of the chosen checkpoint.
**Fix.** State that P0 must run the BI validation on Instruct-2507 itself (one line in §4.2 and the P0 exit gate), and that super-expert re-profiling is required (already said for spokes; say it for the hub).

### MINOR

#### K10 — Numbers that do not match the atlas JSON. R14 and §10 say greedy text answered "no" to "14 of 15" decisions; `E_decide_groups_greedy_text` shows 13 "no", 1 "other", 1 "yes". §4.1 says task NMI is "4–8×" the null; the per-layer ratio of `D_task_nmi_by_layer` to `D_null_nmi_mean_by_layer` is 5.0–7.6×. §10's "abstain band of ~0.6 gives PIR 1.0 at 67% coverage" is correct but, per R14's own rule, should be paired with accuracy at that band (8/10 decided items correct; CCR 4/5 groups).

#### K11 — Free D1 measurement left on the table. 03 compares ALT's single-pass teacher-forced logits with REF's *incremental* generation; REF's own single-pass vs incremental agreement is the CPU analogue of the prefill-vs-decode leak Vosti found (01:35) and costs one FP32 pass per prompt. Add it as a baseline so `teacher_forced_agreement_fresh` is interpretable.

#### K12 — §0.3 presents "FP32-vs-BF16 flips 9.0% … 79% of tokens" without the stock-router caveat that §10 carries; add "(stock router, R7-violating)" inline.

#### K13 — `dmc/http_backend.py` (OpenAI-compatible scoring via `echo` log-probs, for GPU Phase 0) is untracked, tested, and absent from §9. Add it, and note that its `score` depends on the server returning echoed prompt log-probs under the batch-invariant flag — verify on vLLM before relying on it.

#### K14 — 04's `oracle_accuracy_within_2pts` guard on 126 decisions is a 2.5-item tolerance; fine as a guard, but say "guard, not a test" in the pre-registration so a 2-point drop is not reported as "no capability loss".

#### K15 — `test_plan_citations.py` checks `file::symbol` existence but not attribute claims (`Receipt.rendered_prompt_hash`) or count claims ("twelve op types"); add `Program`-level assertions for the two the plan makes.

---

## 3. Pilot-size cuts: what is acceptable (for the dated amendments)

The BF16-emulated forwards are the cost. FP32 (STRESS, 04 training, 04 FP32 decisions) is cheap on this CPU, so cutting STRESS saves little.

**03 — acceptable:** oracle groups 20 → 15 per family (270 + 30 probe = 300 decisions) *provided* H1 uses the paired statistic (K5); drop the free-running pinned run for STRESS only; keep all 25 generations (teacher-forced is one pass per stack). **Not acceptable:** dropping the ALT pinned arm, the 30 probe decisions, Part 4 (stock vs hardened; one forward per prompt), or the 1,000-sample bootstrap (CPU-trivial). H4 at n ≈ 300 is exploratory either way; say so (K5).

**04 — acceptable:** grid 6 → 4 arms (γ ∈ {0.05, 0.2} × λ ∈ {1, 5}); the secondary BF16 thread-parity on 60 oracle decisions → 30 or dropped (it is informational); flip-rate eval inputs 75 probe prompts → 45 (15 per family) plus the 16 held-out chunks (still ~10⁵ routing decisions); oracle agreement set 126 → 90 (5 groups per family). **Not acceptable:** removing the λ = 0 control, the seed-1 replication of survivors, any of the five guards, cutting steps for treatment arms but not the control (keep 80 for all or lower all together), or fewer than 16 held-out chunks for the CE guard.

---

## 4. Has the plan converged?

Yes, on everything that is not a pilot. Independent of pilot outcomes, the only things that would stop me approving are K1–K4 and the calibration artifact (K2/K7): together they are under a day of work and change no architecture. Once they are in, each pilot outcome already has a pre-registered consequence (lever removed / kept; ND2 MUST / diagnostic / inconclusive-retest; H1 withdrawn), so the pilots inform the plan without being able to break it. One labelling condition to carry forward: if `05_proxy_calibration` shows the CPU BF16 emulation is a stress regime (KL ≫ 10× the published cross-engine 1.5e-3), the CPU pilots may only be used to *remove* levers and to validate mechanisms, never to set production C₀ or to claim a lever works — the hub-scale S5 replication remains the real test, which §7 already says.

---

## 5. What v3 gets right — do not change

- D2-fresh / D2-replay with D2-replay labelled diagnostic + migration, and H1 as an explicit hypothesis.
- Pinned prefix routes and the stack-independent router as pre-registered fresh-execution levers, with the honest note that only the latter removes the amplifier.
- R26's fallback ladder with the fallback-1 acceptance cell, merged checkpoints as the certification reference, and scheduler policy in the lock.
- Pre-registered coverage floors that are product requirements, not fitted.
- R7 implemented and tested; the atlas finding that stock BF16 checkpoints violate it, and the §10 caveat.
- R14's PIR-with-CCR rule (a real lesson from the atlas: a constant answer scores PIR 0.8).
- Strict, token-verified route pinning; `NondeterminismError` on divergent recompute; `test_plan_citations.py`.
- The control arm, the grid-with-replication rule and the five guards in amendment 1; the synthetic oracle as a pure function of the seed.

---

## VERDICT: APPROVE WITH CHANGES

Minimum changes required for approval (all before the pilots run; each with a dated amendment to the pre-registration it touches):

1. **K1** — record the executing stack's own margin (and fragile share) and rank the D2-fresh curve by it; REF-ranked curve as secondary.
2. **K2 / K7** — `05_proxy_calibration.py` + JSON (Δ router-logit and Δ output-logit distributions, per-position KL, flip rate; with/without first token); declare the proxy regime; pre-register ε for H4 from the measured p95.
3. **K3** — pre-register the scorer as `marginalize_case=True`, make it the default, add space-prefixed surface forms if the tokenizer distinguishes them.
4. **K4** — `HFBackend` hardens routers by default and records it in the lock; re-run or relabel 02.
5. **K5** — paired H1 statistic; three-branch H4 consequence (support / demote / inconclusive-retest); content-only fragile share as secondary.
6. **K6** — record pinning failures instead of aborting; write intermediate results.

K8–K15 are recommended; K8 and K9 should be fixed in the next revision's text even if the code lands later.

# Round 2 — author response to Fable 5.1 review

**Plan version reviewed:** v2 · **Revised to:** v3 · **Date:** 2026-10-21

## Adjudications

- **(a) D2-fresh / D2-replay.** Accepted in full. D2-replay is relabelled as a diagnostic decomposition and a migration/audit tool, and "expected ≫ D2-fresh" is now hypothesis H1 under test in `experiments/03_route_pinning.py`. Both fresh-execution levers the reviewer derived are added to §4.4 and pre-registered: **pinned prefix routes** (template-prefix routes shipped in the lock, P1) and a **stack-independent router** (embedding-input or hashed routing, StableMoE-style, P5), the latter described as the only lever that removes the routing amplifier for fresh requests rather than abstaining around it.
- **(b) Fallback ladder.** Accepted with the acceptance test. R26 now has the fallback-1 cell ({same-adapter batch-mates 0..k} × {adapter on/off} → identical logits), merged checkpoints are named the certification reference, merged ≠ unmerged is called out (re-certify under the merged hash), the 6 × 61 GB memory arithmetic is stated, and the scheduler policy is a lock field (R1).

## Findings

| ID | Disposition | Change |
|---|---|---|
| G1 | Accepted | `RoutePinner` now records and verifies token ids (input-embedding pre-hook) and is **strict by default**: `TraceMismatch` / `TraceOverrun` instead of silent live routing; `status` = full/partial. The pilot was redesigned before it ever ran: teacher-forced per-position agreement for generation, free-running reported only up to first divergence, risk–coverage curves from the margins, 390 decisions (360 synthetic-oracle + 30 probe) with Wilson CIs. New tests: strict mismatch/overrun, teacher-forced replay of a generation trace, BF16 own-trace identity. Pre-registration: `experiments/03_route_pinning_prereg.md`. |
| G2 | Accepted | Pre-registration amended before any run (`04_margin_regularization_prereg.md`, amendment 1): λ = 0 control; γ × λ grid declared; second-seed replication rule; flip rate is the sole primary (OR clause removed); expert-load entropy guard; Decide-agreement and oracle-accuracy bounds inside PASS; deterministic algorithms off during training stated; "ce" labelled as pure CE (aux coef 0.0). |
| G3 | Accepted | Made the claims true in code: `Context.session_state_hash` and context minimization, `CreateText.parent_seed`, all twelve op types, `Receipt` fields (session, rendered-prompt hash, coverage band, trace ref, pinned, sampling mask, normalization), `DISPATCH_TABLE`. `experiments/results/` exists (the atlas finished after the reviewer's read). Status labels in §7/§10 now distinguish done / running / pending. New `tests/test_plan_citations.py` fails if PLAN.md cites a nonexistent `dmc/...` file or symbol. |
| G4 | Accepted | Create spoke on Qwen3.8-27B is **D0-only** until a GDN batch-invariance audit passes under the Qwen3.6 go/no-go; P3 bake-off adds a plain-attention writer for a D1 path. |
| G5 | Accepted | `harden_routers` (FP32 router GEMM + lowest-index tie-break) implements R7 for HF models. Treatment-blind calibration: with hardened routers, BF16 at 4 vs 3 CPU threads flips ~4.5% of routes and moves logits by up to ~4, so **BF16-vs-BF16 reduction-order variation is the primary proxy** in both pilots; FP32-vs-BF16 is the stress bound. Stock-vs-hardened flip rates are measured (03, part 4). The 00 probe and 01 numbers are labelled R7-violating in §10. |
| G6 | Accepted | Trace size corrected to 384 B/token; external-route fused-MoE entry point added as a P1 engineering item; until it lands, D2-replay is certified on the HF reference stack. |
| G7 | Accepted | ND2/R24 acceptance is the incremental test (combined signal vs margin-only at C₀, CI excluding zero); if it fails, ND2 is a diagnostic. Measured in 03 (H4). |
| G8 | Accepted | Coverage floors pre-registered as product requirements: Parse 0.90, Decide 0.85, Call 0.90, Judge 0.80, Reason 0.80; reaching parity only below the floor is a gate failure. |
| G9 | Accepted | §0.6 reworded: the hub is the only tested MoE with all five discovery assets. |
| G10 | Accepted | Pairwise covering array + full product on ~20 prompts; unique@1000 only in base config; unique@20–50 per cell. |
| G11 | Accepted | R7 cites the V4-Flash FP32 gate GEMM and draft PR #54955 separately. |
| G12 | Accepted | R13a: template-specific mechanism stated; date hooks bound to `Context.now`; rendered prompt bytes hashed into the receipt. |
| G13 | Accepted | R17: single-token labels or length normalization; case marginalization added (from the atlas finding). |
| G14 | Accepted | Hub = Qwen3-30B-A3B-Instruct-2507, HF revision pinned in the lock. |
| G15 | Accepted | P6 says "extend Vosti's analyzer if released, otherwise budget a re-implementation". |
| G16 | Accepted | R9 states that truncation is not implemented yet. |
| G17 | Accepted | `dmc/compiler.py::DISPATCH_TABLE` maps `research` and `analyze` to Reason, matching §3. |
| G18 | Accepted | BF16 own-trace identity and strict overrun/mismatch tests added (`tests/test_routing.py`, 12 tests). |
| G19 | Accepted | The 00 probe note is superseded and labelled R7-violating in §10. |

## New evidence since round 2

- **Router atlas** (01, 75 prompts): see PLAN §10. Two results changed the plan beyond the review: (1) PIR must be reported with accuracy and the consistent-correct rate (a constant answer scored PIR 0.8); (2) label scoring must marginalize over surface forms (case), now in R17 and `DecideProfile(marginalize_case=True)`.
- **Proxy calibration** (hardened routers, BF16 thread variation): basis for the G5 proxy choice.
- **Task specificity** is real but weak (NMI ≤ 0.04 vs null p95 ≤ 0.0095), which lowers the prior on clean per-task expert masks (§4.1).

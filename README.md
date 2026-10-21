# model-compiler

This project aims at creating high determinism with probabilistic models using various approaches, so that models become as reliable as compilers in spirit: **same prompt + same toolchain → same output; equivalent prompts → equivalent output; ill-formed prompts → a defined error, never a guess.**

## Where things are

| Path | What |
|---|---|
| `docs/PLAN.md` | The architecture requirements and implementation plan (determinism contract D0–D4/D⊥, 27 output-parity requirements, compiler pipeline, Deterministic MoE hub + spokes, discovery program, six profiles, roadmap). |
| `docs/reviews/` | Adversarial review rounds (Fable 5.1) and the author's responses. |
| `research_notes/Deterministic MoE model compiler/` | Seven cited research notes (state of the art as of Oct 2026). |
| `reports/` | Synthesized research report. |
| `dmc/` | Phase-0 reference implementation (below). |
| `experiments/` | CPU experiments on a real small MoE (granite-3.1-1b-a400m); results in `experiments/results/`. |
| `tests/` | Unit tests. |

## `dmc` — Deterministic Model Compiler (Phase 0)

| Module | Role |
|---|---|
| `canonical.py` | RFC 8785-style canonical JSON and content hashes |
| `lockfile.py` | Toolchain identity: everything that can change output bits |
| `ir.py` | Typed IR v0: discriminated union of ops, required slots, ambiguity channel |
| `compiler.py` | Pipeline: normalize → parse → canonicalize → static dispatch → execute → store, with receipts |
| `store.py` | Content-addressed result store; replay; a divergent recompute is an error |
| `runtime.py` | Backends: deterministic CPU backend (explicit system turn, fixed shape, counter-based sampling), table backend for tests |
| `profiles/decide.py` | Decide profile: order-averaged label scoring, calibrated confidence, abstention band |
| `metrics.py` | TARr/TARa@N, unique@N, first divergence, paraphrase invariance, flips, ECE |
| `moe_probe.py` | Router margins, flips, stable top-k, margin→flip AUC, task specificity |
| `routing.py` | Routing traces and route pinning (re-execution with recorded experts) |
| `conformance.py` | Run requests under execution variations and report parity |

## Run

```bash
python -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers safetensors pydantic pytest pandas pyarrow
.venv/bin/python -m pytest -q                      # unit tests
.venv/bin/python experiments/01_router_atlas.py    # router atlas on granite-3.1-1b-a400m (CPU, ~15 min)
```

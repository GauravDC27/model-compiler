# GCAM v1: the first working Goal-Compiled Agentic MoE

v1 implements Part I of `docs/PLAN.md` end to end on CPU. It covers the whole path: a goal in natural language is compiled into a static, budgeted, guarded program, and the program runs on an assembly of executors under the determinism contract. Every part of the architecture is exercised by at least one goal family:

- compiled code;
- classic ML;
- System-1 encoder and hub-state heads;
- specialized layers inside the MoE hub (goal-gated experts and hot-expert LoRA);
- a specialist spoke model;
- cascades, best-of-K selection and two-order judging;
- confirmation-gated tool use and replay.

On CPU the models are smaller stand-ins for the plan's:

| Role | Plan | v1 on CPU |
|---|---|---|
| Hub (MoE) | Qwen3.6-35B-A3B | granite-3.1-1b-a400m-instruct (32 experts, top-8, 24 layers, 0.4B active) |
| Specialist spoke | Qwen3.8-27B | Qwen2.5-1.5B-Instruct (dense) |
| System-1 encoder | ModernBERT / GLiNER2 class | all-MiniLM-L6-v2 |
| Classic ML | GBDT / TabPFN / Chronos-2 | scikit-learn HistGradientBoosting, Ridge forecaster, TF-IDF |

Cost is counted in **hub-token equivalents**: tokens weighted by active parameters per token relative to the hub. The spoke weighs 3.75, the encoder 0.057, and code and classic ML 0. This is the CPU analogue of the plan's cost basis r.

## The company it works for

`gcam/domain.py` builds "Acme Ops" deterministically from a seed:

- 240 customers and 3,400 orders over 180 days, with weekly seasonality, a trend and per-category refund rates;
- 1,600 expense reports, where a memo decides the borderline cases (reports from the last 30 days, the evaluation period, word every memo differently);
- an eight-section refund and expense policy;
- generators for refund claims, support requests, reply pairs, policy questions and cost questions.

Each text generator has separate training and held-out phrasings, so every evaluation number is on out-of-template text as well as in-template text.

## Goal families and what each exercises

| Goal (example) | Family | Task graph | Parts exercised |
|---|---|---|---|
| `Customer C-… requests a refund for order A-…: "<claim>"` | refund_decision | order lookup → claim reason → refund rule | code; System-1 encoder and hub-state probe; **hub with goal-gated experts** (policy family); spoke; guard `order_found` |
| `Handle request from C-…: "<request>"` | support_request | tool selection → argument extraction → confirmation → execution | code grammar; encoder; **hub with hot-expert LoRA** (support family); spoke; capability ceiling; **confirmation that gates destructive tools**; idempotent effects |
| `Which reply to C-… about order A-… is better? A: "…" B: "…"` | compare_replies | pairwise verdict | executable verifiers first, then a two-order judge on the hub, then the spoke judge (contrastive evaluation) |
| `Write a reply to C-… about order A-… that arrived damaged` | write_reply | order → refund rule → reply | **best-of-3 on the hub with verifier selection**; guard → pre-compiled **spoke variant** |
| `Which category had the highest refund rate in the last 90 days?` | refund_analytics | rates and z-test → summary | exact analytics in code; hub summary checked against the facts; guard → **code template variant** |
| `Estimate: <cost question>` | cost_estimate | estimate | code grammar, then program-of-thought on the hub, then on the spoke (the model writes an expression, code evaluates it) |
| `Review expense reports from the last 7 days` | expense_review | one node per report (expanded at compile time) → collect | **GBDT** on the structured fields with conformal abstention, then a **System-1 memo head** for the borderline reports, then the hub, then the spoke |
| `Forecast daily orders for the next 7 days` | forecast_orders | forecast | Ridge vs seasonal-naive chosen by backtest, with intervals |
| `What does our policy say about <topic>?` | policy_question | retrieve → answer | TF-IDF, then dense encoder retrieval, then hub and spoke |

Anything else is a `Clarify` with the supported forms. So is a supported form with a missing or inconsistent slot, such as a refund request without an order number: the parser lists the customer's recent orders instead of guessing.

## How a goal runs

1. **Parse** (`gcam/parser.py`): a versioned grammar compiles the goal to a Goal IR with explicit context (time and the world snapshot hash only when the family needs them).
2. **Plan** (`gcam/planner.py`): a versioned template per family gives a task graph of typed nodes, with an accuracy bar, contract, input cap and capabilities per node, plus guards and pre-compiled variants. Plans are cached by template hash.
3. **Compile** (`dmc/scheduler.py::compile_program`):
   - for each node, the cheapest executor ladder whose accuracy, priced on the joint calibration records, meets the bar;
   - the capability ceiling from the goal's autonomy level;
   - confirmation gating for irreversible tools;
   - a worst-case token bound across the main graph, every variant and the recompile allowance;
   - a schedule hash that includes the lock hash.
4. **Run** (`gcam/runtime.py`):
   - capped inputs are truncated deterministically;
   - each ladder runs rung by rung and escalates only on calibrated abstention;
   - guards check typed outputs, and a failure selects a pre-compiled variant or abstains;
   - every node writes a receipt to the content-addressed store, so re-running a goal replays recorded bytes (D0).

## Executors

| Rung | What it is | Answers when |
|---|---|---|
| `code.*` | compiled code: lookups, the refund policy, argument extraction, the confirmation policy, tool effects, verifiers, analytics, the arithmetic grammar, keyword rules | always (exact), or the grammar matches exactly one rule |
| `ml.gbdt`, `ml.forecast`, `ml.tfidf` | classic ML | its calibrated confidence clears a threshold set for ≥ 95% answered accuracy |
| `s1.encoder` | MiniLM embeddings + logistic heads (claim reason, tool, expense memo); dense policy retrieval | same |
| `s1.hubprobe` | logistic head on the hub's layer-12 hidden states (one prefill, no generation) | same |
| `hub`, `hub.policy`, `hub.support` | the MoE hub. `hub.policy` switches on the goal-gated experts; `hub.support` switches on the hot-expert LoRA | option scoring clears its threshold; generation passes its verifier or parses |
| `spoke` | Qwen2.5-1.5B | last rung: always answers |

Decisions are read from option-letter scores averaged over two option orderings, never from sampled text (R17).

## Running it

```bash
python -m gcam train     # hub capacity (≈ 35 min on 4 CPU cores), then System-1 fit + calibration (≈ 2.5 h)
python -m gcam eval      # per-op ladders vs baselines on held-out items, end-to-end goals, replay check
python -m gcam run "Review expense reports from the last 7 days"
python -m gcam run 'Handle request from C-1222: "Cancel order A-8374."' --approve cancel_order:A-8374
GCAM_OPS=decide.expense_review python -m gcam.train.calibrate   # recalibrate only the listed ops
```

Model weights and trained artifacts live in `artifacts/gcam/` (not committed). Every number in the results below is regenerated by these commands; the evaluation output is committed as `experiments/results/gcam_v1_eval.json`.

## Results

(Filled in from `experiments/results/gcam_v1_eval.json` once the CPU runs finish.)

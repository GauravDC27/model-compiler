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

All numbers are from `experiments/results/gcam_v1_eval.json`. The hub's capacity was trained for about 36 minutes; System-1 fitting and calibration took about 2.5 hours, with partial recalibrations noted in the calibration meta. Everything ran on 4 CPU cores.

Every ladder below was compiled from the calibration records alone. The test items were generated from different seeds, and half of them use held-out phrasings. Cost is in hub-token equivalents per item.

### 1. The compiler picks cheap executors when they meet the bar

| Op (held-out items) | Bar | Compiled ladder | Predicted acc · cost | Realized acc · cost | Always spoke | Hub (base) | Hub + family layers |
|---|---|---|---|---|---|---|---|
| Claim reason (80) | 0.92 | keywords → encoder | 0.963 · 1 | **0.963 · 1** | 0.95 · 777 | 0.26 · 225 | 0.95 · 225 |
| Tool selection (80) | 0.92 | keywords → encoder | 0.988 · 1 | **0.950 · 1** | 0.825 · 959 | 0.625 · 268 | 1.00 · 268 |
| Pairwise judging (40) | 0.90 | verifiers | 1.000 · 0 | **1.000 · 0** | 0.95 · 1327 | 0.70 · 387 | — |
| Cost estimate (30) | 0.90 | arithmetic grammar → spoke PoT | 0.933 · 174 | **0.933 · 174** | 0.70 · 337 | 0.33 · 92 | — |
| Expense review (120) | 0.90 | GBDT → memo head | 0.938 · 0 | **0.917 · 0.3** | 0.33 · 1446 | 0.40 · 420 | — |
| Policy retrieval (24) | 0.85 | TF-IDF → encoder → spoke | 1.000 · 1113 | **1.000 · 1112** | 0.92 · 2956 | 0.17 · 863 | — |
| Analytics summary (7) | 0.60 * | hub | 0.875 · 180 | **1.000 · 175** | 1.00 · 651 | 1.00 · 175 | — |
| Customer reply (9) | 0.60 * | hub best-of-3 | 0.778 · 860 | **0.556 · 862** | (fallback variant) | — | — |

\* These nodes generate text and are gated by a verifier. Only verified text ships; otherwise the guard runs the pre-compiled variant (the spoke for replies, a code template for summaries) or abstains. The bar therefore prices how often the main path suffices, not whether shipped text is correct.

On every decision op the compiled ladder matches or beats "always spoke", at under 0.2% of its cost on claims, tools and expenses. It costs 52% of the spoke's on estimates and 38% on policy retrieval.

Predicted and realized accuracy agree within 4 points on every op with at least 24 test items. The two text ops have only 7 and 9 items. The summary node was priced at 0.875 and realized 1.0. The reply node was priced at 0.778 and realized 0.556, below its bar; 9 items are too few to price it. At runtime this costs money, not correctness: a reply that fails verification falls through to the spoke variant (0.778 verified on held-out items), then to abstention.

### 2. Every part is load-bearing (ablation)

The same per-op compilation, with executors removed:

| Op | Full system | LLM rungs only | LLM rungs, no goal-family layers |
|---|---|---|---|
| Claim reason | keywords → encoder · 0.963 · 1 | **hub + goal experts** · 0.95 · 225 | spoke · 0.95 · 777 |
| Tool selection | keywords → encoder · 0.950 · 1 | **hub + LoRA** · 1.00 · 268 | cannot compile (best 0.81 < 0.92) |
| Pairwise judging | verifiers · 1.00 · 0 | cannot compile (0.875 < 0.90) | cannot compile |
| Cost estimate | grammar → spoke · 0.933 · 174 | cannot compile (0.77 < 0.90) | cannot compile |
| Expense review | GBDT → memo head · 0.917 · 0.3 | cannot compile (0.51 < 0.90) | cannot compile |
| Policy retrieval | TF-IDF → encoder → spoke · 1.00 · 1112 | spoke · 0.92 · 2956 | spoke · 0.92 · 2956 |

The ablation shows what each part buys:
- **Code, classic ML and System-1** make pairwise judging, estimates and expense review compile at all, and cut costs by two to three orders of magnitude elsewhere.
- **The hub's goal-family layers** make it the cheapest LLM rung that meets the bar on its families. Without them, claims need the spoke at 3.5× the cost and tool selection cannot compile.

Refund and support goals run end to end with LLM rungs only. They reach exactly the same results as the full system. The specialized layers answer (`hub.policy` for the claim reason, `hub.support` for the tool) at 212–254 hub-token equivalents per goal. The full system pays 0–2.

### 3. Specialized layers inside the MoE hub

| Family | Mechanism | Trainable params | Base hub | Family on | Other family's layers on |
|---|---|---|---|---|---|
| Refund-claim reason | goal-gated experts (item ii): one additive expert in each of the last 8 layers, sigmoid gate from a zero router row, 40 steps | 12.6 M | 0.271 | **0.958** | 0.708 |
| Tool selection | hot-expert LoRA (item i): rank 4 on each layer's hot experts (80% of selection mass, 11–21 experts per layer), 90 steps | 6.0 M | 0.500 | **0.979** | 0.896 |

These are capacity-eval numbers: 48 items, half on held-out phrasings. The per-op test records (80 items, different seeds) agree: 0.95 and 1.00. With no family set, the hub is bit-identical to the base model, which is checked on fixed probes after training.

Each family's layers also lift the other family's task. The base hub mostly fails the option-letter format (it answers "damaged" for 47 of 48 claims), and either adaptation partly fixes that. The family-specific gain is the remainder: 0.958 vs 0.708 on claims, and 0.979 vs 0.896 on tools.

### 4. End-to-end goals

The evaluation ran 18 goals, one or more per family:

| Outcome | Goals |
|---|---|
| done | 16 |
| abstained | 1: a cancellation without approval, stopped at the confirmation gate |
| clarify | 1: a refund request without an order number |

All 18 stayed within their compiled worst-case bound. Examples:

- **Refund decision:** "The puzzle was torn when it arrived." is answered by the keyword rules as `damaged`, then the policy code denies it under P2 (reported 174 days after delivery, limit 90). Cost: 0.
- **Held-out phrasing:** "two weeks and counting, the board game is nowhere…" makes the keyword rules abstain and escalates to the encoder, which returns `not_received`. Cost: 2.
- **Support request:** "Can someone phone me tomorrow at 3pm?" maps to `schedule_callback` with the date and time resolved by code, then executes with an idempotency key. Cost: 0.
- **Cancellation:** the destructive tool runs only with the approval token `cancel_order:<order>`, and only once however often the goal is re-run.
- **Write reply:** the hub's best-of-3 writes a reply that passes every check. Cost: 882; bound: 4,226.
- **Refund analytics:** exact rates and a z-test in code (apparel 16.0% vs 6.1%, z = 6.09), then a hub summary verified against those numbers. Cost: 174.
- **Expense review (last 7 days):** 71 reports. The GBDT decides 50 and the memo head 21. Accuracy is 0.93 against ground truth; 8 reports are flagged. Cost: 21.
- **Forecast:** Ridge with weekday and trend terms beats seasonal-naive in the backtest (MAE 4.35 vs 6.43). Forecasts carry 80% intervals.

On the replay pass every node replayed from its receipt (D0). The results were identical, the replay cost was 0, and side effects were not repeated.

### 5. What did not work, and what was fixed

- **Training data confound.** The first capacity runs paired each label with one fixed option ordering (labels cycle every 4 or 6 items, orderings every 2), so the answer letter was predictable without reading the text. The goal experts learned that shortcut and scored at chance under order-averaged inference. Every item is now trained in both orderings. The routed variant's earlier failure was confounded by the same bug, so v1 makes no routed-versus-additive claim.
- **Goal experts in all 24 layers diverged** at learning rates of 1e-3 and 1e-4. The last 8 layers at 1e-4 work.
- **Missing information, not missing ability, on expenses.** The LLM rungs were at chance on borderline expense reports. The memo decides those reports, but the policy text did not say how. Policy P5 now states the rule. The hub (0.40) and the spoke (0.33) still do not apply it at this scale, so expense review uses a System-1 memo head, which is tested on memo wording it never saw.
- **Replies.** A reply that hit the token cap used to stop mid-word and still pass. Replies are now cut to their last complete sentence and verified as shipped. The reply node needs a larger calibration set (n = 9).
- **The CPU hub is weak on open-ended work** (estimates 0.33, policy retrieval 0.17), and the 1.5B spoke carries those rungs. That is expected at 0.4B active parameters. The plan's GPU-scale hub is where the reasoning gap matters.

# GCAM vs frontier models: accuracy, cost and latency on held-out use cases

This report compares GCAM, the goal-compiled agentic MoE from `docs/GCAM_V1.md`, with three Claude models on the eight use cases GCAM serves:
- **Claude Opus 5.5** and **Claude Fable 5.1**, the most capable models available here;
- **Claude Haiku 5.5**, the cheapest current Claude, included as a cost reference.

The data, answers and scoring are all in the repository:
- `experiments/benchmark/` holds the items;
- `experiments/results/benchmark_reference/` holds the reference models' answers;
- `experiments/results/benchmark_gcam.json` holds GCAM's per-item results;
- `experiments/results/benchmark_report.json` holds the scored comparison.

## Results

### Accuracy, and GCAM's deviation from the best reference model

| Use case (n) | GCAM | Opus 5.5 | Fable 5.1 | Haiku 5.5 | GCAM − best reference (95% CI) |
|---|---|---|---|---|---|
| Refund-claim reason (200) | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 (+0.000, +0.000) |
| Tool selection (200) | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 (+0.000, +0.000) |
| Reply judging (80) | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 (+0.000, +0.000) |
| Cost estimate (60) | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 (+0.000, +0.000) |
| Policy retrieval (48) | 0.979 | 1.000 | 1.000 | 1.000 | −0.021 (−0.062, +0.000) |
| Customer reply (30) | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 (+0.000, +0.000) |
| Analytics summary (15) | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 (+0.000, +0.000) |
| Expense review (149) | **0.966** | 0.919 | 0.919 | 0.906 | **+0.047** (+0.000, +0.101) |
| **All 782 items** | **0.992** | 0.985 | 0.985 | 0.982 | |

The CIs are paired item-level bootstraps. Per use case:
- **Six of eight:** GCAM matches the reference models exactly.
- **Policy retrieval:** GCAM misses one item of 48, and the interval includes zero. The query "calling off a purchase I made by accident" got the refund-window section instead of cancellations. The section head answered with low confidence because it was the ladder's last rung, so it had to answer.
- **Expense review:** GCAM is ahead. The ground truth depends on risk factors that appear only in historical data (vendor, weekend, round amounts, duplicates, tenure), not in the policy text. The reference models applied the policy as written. GCAM's GBDT learned those factors from past reports, and its memo head reads the memo the policy says decides borderline cases.

### Cost per 1,000 requests, and latency

| Use case | GCAM | Opus 5.5 | Fable 5.1 | Haiku 5.5 | GCAM p50 / p90 latency | GCAM answered by |
|---|---|---|---|---|---|---|
| Refund-claim reason | $0.011 | $0.383 | $0.957 | $0.010 | 107 ms / 329 ms | keywords 50%, System-1 agreement 46%, hub + goal experts 4% |
| Tool selection | $0.007 | $0.478 | $1.196 | $0.012 | 1 ms / 294 ms | keywords 50%, System-1 agreement 50% |
| Reply judging | $0.000 | $0.672 | $1.680 | $0.017 | < 1 ms | executable verifiers 100% |
| Cost estimate | $0.228 | $0.240 | $0.600 | $0.006 | 3.4 s / 8.9 s | arithmetic grammar 50%, spoke with a reasoning step 50% |
| Policy retrieval | $0.0002 | $1.544 | $3.859 | $0.039 | 5 ms / 7 ms | TF-IDF 15%, section head 85% |
| Customer reply | $2.113 | $2.115 | $5.581 | $0.049 | 29 s / 76 s | hub best-of-3 80%, spoke variant 20% |
| Analytics summary | $0.259 | $1.779 | $4.548 | $0.045 | 4.6 s / 4.9 s | hub 100% |
| Expense review | $0.0003 | $0.738 | $1.846 | $0.018 | 2 ms / 9 ms | GBDT 58%, memo head 42% |

Totals, weighted by item count:

| | GCAM | Opus 5.5 | Fable 5.1 | Haiku 5.5 |
|---|---|---|---|---|
| Decisions (5 use cases, 677 items): $ per 1k | **$0.005** | $0.606 | $1.514 | $0.015 |
| Generation (3 use cases, 105 items): $ per 1k | $0.771 | $0.996 | $2.587 | **$0.024** |
| All 782 items: $ per 1k | $0.108 | $0.658 | $1.658 | **$0.016** |

What this shows:
- **Decisions** (claims, tools, judging, policy, expenses) are where GCAM's design pays off. Accuracy is equal or better, at about 1/110 of Opus 5.5's cost, 1/280 of Fable 5.1's and 1/3 of Haiku 5.5's. The median latency of under 1 ms to 107 ms sits at or below the network round trip to a hosted model, before that model computes anything.
- **Generation** (replies, summaries, estimates) runs GCAM's language models on this 4-core CPU. GCAM is cheaper than Opus 5.5 (0.77×) and Fable 5.1 (0.30×). It is 32× more expensive than Haiku 5.5 and much slower: 3–29 s median. On a GPU the hub and the spoke would be one to two orders of magnitude faster and cheaper per token, but that was not measured here.

## How the comparison was run

### Items

The benchmark has 782 items from seeds that no training, calibration or dev-test split used:
- Half of every text use case is phrased with templates no GCAM component was trained on.
- Expense rows come from the held-out period that the dev split did not use, so their memos use held-out wording.
- GCAM was run on the benchmark once, after all training and design decisions were made, and the items were not inspected before that run.

### Reference models

- **Blind, shuffled input.** Each model got the inputs only, in shuffled order under opaque ids, and no labels.
- **Same prompts as GCAM.** They saw the system instruction and prompt that GCAM's own language-model executors see, with options in canonical order. Estimates were the one exception: they were asked for the number directly.
- **Same scoring as GCAM.** Answers were scored locally with GCAM's checks: exact labels, the reply verifier, the summary fact checks, and a numeric tolerance for estimates.
- **No tools.** The models ran through the agent harness with no tools beyond reading their item file and writing their answers. They answered a whole file per call, not one API request per item.

### Cost

- **Reference models:** list price (Opus 5.5 $4/$20, Fable 5.1 $10/$50, Haiku 5.5 $0.10/$0.50 per million input/output tokens) times tokens. Tokens are counted as one request per item: prompt plus answer, measured with a BPE tokenizer. Thinking tokens are excluded, and Opus 5.5 and Fable 5.1 always think, so their costs are lower bounds.
- **GCAM:** measured wall-clock seconds per item on this container (4 vCPUs, 16 GiB) at the on-demand price of a machine that size: AWS m7i.xlarge, $0.2016/hour, us-east-1, per [DoiT](https://www.doit.com/compute/spot/us-east-1/m7i.xlarge) and [Holori](https://calculator.holori.com/aws/ec2/m7i.xlarge/us-east-1). Model loading is excluded.

### Latency

- **GCAM:** measured per item, one request at a time.
- **Reference models:** per-request latency could not be measured without API access. The network round trip from this container to the Claude API alone was 67–110 ms per fresh connection (an unauthenticated request answered with 401), before any model computation.

## What it took to reach parity: training and design changes

GCAM v1 (`docs/GCAM_V1.md`) compiled ladders against bars of 0.85–0.92. Once the reference results set the target at about 1.0, the bars moved to frontier parity:
- 0.98 for decisions where the frontier models score 1.0;
- 0.92 for expense review;
- text generation stays verifier-gated.

The changes below were made on the dev split, before GCAM's benchmark run:

| Use case | v1 on dev-test | Change | v2 on dev-test |
|---|---|---|---|
| Refund-claim reason | 0.963 at 1 tok-eq | Frontier-teacher phrasings for the System-1 heads and the hub probe; the agreement rung; the hub's goal experts retrained | **1.000** at 18 |
| Tool selection | 0.950 at 1 | Teacher phrasings; a hub-probe head for tools; the agreement rung | **1.000** at 11 |
| Cost estimate | 0.933 (no ladder reached 0.95) | The spoke writes a line or two of reasoning before its expression | **1.000** at 254 |
| Policy retrieval | 1.000 at 1,112 | A section head trained on teacher phrasings | **1.000** at 1 |
| Expense review | 0.917 | The memo-signal head (approve / review / neither, per policy P5) decides the reports the GBDT hands off | **0.950** |
| Reply judging, replies, summaries | 1.000 (verified) | none needed | 1.000 |

Tok-eq is hub-token equivalents.

### The changes in detail

- **Capability transfer from a frontier teacher.** Opus 5.5 wrote 846 training texts (`gcam/data/teacher.json`): claims, tool requests, expense memos and policy topics. It worked from the label definitions and training-split examples only. Four texts containing a held-out template's fixed wording were removed. This is a one-time cost of under $2 at list price (about 91k tokens), not a per-request cost.
- **Agreement rung (`s1.agree`).** A System-1 answer on claims and tools needs two independent views to agree: the MiniLM head and a head on the hub's own hidden states. On its own, the MiniLM head was confidently wrong on one held-out tool phrasing family, and no threshold separated it. The agreement rung scored 1.000 on dev-test.
- **Stricter abstention.** Every threshold now targets 0.99 answered accuracy (was 0.95), set on a 200-item threshold split (was 40).
- **The hub's specialized layers.**
  - The goal-gated experts and the hot-expert LoRA were retrained with the teacher phrasings.
  - In the final ladders, the hub with goal experts answers the claims on which the two System-1 views disagree: 4% of benchmark claims, all correct.
  - The hub-state probe, one prefill of the hub with no generation, is half of the agreement rung.

## Caveats

- **Easy tasks.** The domain is synthetic and its tasks are easy for frontier models, which score 1.0 on seven of eight. The result shows parity on these specialized use cases at a fraction of the cost. It does not show general capability parity, which is what the plan's GPU-scale targets are about.
- **Benchmark not fully independent of design.** The benchmark shares its held-out phrasing families with the dev split, and two design decisions were made from dev errors: the agreement rung and the estimate reasoning step. The teacher never saw evaluation items and was filtered against held-out template wording. The benchmark items were never inspected before GCAM's single run.
- **Small text-generation samples.** Replies (30) and summaries (15) have small samples. On replies, GCAM's main path (hub best-of-3) verified 24 of 30. The pre-compiled spoke variant delivered the other 6, and nothing unverified ships.
- **Reference costs are lower bounds.** Reference-model costs exclude thinking tokens. GCAM's costs are measured on CPU, which flatters neither its cost nor its latency for generation.

## Reproducing

```bash
python -m gcam.benchmark export     # blinded item files for reference models (experiments/benchmark/)
# reference models answer the item files; answers go to experiments/results/benchmark_reference/<model>.json
python -m gcam.train.capacity       # hub goal-family layers (≈ 45 min on 4 CPU cores)
python -m gcam.train.calibrate      # System-1, classic ML, thresholds, joint records (≈ 3 h)
python -m gcam.benchmark gcam       # GCAM's compiled ladders on the benchmark, per item
python -m gcam.benchmark score claude-opus-5-5 claude-fable-5-1 claude-haiku-5-5
python -m gcam.benchmark render     # the tables above
```

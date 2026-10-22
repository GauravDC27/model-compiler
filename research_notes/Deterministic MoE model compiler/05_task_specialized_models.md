# Task-Specialized Open Models & Training Recipes for Five Task Families, with Emphasis on Output Consistency/Determinism (state as of 2026-10-21)

Research conventions used in these notes:
- "Retrieved 2026-10-21" means the number was pulled directly from the live primary source (leaderboard data file, Hugging Face model card or HF API) on that date. Everything else carries the date of the source.
- Vendor model-card numbers are self-reported, and many were measured at temperature 1.0 with single or avg@3 samples. Treat them as point estimates of a stochastic process, not deterministic facts.
- Third-party aggregators (benchlm.ai, llm-stats.com, pricepertoken.com, codesota, vellum) disagreed with each other and with the primary sources. They are cited only where flagged.
- The shared web-search budget ran out partway through. Later items were verified by fetching known primary URLs (arXiv abstracts, HF cards, leaderboard data files) directly. Anything that could not be verified is listed under Gaps.

---

## Q1a. Decision models (classification / approve-deny / triage / policy choice, calibrated classifiers, generative classifiers, RM-style heads, abstention): best open models, datasets, recipes, benchmarks

### Takeaway
No single 2026 "decision-model" leaderboard exists. The practical open-weight options fall into three groups:
- **Policy-conditioned reasoning classifiers**, e.g. gpt-oss-safeguard-20b/120b, Apache-2.0.
- **Scalar-head classifiers**, i.e. reward-model-style Bradley-Terry heads such as Skywork-Reward-V2 (0.6B–8B), or distilled small LLMs with a classification head (the Kaggle LMSYS winning recipe).
- **General LLMs prompted to decide.**

The strongest documented consistency levers are: a fixed output space (scalar head or single label token), calibration fine-tuning on roughly 1k graded examples, and explicit abstention training. Reasoning-tuned models abstain about 24% less than their non-reasoning counterparts, and prompt-format changes alone can swing accuracy by up to 76 points.

### Cited Findings
**Open models usable as decision models**
- **gpt-oss-safeguard (2025-09-18 HF release).** Two sizes:
  - gpt-oss-safeguard-20b: 21B total / 3.6B active, fits in 16GB VRAM.
  - gpt-oss-safeguard-120b: 117B / 5.1B active.
  - Both are safety-reasoning classifiers that take a developer-written policy at inference ("bring your own policy"), expose the reasoning, and have configurable low/medium/high effort.
  - License is Apache-2.0. They must be used with the harmony format.
  - Source: [HF card gpt-oss-safeguard-20b](https://huggingface.co/openai/gpt-oss-safeguard-20b)
- **Reward-model-style scalar heads as classifiers: Skywork-Reward-V2 (2025-06/07).**
  - Eight Bradley-Terry RMs from 0.6B to 8B, on Qwen3 and Llama-3.x bases, trained on 26M curated preference pairs.
  - The Qwen3-0.6B variant already scores avg 70.9 across 7 RM benchmarks. The Qwen3-1.7B variant scores avg 75.2.
  - Source: [HF card Skywork-Reward-V2-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)
  - Qwen3-based variants are Apache-2.0 per the [HF API](https://huggingface.co/api/models/Skywork/Skywork-Reward-V2-Qwen3-8B) (retrieved 2026-10-21).
- **Distilled small classifier recipe (Kaggle LMSYS "Chatbot Arena Human Preference Predictions", 2024).**
  - Task: predict which of two responses humans prefer, a 3-way decision. Data: about 55K training conversations, 25K hidden test, $100K prizes. Source: [LMSYS blog](https://lmsys.org/blog/2024-05-02-kaggle-competition)
  - First place trained Llama3-70B or Qwen2-72B teachers in 5 folds, distilled them into Gemma2-9B students with a KL loss, then averaged the LoRA layers of the 5 students into one model. Source: [Nebius review, 2024-09-19](https://nebius.com/blog/posts/chatbot-arena-competition-review)

**Calibration and abstention**
- **Prompting alone is insufficient for calibration.** Fine-tuning on about 1,000 graded correct/incorrect examples, using LoRA "through the model's features", produces uncertainty estimates that generalize, and some models can estimate other models' uncertainty. Source: [Kapoor et al., NeurIPS 2024, arXiv 2406.08391](https://arxiv.org/abs/2406.08391)
- **AbstentionBench (Meta FAIR, NeurIPS 2025)** covers 20 datasets: unanswerable, underspecified, false-premise, subjective and stale questions.
  - Across 20 frontier LLMs, reasoning fine-tuning reduced abstention by about 24% on average, even in math and science.
  - Scale helps little. A crafted system prompt raises abstention but "does not resolve models' fundamental inability to reason about uncertainty."
  - Abstention precision is generally high; recall is the problem.
  - Sources: [arXiv 2506.09038](https://arxiv.org/abs/2506.09038v1), [GitHub](https://github.com/facebookresearch/AbstentionBench)

**Decision instability (directly relevant to determinism)**
- **"Deterministic" settings are not deterministic** (Atil et al., v5 2025-04-02).
  - Setup: 5 LLMs at deterministic settings, 8 tasks, 10 runs each.
  - Accuracy varied up to 15% across naturally occurring runs, with up to a 70% gap between best and worst possible performance. "None of the LLMs consistently delivers repeatable accuracy across all tasks, much less identical output strings."
  - The paper proposes TARr@N (agreement on raw output) and TARa@N (agreement on parsed answer) as stability metrics.
  - Source: [arXiv 2408.04667](https://arxiv.org/abs/2408.04667)
- **Prompt-format sensitivity** (FormatSpread, ICLR 2024): up to 76 accuracy points from formatting changes alone (LLaMA-2-13B). The sensitivity persists with larger models, more few-shot examples and instruction tuning. Source: [arXiv 2310.11324](https://arxiv.org/abs/2310.11324)
- **Tool-call / no-call decisions on BFCL v4** (official data, page "last updated 2026-04-12", retrieved 2026-10-21). The leaderboard separates relevance detection (calls when it should) from irrelevance detection (refrains when it should). Source: [BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), data file `data_overall.csv`
  - Llama-4-Scout: relevance 100%, irrelevance 44.92%.
  - Llama-3.3-70B: relevance 100%, irrelevance 53.53%.
  - Gemini-2.5-Flash (FC): irrelevance 93.67%.
  - GLM-4.6 (FC thinking): relevance 75.00%, irrelevance 84.96%.

### Inferences
- For approve/deny/triage decisions that must behave like a compiler, a scalar or label-token head has a big advantage. It is a single forward pass with no sampled reasoning, so the only run-to-run variance is numerical (see Q2). Skywork-Reward-V2-style BT heads or a Gemma/Qwen classifier distilled from a large teacher are the best-documented open recipes.
- Policy-conditioned reasoning classifiers such as gpt-oss-safeguard trade that determinism for flexibility. The policy changes without retraining, but every decision now depends on a sampled chain of thought. In a deterministic pipeline you would fix effort, use greedy decoding with batch-invariant kernels, and ideally parse a single final label.
- Abstention ("defer/unsure") should be a trained output class with calibrated thresholds rather than a prompt instruction. AbstentionBench and Kapoor et al. both indicate prompting is insufficient.
- Decision instability near thresholds is not well benchmarked. Atil's TARa@N plus per-item margin (logit gap between top-2 labels) is the natural metric to adopt.

### Gaps
- No 2026 leaderboard dedicated to LLM decision/classification calibration (ECE/Brier at scale) was found. The search budget ran out before a targeted search.
- Not found:
  - published quantitative accuracy for gpt-oss-safeguard vs dedicated classifiers (the OpenAI blog returned HTTP 403);
  - head-to-head data on decision flip rates near thresholds for generative vs discriminative classifiers;
  - log-loss values for the Kaggle LMSYS winners, and whether they used position-swap augmentation.

---

## Q1b. Function-calling / tool-use models: best open models, datasets, recipes, benchmarks

### Takeaway
On the official BFCL v4 board (frozen since 2026-04-12), the best open-weight entry is GLM-4.6 (FC thinking) at 72.38%, #4 overall, about 5 points behind Claude Opus 4.5 at roughly 1/19th the cost.

Small specialists (xLAM-2 8B–70B, ToolACE-2, Hammer2.1, Arch-Agent, Nanbeige4-3B) excel on single- and multi-turn AST categories. xLAM-2-70b has the highest multi-turn score of any model on the board, but specialists collapse on web-search and memory agentic categories.

On τ²-bench (official, 4 trials), open-weight Qwen3.5-397B-A17B has the best telecom reliability of any model (pass^4 92.1). Every model loses 10–30 points going from pass^1 to pass^4 on retail and airline, so consistency, not single-shot accuracy, is the binding constraint.

By late 2026 frontier labs have largely moved to Toolathlon-Verified, MCP-Atlas, MCPMark and τ³-Banking. Epoch AI rated BFCL v4 "Flawed" (48% of sampled tasks defective).

### Cited Findings
**BFCL v4 official data** ([leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html); data `data_overall.csv` retrieved 2026-10-21; page states last updated 2026-04-12, eval commit f7cf735, `bfcl-eval==2025.12.17`; overall = unweighted mean of subcategories)

| Model | Rank | Overall | Other scores | License | Cost / latency |
|---|---|---|---|---|---|
| Claude-Opus-4.5 (FC) | #1 | 77.47% | multi-turn 68.38, web-search 84.5, memory 73.76 | proprietary | $86.55 |
| GLM-4.6 (FC thinking) | #4, top open | 72.38% | non-live 87.56, live 80.90, multi-turn 68.00, web-search 77.50, memory 55.70 | MIT | $4.64 / 4.34s mean |
| Kimi-K2-Instruct (FC) | #11 | 59.06% | — | modified-MIT | — |
| DeepSeek-V3.2-Exp (Prompt+Thinking) | #14 | 56.73% | — | MIT | — |
| DeepSeek-V3.2-Exp (FC mode) | — | 54.12% | non-live AST only 34.85% | MIT | — |
| xLAM-2-32b-fc-r | #18 | 54.66% | non-live 89.60, multi-turn 69.50, web-search 25.5, memory 20.86 | cc-by-nc-4.0 | — |
| xLAM-2-70b-fc-r | #22 | 53.07% | **multi-turn 77.38%, highest on the entire board**; web-search 15.0, memory 14.41 | cc-by-nc-4.0 | — |
| Qwen3-235B-A22B-Instruct-2507 (Prompt) | #23 | 52.15% | — | Apache-2.0 | $3.12 |
| Qwen3-235B-A22B-Instruct-2507 (FC) | — | 47.99% | non-live 37.40 | Apache-2.0 | — |
| Nanbeige4-3B-Thinking-2511 (3.9B) | #25 | 51.40% | multi-turn 51.12 | Apache-2.0 | — |
| GPT-5-nano (FC) | — | 51.45% | — | proprietary | — |
| Qwen3-32B (FC) | #29 | 48.71% | — | Apache-2.0 | — |
| xLAM-2-8b-fc-r | #34 | 46.68% | multi-turn 70.00 | cc-by-nc-4.0 | — |
| GPT-5.2 (FC) | — | — | multi-turn 28.12 | proprietary | — |
| BitAgent-Bounty-8B | #36 | 46.23% | **live 93.12%, highest live score**; web-search 0.0 | — | — |
| Arch-Agent-32B | #37 | 45.37% | — | — | — |
| ToolACE-2-8B | #40 | 42.44% | — | — | — |
| xLAM-2-3b-fc-r | #42 | 41.22% | multi-turn 58.38 | — | — |
| Hammer2.1-7b | #64 | 31.67% | web-search 0, memory 0 | — | — |
| Granite-4.0-350m | — | 18.98% | — | — | — |

Other BFCL v4 observations:
- **Interface-mode swings for the same model:**
  - Claude-Opus-4.5: 77.47% FC vs 33.47% Prompt.
  - Claude-Haiku-4.5: 68.70% FC vs 25.26% Prompt.
  - Qwen3-235B-2507: 52.15% Prompt vs 47.99% FC.
- **Format sensitivity** (max delta / SD across prompt formats, prompt mode only):

| Model | Max delta | SD |
|---|---|---|
| Qwen3-235B-2507 | 8.0 | 1.95 |
| Gemini-3-Pro | 8.5 | 1.7 |
| Qwen3-32B | 15.5 | 3.75 |
| Mistral-Small-2506 | 50.0 | 13.57 |
| Llama-3.1-8B | 74.5 | — |
| ToolACE-2-8B | 81.5 | 27.92 |
| Phi-4 | 81.5 | — |

- **Parse/format failures zero out models entirely:** Llama-3.1-Nemotron-Ultra-253B (FC) scores 10.00% overall and Ministral-8B (FC) 11.10%, each with 0% on every category except irrelevance (100%).
- **Coverage is stale:** none of the 2026 open frontier models (GLM-5.x, Kimi K2.6/K3, Qwen3.5+/3.8, DeepSeek-V4) is on the official board as of 2026-10-21.
  - Aggregators list vendor-reported BFCL-V4 numbers, e.g. Qwen3.5-397B-A17B 72.9 on [llm-stats](https://llm-stats.com/benchmarks/bfcl-v4) (secondary; updated Oct 2026).
  - [benchlm.ai](https://benchlm.ai/benchmarks/bfcl-v4) lists an unverifiable "BTL-3" at 88.5%. Treat both as unverified.

**BFCL validity**
- Epoch AI review (2026-09-10), verdict "Flawed": 24 of 50 (48%) randomly sampled tasks had defects that could affect accuracy. Source: [Epoch AI](https://epoch.ai/benchmarks/berkeley-function-calling-leaderboard/review)
  - Defect rate by category: memory 7/10, multi-turn 8/15, web search 4/10, live AST 2/5, irrelevance 2/5, non-live AST 1/5.
  - Examples: stale web-search answer keys; irrelevance tasks that mark a correct `get_local_time` call as failure; strict argument matching that rejects `include_status=true`.

**τ²-bench official submissions** ([taubench.com](https://taubench.com/); per-submission JSON in Sierra's public bucket, e.g. [Qwen3.5-397B-A17B submission](https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions/qwen3.5-397b-a17b-think_sierra_2026-03-02/submission.json), retrieved 2026-10-21)
- **Protocol:** tau2-bench v1.0.1, 4 trials, user simulator gpt-5.2 (low effort), seed 300.
- **pass^k definition:** probability that all k trials succeed. Source: [τ-bench, arXiv 2406.12045](https://arxiv.org/abs/2406.12045)
- **Comparability:** banking_knowledge was re-graded 2026-07-15. Scores before v1.0.1 are not comparable.

pass^1 → pass^4, current submissions (airline / retail / telecom / banking_knowledge):

| Model | Weights | Airline | Retail | Telecom | Banking_knowledge |
|---|---|---|---|---|---|
| Qwen3.5-397B-A17B (thinking; 2026-03-02) | open | 81.5→68.0 | 84.4→59.6 | **97.8→92.1** | 9.8→5.2 |
| GLM-5 (thinking) | open | 82.5→70.0 | 73.7→43.9 | 86.8→62.3 | 9.8→3.1 |
| Claude Opus 4.5 (high) | closed | 84.0→70.0 | 79.6→51.8 | 92.3→78.1 | — |
| GPT-5.2 (high) | closed | 83.0→72.0 | 81.6→51.8 | 89.7→71.9 | — |
| GPT-5.2 (reasoning effort none) | closed | 52.5→22.0 | — | 57.2→30.7 | — |

Banking_knowledge (hardest domain), Aug 2026 submissions:

| Model | Weights | pass^1→pass^4 |
|---|---|---|
| Qwen 3.8 Max (hosted version of open Qwen3.8-2.4T-A95B) | open base | **55.2→35.1**, best of all |
| Claude Opus 5 | closed | 48.7→32.0 |
| Grok 4.5 | closed | 47.9→32.0 |
| GPT-5.6-sol | closed | 46.9→27.8 |
| Kimi K3 | open | 37.1→17.5 |
| GLM-5.2 | open | 37.1→13.4 |

- RAFT-30B-A3B (Northeastern, custom scaffold, user simulator Claude Sonnet 4.5, so not directly comparable): retail 82.5→61.4.
- Legacy v0.1.3 runs (user simulator gpt-4.1):
  - Qwen3-Max-Thinking: telecom 98.2→93.9.
  - GPT-5: telecom 95.8→85.1.
  - GPT-4.1: telecom 34.0→20.4.
  - NVIDIA Nemotron-Orchestrator-8B (custom), pass^1: airline 56.0, retail 84.2, telecom 88.6.
  - DeepSeek-V3.2, pass^1: 63.8 / 81.1 / 96.2.

**Original τ-bench (2024-06):** gpt-4o succeeded on <50% of tasks, and pass^8 was <25% in retail. Source: [arXiv 2406.12045](https://arxiv.org/abs/2406.12045)

**2026 frontier open models' tool-use numbers** (vendor model cards; temperature 1.0)
- **Kimi K3** (HF 2026-06-13): Toolathlon-Verified 76.5, MCPMark-Verified 94.5, MCP-Atlas 84.2, τ³-Banking 33.4 (cited from Artificial Analysis as of 2026-07-23). Source: [HF card](https://huggingface.co/moonshotai/Kimi-K3)
- **GLM-5.3** (2026-08-25): Toolathlon Verified 73.0, AutomationBench 48.2. Source: [HF card](https://huggingface.co/zai-org/GLM-5.3)
- **Qwen3.8-Max:** Toolathlon 72.5. Source: [HF card Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
- **NVIDIA Nemotron-3.5-Bolt-74B-A9B** (Oct 2026; OpenMDW-1.1): Tau3 Banking 17.01, vs Qwen3.6-35B-A3B 10.52 and Gemma4-26B-A4B 14.02. Source: [HF card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16)
- **Kimi K2** (Jul 2025, 1T/32B): Tau2-Bench 66.1, ACEBench (En) 76.5, non-thinking. Training was a large-scale agentic data-synthesis pipeline plus joint RL in real and synthetic environments. Source: [arXiv 2507.20534](https://arxiv.org/abs/2507.20534)

**Training recipes and datasets**
- **APIGen-MT → xLAM-2 (2025-04).** Two-phase simulated agent–human interplay for verified multi-turn data.
  - xLAM-2-70b-fc-r: τ-bench 56.2% pass@1 (≥5 trials), vs base Llama-3.1-70B-Instruct 38.2%, DeepSeek-V3 40.6%, GPT-4o 52.9%, Claude 3.5 Sonnet(new) 60.1%. 32B: 54.6%. 8B: 46.7%.
  - Reported as #1/#2 on BFCL v3 (78.19% / 75.83%) as of early Apr 2025.
  - The paper reports pass^k curves explicitly to measure consistency.
  - Sources: [arXiv 2504.03601](https://arxiv.org/pdf/2504.03601), [HF card](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r)
  - License cc-by-nc-4.0 per the [HF API](https://huggingface.co/api/models/Salesforce/Llama-xLAM-2-70b-fc-r).
- **ToolACE (Huawei Noah/USTC).** A self-evolution synthesis builds a 26,507-API pool; multi-agent dialog generation; dual-layer (rule plus model) verification. Follow-ups: ToolACE-R, ToolACE-DEV, ToolACE-MT. ToolACE-2.5-Llama-3.1-8B (2025-08, Apache-2.0) adds multi-turn. Sources: [HF card](https://huggingface.co/Team-ACE/ToolACE-2.5-Llama-3.1-8B), [arXiv 2409.00920](https://arxiv.org/abs/2409.00920)
- **Hammer (MadeAgents).** Function masking (hiding function names) plus an irrelevance-augmented dataset, because function-calling models are "often misled by specific naming conventions." Source: [arXiv 2410.04587](https://arxiv.org/abs/2410.04587)
- **Constrained decoding for format determinism.** XGrammar (MLSys'25) gives CFG-constrained generation with up to 100x speedup over prior engines and near-zero end-to-end serving overhead. Source: [arXiv 2411.15100](https://arxiv.org/abs/2411.15100)

### Inferences
- **Fine-tuning moves sub-skills, not the whole board.** Specialist tool-callers dominate the narrow sub-skills they were trained on (multi-turn AST: xLAM-2-70b 77.4 > every frontier model) yet score near 0 on web-search and memory. BFCL's unweighted average therefore understates them for pure tool-emission workloads and overstates them for agentic ones.
- **Freeze the interface.** For a "only emit tool calls" deterministic component, the evidence points to: a fixed native FC chat template (FC-vs-prompt swings of 20–45 points; format-sensitivity deltas up to 81.5 for ToolACE-2-8B), grammar-constrained decoding (XGrammar) to remove malformed-call failures (the 0%-scoring Nemotron-Ultra and Ministral rows), and an explicit irrelevance/no-call class.
- **Report pass^k, not pass^1.** The pass^1→pass^4 drop is the best public consistency metric. Telecom drops are small for the strongest models (Qwen3.5-397B −5.7; Qwen3-Max-Thinking −4.3), while retail drops are large for everyone (−21 to −30). Reasoning effort strongly affects consistency: GPT-5.2 at effort none drops airline pass^4 to 22.0, vs 72.0 at high.
- **Do not use BFCL v4 for fine-grained ranking.** Given Epoch's 48% defect rate, differences of a few points are noise. τ²-bench (with trajectories published) and Toolathlon/MCP-Atlas are better 2026 targets.

### Gaps
- No primary 2026 BFCL v4 numbers for GLM-5.x, Kimi K3, Qwen3.5/3.8 or DeepSeek-V4. The official board is frozen at 2026-04-12.
- No primary ToolSandbox or ACEBench 2026 leaderboard data was retrieved, beyond Kimi K2's ACEBench 76.5.
- No leaderboard found reports argument-hallucination rates (fabricated parameter values) as a separate metric. The closest proxy is BFCL v4's "Multi Turn Miss Param" category, which tests whether the model asks for a missing parameter rather than inventing one. Values from `data_overall.csv`, retrieved 2026-10-21 ([BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)):

| Model | Miss Param |
|---|---|
| xLAM-2-70b | 74.0% |
| xLAM-2-8b | 65.0% |
| GLM-4.6 (FC thinking) | 63.0% |
| Claude-Opus-4.5 (FC) | 58.0% |
| Qwen3-235B-2507 (FC) | 33.5% |
| ToolACE-2-8B | 30.5% |
| Hammer2.1-7b | 21.5% |

  - Here too the specialists trained on multi-turn data (xLAM-2) beat frontier generalists.
- No 2026 small function-calling specialist release (successor to xLAM-2, Hammer2.1, ToolACE-2.5) was confirmed via HF API. Salesforce's latest xLAM is 2025-03.

---

## Q1c. Contrastive / cross-evaluation models (pairwise/pointwise judges, RMs, PRMs, verifiers, generative verifiers, self-consistency, debate): best open models, benchmarks, recipes

### Takeaway
Small, data-curated Bradley-Terry reward models still lead open RM benchmarks. Skywork-Reward-V2-Llama-3.1-8B posts RewardBench 2 = 84.1, JudgeBench 80.0 and RM-Bench 92.8, well above prompted GPT-4o and Claude-3.5 judges (RB2 about 65).

Generative/"thinking" verifiers (GenRM, ThinkPRM, J1, RM-R1, DeepSeek-GRM) win on data efficiency and out-of-domain generalization and can use self-consistency (maj@k), at much higher inference cost. For PRMs, the Qwen "lessons" recipe matters: consensus-filtered labels, and do not trust best-of-N evaluation alone. PRMBench and ProcessBench show current PRMs remain weak.

Judge bias is now well characterized:
- Position bias depends on the quality gap.
- Style/markdown bias is the dominant bias in 2026 audits.
- Verbosity bias is model-dependent.
- Self-preference is linked to low perplexity and is not reduced by capability.

### Cited Findings
**Reward model leaders (open)**
- **Skywork-Reward-V2 (2025-07)**, 7-benchmark table. Source: [HF card](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)

| Model | RB1 | RB2 | PPE-Pref | PPE-Corr | RMB | RM-Bench | JudgeBench | Avg |
|---|---|---|---|---|---|---|---|---|
| Skywork-Reward-V2-Llama-3.1-8B | 96.4 | 84.1 | 77.3 | 83.4 | 86.4 | 92.8 | 80.0 | 85.8 |
| …-8B-40M (research-only; trained on the full 40M pairs incl. ~1/3 flipped) | — | — | — | — | — | — | — | 88.6 |
| Skywork-Reward-V2-Qwen3-8B | — | 78.2 | — | — | — | — | 73.4 | 79.3 |
| INF-ORM-Llama3.1-70B | — | 76.5 | — | — | — | — | — | 73.5 |
| GPT-4o (generative judge) | — | 64.9 | — | — | — | — | 59.8 | — |
| Claude-3.5-Sonnet (generative judge) | — | 64.7 | — | — | — | — | 64.8 | — |

- **Generative RMs in the same table:**
  - J1-Llama-70B: RB1 93.3, RM-Bench 82.7, JudgeBench 60.0. Maj@32 lifts PPE-Corr only from 72.9 to 73.7.
  - RM-R1-DeepSeek-Distill-Qwen-32B: RM-Bench 83.9.
  - DeepSeek-GRM-27B with MetaRM: RB1 90.4.
  - Source: [HF card](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)
- **RewardBench 2 itself.** Source: [arXiv 2506.01937](https://arxiv.org/abs/2506.01937)
  - ICLR 2026; v2 dated 2026-04-23.
  - Models score about 20 points lower than on RewardBench 1.
  - Uses new human prompts and is "highly correlated" with downstream best-of-N and PPO RLHF performance.
  - Composition: 1,865 items in 6 subsets (Factuality, Precise IF, Math, Safety, Focus, Ties), per the [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/ea4fe0a56d02c93401902b5b4c6b12da-Paper-Conference.pdf) (via search snippet).
  - The paper's top-models table lists Skywork-Reward-V2-Llama-3.1-8B first at 84.1 (same source).
  - Conflict: the secondary [Klu page](https://klu.ai/benchmarks/reward-bench) lists a different top-5 (Gemini-2.5-flash 77.2), which is likely an outdated snapshot.
  - The official results dataset was last modified 2025-12-11 per the [HF API](https://huggingface.co/datasets/allenai/reward-bench-2-results), so no 2026 RMs are scored there.
- **Other open RMs (existence and dates via HF API, numbers not retrieved):**
  - nvidia/Qwen-3-Nemotron-32B-Reward (2025-05-29) and Llama-3.3-Nemotron-70B-Reward-Principle (2025-10-12). Source: [HF API](https://huggingface.co/api/models?author=nvidia&search=Reward)
  - nvidia/Flora-Rewarder (2026-10-15): a 27B robotics RM on Qwen3.5-27B. It emits a 1–5 outcome score plus dense per-0.5s action-quality scores. Non-commercial license. Source: [HF card](https://huggingface.co/nvidia/Flora-Rewarder)

**Judge benchmarks**
- **JudgeBench (ICLR 2025):** objectively-labeled pairs in knowledge, reasoning, math and coding. Strong models like GPT-4o perform "just slightly better than random." Source: [arXiv 2410.12784](https://arxiv.org/abs/2410.12784)
- **RM-Bench (2024-10):** tests sensitivity to subtle content differences and resistance to style bias. Across about 40 RMs, even state-of-the-art RMs average only 46.6%, below random (50%) under style-bias interference. Strongly correlated with policy performance. Source: [arXiv 2410.16184](https://arxiv.org/abs/2410.16184)

**PRMs and verifiers**
- **PRMBench (ACL 2025):** 6,216 problems and 83,456 step-level labels, scored on simplicity, soundness and sensitivity. Across 15 models (open PRMs plus prompted LLM critics), it finds "significant weaknesses." Source: [arXiv 2501.03124](https://arxiv.org/abs/2501.03124)
- **Qwen "Lessons of developing PRMs" (2025-01, ACL Findings 2025).** Source: [arXiv 2501.07301](https://arxiv.org/pdf/2501.07301)
  - Monte-Carlo step labels are worse than LLM-judge or human labels.
  - Consensus filtering (keep steps where MC and the LLM judge agree) is the recommended labeling recipe.
  - Best-of-N is a biased PRM metric: BoN-optimized PRMs drift toward outcome-RM behavior, so pair BoN with step-level ProcessBench.
  - Released Qwen2.5-Math-PRM-7B/72B. The 7B has 720K downloads per the [HF API](https://huggingface.co/api/models?author=Qwen&search=PRM).
  - ProcessBench F1 of 73.5 (7B) and 78.3 (72B) appears only in a secondary summary ([HF discussion](https://huggingface.co/datasets/rl-llm-wiki/knowledge-base/discussions/670)) and is unverified.
- **ThinkPRM (v5 2025-12).** A generative long-CoT PRM using 1% of PRM800K labels. Source: [arXiv 2504.16828](https://arxiv.org/abs/2504.16828)
  - Beats discriminative PRMs trained on the full PRM800K by 8% on a GPQA-Diamond subset and 4.5% on a LiveCodeBench subset.
  - Beats LLM-as-judge by 7.2% on a ProcessBench subset at equal token budget.
- **Generative Verifiers / GenRM (ICLR 2025).** Trained jointly on verification and generation with next-token prediction; CoT verification with majority voting. Source: [arXiv 2408.15240](https://arxiv.org/abs/2408.15240)
  - Best-of-N gains: 5%→45.3% on algorithmic tasks; 73%→93.4% on GSM8K.
  - Outperforms discriminative, DPO and LLM-judge verifiers.
- **Self-consistency with confidence (DeepConf, Meta, 2025-08):** DeepConf@512 reaches up to 99.9% on AIME 2025 (Qwen3 and GPT-OSS models), with up to 84.7% fewer tokens than full parallel thinking. Source: [arXiv 2508.15260](https://arxiv.org/abs/2508.15260)
- **Domain-specific trained judges beat zero-shot frontier judges:** on LitBench (creative writing), trained Bradley-Terry and generative RMs reach 78% vs 73% for the best zero-shot judge (Claude-3.7-Sonnet). Source: [arXiv 2507.00769](https://arxiv.org/abs/2507.00769)

**Judge consistency failure modes and mitigations**
- **Position bias** ("Judging the Judges", AACL-IJCNLP 2025; v9 2025-11). Source: [arXiv 2406.07791](https://arxiv.org/abs/2406.07791)
  - Scope: 15 judges, 22 tasks (MTBench and DevBench), about 40 answer models, >150K instances.
  - Metrics: repetition stability, position consistency and preference fairness.
  - Position bias is not random, varies by judge and task, is weakly affected by prompt length, and is strongly affected by the quality gap between candidates.
- **2026 TMLR audit of 9 mitigation strategies.** Source: [arXiv 2604.23178](https://arxiv.org/abs/2604.23178)
  - Position bias was small (≤0.04).
  - Style bias was dominant (0.10–0.76, favoring markdown).
  - Verbosity bias is model-dependent: +0.24 to +0.44 for Gemini Pro/Flash and Llama; −0.12 for Claude (prefers concise); −0.04 for GPT-4o.
  - Best debiasing gains: +11.5 pp (Claude, S8) and +7.5 pp (Flash, S8), both surviving Holm-Bonferroni correction.
  - Gemini-2.5-Flash with combined strategies: 71.0% agreement (κ 0.549) at about $0.001 per evaluation.
- **Self-preference:**
  - GPT-4 shows significant self-preference. The mechanism is a preference for lower-perplexity (familiar) text, independent of authorship. Source: [arXiv 2410.21819](https://arxiv.org/abs/2410.21819)
  - A 2026 gold-free method using equal-quality pairs across 20 LLMs finds capability is "uncorrelated or even negatively correlated" with low self-preference. A multi-dimensional "cognitive-load decomposition" evaluation reduces it by 31.5% on average. Source: [arXiv 2604.22891](https://arxiv.org/abs/2604.22891)
- **Pluralistic miscalibration:** reward models and LM judges agree less with humans on items with idiosyncratic preferences, even when quality is comparable. Source: [Artificial Hivemind, arXiv 2510.22954](https://arxiv.org/abs/2510.22954)
- **Production mitigations (EQ-Bench):** judge both orders and average; truncate outputs to 4,000 characters before pairwise judging. EQ-Bench states that self-bias, positivity bias and slop bias are not controlled. Source: [eqbench.com/about](https://eqbench.com/about.html)

### Inferences
- **For a deterministic evaluator, prefer a scalar-head BT RM.** It is one forward pass, has no sampling, and on these benchmarks is more accurate than prompted frontier judges. Use generative verifiers only where you need rationales or OOD robustness, and then fix the decoding (greedy plus batch-invariant) or aggregate with maj@k. J1's Maj@32 gain was under 1 point, so sampling adds variance-reduction more than accuracy.
- **Use both-order evaluation by default.** Pairwise judges should always run in both orders. A judge that flips under swap is the cheapest available "inconsistency detector" and can trigger abstention or escalation.
- **Cross-model agreement panels** are motivated by the self-preference evidence, but Artificial Hivemind's inter-model homogeneity finding means panels of similar models may share the same blind spots. Diversity of model families matters more than panel size.
- **PRM training should copy the Qwen recipe:** consensus-filtered hard labels plus step-level evaluation. Otherwise PRMs silently degrade into outcome RMs.

### Gaps
- No live RewardBench 2 or JudgeBench leaderboard entries for 2026-released RMs were found. Whether any 2026 RM (e.g. Nemotron or Qwen3.5-based) surpasses Skywork-Reward-V2 is unknown.
- Scores were not retrieved for: Nemotron reward models, Qwen3/3.5-based PRMs, and "debate" protocols.
- Run-to-run judge inconsistency (same prompt, same judge, different seeds) quantified at scale beyond Shi et al.'s repetition-stability metric was not found.

---

## Q1d. Creative content models: best open creative-writing models, how quality is achieved, creativity vs determinism

### Takeaway
On EQ-Bench Creative Writing v3 (live data retrieved 2026-10-21):
- The best open-weight writers are Kimi K3 (Elo 2082.3) and GLM-5.3 (2075.0). Both have unusually low "slop" (9.7 and 8.4) and trail the top closed models (GPT-6-Astra 2173, Claude Fable 5.1 2162, Claude Opus 5 2133) by about 90–100 Elo.
- Qwen3.8-27B (dense 27B, Apache-2.0, Elo 1671) is the best small open writer found.
- Older community fine-tunes (Darkest-muse, Gemma-3-Glitter) are now far behind.

Quality comes from frontier-scale RL plus trained creative reward models (LitBench, WritingBench critic) and anti-slop measures. The central tension is well documented: preference tuning causes "typicality bias" mode collapse and cross-model homogeneity. Verbalized Sampling recovers 1.6–2.1x diversity.

### Cited Findings
**EQ-Bench Creative Writing v3 leaderboard** (from the leaderboard data embedded in [eqbench.com/creative_writing.html](https://eqbench.com/creative_writing.html), file `creative_writing.js`, retrieved 2026-10-21; columns Elo / rubric / slop)

| Model | Weights | Elo | Rubric | Slop | Notes |
|---|---|---|---|---|---|
| gpt-6-astra | closed | 2173.3 | 16.80 | 8.41 | Asterisk-marked; meaning not documented in retrieved data, possibly provisional |
| claude-fable-5-1 | closed | 2162.0 | 16.95 | 8.16 | Asterisk-marked |
| claude-opus-5 | closed | 2132.6 | 17.07 | 6.59 | — |
| gpt-6-sol | closed | 2124.7 | — | — | — |
| Kimi K3 | open | **2082.3** | 16.85 | 9.70 | — |
| GLM-5.3 | open | **2075.0** | 17.04 | 8.42 | — |
| Qwen3.8-2.4T-A95B | open | 1842.5 | — | — | — |
| meta-models/Muse-Glimmer-30B | open? | 1798.3 | — | — | Open-weight status not verified |
| GLM-5.2 | open | 1756.5 | — | — | — |
| Kimi-K2.6 | open | 1724.5 | — | — | — |
| NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | open | 1692.4 | — | — | — |
| Qwen3.8-27B | open | 1671.3 | 15.50 | 12.12 | Dense 27B |
| thinkingmachines/Inkling | open? | 1610.8 | — | — | Open-weight status not verified |
| DeepSeek-V4-Flash | open | 1559.1 | — | 20.92 | — |
| DeepSeek-V4-Pro | open | 1553.2 | — | — | — |
| Qwen3.5-397B-A17B | open | 1478.2 | — | — | — |
| NousResearch/Hermes-4-405B | open | 1420.8 | — | 39.62 | — |
| gemma-4-31B-it | open | 1368.2 | — | 29.69 | — |
| Gemma-3-Glitter-12B | open (community) | 1118.4 | — | — | — |
| Darkest-muse-v1 (Gemma-2-9B fine-tune) | open (community) | 1104.2 | — | — | — |
| gpt-oss-120b | open | 961.0 | — | — | — |

- **Elo drifts as models are added.** The Featherless snapshot of 2026-09-17 gave GLM-5.2 1745.8 and Kimi K2.6 1715.5, vs 1756.5 and 1724.5 on 2026-10-21. Source: [Featherless blog](https://featherless.ai/blog/best-ai-for-creative-writing-2026)
- **Methodology.** Source: [eqbench.com/about](https://eqbench.com/about.html)
  - 32 prompts × 3 iterations at temperature 0.7, min_p 0.1.
  - Rubric judge: Claude Sonnet 4. Elo judge: Claude Sonnet 4.6 since 2026-03-01 (pre/post scores not strictly comparable).
  - Glicko-style pairwise matching weighted by win margin; anchors DeepSeek-R1 = 1500 and ministral-3b = 200.
  - Both-order judging; 4,000-character truncation.
  - Rubric scores are compressed: most of the top 60 models fall between 16.0 and 17.2.
- **Licenses and sizes** (HF API/cards):
  - Kimi K3: 2.8T/104B MoE, Kimi K3 License. Source: [HF](https://huggingface.co/moonshotai/Kimi-K3)
  - GLM-5.3: custom glm-5.3 license. Source: [HF](https://huggingface.co/zai-org/GLM-5.3)
  - GLM-5.2: MIT, about 753B. Source: [HF API](https://huggingface.co/api/models/zai-org/GLM-5.2)
  - Qwen3.8-27B: Apache-2.0. Source: [HF](https://huggingface.co/Qwen/Qwen3.8-27B)
  - gemma-4-31B-it: Apache-2.0. Source: [HF API](https://huggingface.co/api/models/google/gemma-4-31B-it)

**How quality is achieved (recipes and evaluators)**
- **LitBench (2025-07):** 2,480 debiased human-labeled Reddit story-pair test comparisons plus 43,827 training pairs. Trained BT and generative RMs reach 78% vs 73% for zero-shot Claude-3.7-Sonnet, and align with humans on new LLM-generated stories. This makes them usable for best-of-N or RL rewards. Source: [arXiv 2507.00769](https://arxiv.org/abs/2507.00769)
- **WritingBench (v4 2025-11):** 6 domains and 100 subdomains. LLM-generated, query-specific criteria are scored by a fine-tuned critic; data curated with it lets a 7B model outperform GPT-4o at writing. Source: [arXiv 2503.05244](https://arxiv.org/abs/2503.05244)
- **Practitioner sampling advice:** temperature around 0.9 for varied creative output (Featherless, 2026-09-17). Source: [Featherless](https://featherless.ai/blog/best-ai-for-creative-writing-2026)
- **Vendor sampling defaults** for Qwen3.8: thinking temperature 1.0, top_p 0.95, top_k 20; instruct temperature 0.7, top_p 0.8, presence_penalty 1.5. Higher presence_penalty "may occasionally result in language mixing." Source: [HF card Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

**Diversity collapse vs determinism**
- **Verbalized Sampling** (2025-10; v4 2026-07-15). Source: [arXiv 2510.01171](https://arxiv.org/abs/2510.01171)
  - Mode collapse is attributed to typicality bias in human preference data (annotators prefer familiar text).
  - Prompting the model to verbalize a distribution over k responses raises creative-writing diversity 1.6–2.1x over direct prompting, without sacrificing factuality or safety.
  - More capable models benefit more.
- **NoveltyBench (2025-04):** across 20 models, larger models in a family are often less diverse than smaller ones, and all fall well short of human diversity. Source: [arXiv 2504.05228](https://arxiv.org/abs/2504.05228)
- **Artificial Hivemind** (NeurIPS 2025 D&B oral; v2 2026-10-06): the Infinity-Chat set has 26K open-ended queries and 31,250 human annotations. It finds strong intra-model repetition and even stronger inter-model homogeneity. Source: [arXiv 2510.22954](https://arxiv.org/abs/2510.22954)
- A 2026 quality-diversity alignment paper ("Forty Shades of Blue", mode-conditioned RL) targets this trade-off. Only the title and framing were seen, via [arXiv 2609.14896](https://arxiv.org/pdf/2609.14896), so its contents are unverified.

### Inferences
- **"Controlled creativity" decomposes into two controls:**
  1. Diversity comes from an explicit, enumerated latent (Verbalized-Sampling-style k candidates, or mode-conditioning) rather than from temperature noise.
  2. Selection is deterministic: a trained creative RM (LitBench- or WritingBench-style) picks via best-of-N with fixed seeds.
  - With batch-invariant kernels (Q2), seeded sampling becomes bit-reproducible. "Creative but reproducible" is therefore achievable as seed → identical output, while still offering diversity across seeds.
- **Slop score is a cheap, deterministic proxy** for one axis of RLHF homogenization. Kimi K3 and GLM-5.3 show that low slop and top quality can coexist.
- **Rank on Elo, not rubric scores.** Rubric scores saturate (16–17/20) and Elo is relative and drifts. For a deterministic compiler pipeline, a fixed in-house creative RM plus pinned judge versions is preferable to tracking EQ-Bench rank.

### Gaps
- Not retrieved:
  - EQ-Bench Longform and "Open Writing Leaderboard" numbers ([owl.eqbench.com](https://owl.eqbench.com/) exists);
  - LitBench- or WritingBench-trained open RM checkpoints and their 2026 successors;
  - Kimi's or GLM's published recipe for creative RL (e.g. rubric/self-critique rewards). The Kimi K2 abstract does not specify it.
- Whether Muse-Glimmer-30B and Inkling weights are openly downloadable was not verified.
- No quantitative study was found of diversity (e.g. NoveltyBench score) for the 2026 open leaders.

---

## Q1e. Deep reasoning & analytical models (RLVR/GRPO, long CoT, deep-research agents, table/code-execution reasoning, reproducibility of long chains)

### Takeaway
As of October 2026 the open-weight reasoning frontier is held by very large sparse MoEs:
- Kimi K3: 2.8T/104B active.
- DeepSeek-V4-Pro-0813: about 1.6T, MIT.
- DeepSeek-V4.1-Flash: 552B backbone, 8B/16B active, MIT.
- GLM-5.3: about 753B class.
- Qwen3.8-2.4T-A95B.

These reach GPQA-Diamond 90–94 and HLE-with-tools 56–64, roughly at parity with Claude Opus 4.8/5.0 and GPT-5.6 on many agentic benchmarks per vendor cards.

Compact options are Qwen3.8-27B (dense, Apache-2.0; GPQA 89.2, LiveCodeBench v6 90.3), Qwen3.8-Flash-Next (6B active), and Tongyi DeepResearch 30B-A3B for deep research.

Recipes have converged on RLVR with GRPO variants (DAPO, leave-one-out, strictly on-policy) plus distilled long-CoT SFT (OpenThoughts3, 1.2M examples). Reproducibility is fragile: bf16 greedy decoding alone produced up to 9% accuracy and 9,000-token length differences across hardware and batch configurations.

### Cited Findings
**Frontier open-weight reasoning models (vendor-reported; all evaluated at temperature 1.0)**
- **Kimi K3** (HF 2026-06-13; Kimi K3 License). Source: [HF card](https://huggingface.co/moonshotai/Kimi-K3)
  - Architecture: MoE, 2.8T total / 104B active, 896 experts with 16 selected plus 2 shared, KDA+Gated-MLA hybrid attention, 1M context, MXFP4 weights via QAT.
  - GPQA-Diamond 93.5; HLE-Full 43.5 (no tools) / 56.0 (with tools).
  - BrowseComp 91.2 (with context compaction at 300K; 90.4 with the full 1M context); DeepSearchQA F1 95.0; Terminal-Bench 2.1 88.3.
  - Evaluated at temperature 1.0, top-p 0.95 or 1.0.
- **DeepSeek-V4.1-Flash** (2026-09-10; MIT). Source: [HF card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  - Architecture: multimodal MoE with 552B backbone, 8B/16B activated, causal encoder-decoder, plus Engram memory (196B) and DSpark speculative decoding; 45T pretraining tokens.
  - GPQA 90.9; HLE 36.8 (39.1 text-only); HLE with tools 63.9.
  - Codeforces 3471; MathArena Apex 65.6; Terminal-Bench 2.1 90.6; DeepSWE v1.1 74.2. In the vendor's own table these match or beat Opus-5.0 (TB2.1 89.1, DeepSWE 74.0).
  - Ships Python/Rust prompt encoders rather than a Jinja template.
- **DeepSeek-V4-Pro-0813** (2026-08-13; MIT; about 1.6T params per the [HF API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Pro)): HLE 42.7 (no tools) / 60.0 (with tools); TB2.1 87.9. Recommends temperature 1.0 and up to 384K output tokens at max effort. Source: [HF card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)
- **GLM-5.3** (2026-08-25): HLE with tools 62.5, TB2.1 88.2, TB3.0 28.3, DeepSWE 66.9. Source: [HF card](https://huggingface.co/zai-org/GLM-5.3)
  - GLM-5.3-Flash: 320B/18B active, MIT, hybrid sparse+linear attention. Source: [HF card](https://huggingface.co/zai-org/GLM-5.3-Flash)
- **Qwen3.8-2.4T-A95B** (2026-08-08; "qwen3.8-max" license; 512 experts, 10 routed + 1 shared): GPQA 92.6, HLE 43.6, HLE with tools 56.2. Source: [HF card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
- **Qwen3.8-27B** (dense, Apache-2.0, 2026-08-05): GPQA 89.2, HLE 30.8, LiveCodeBench v6 90.3, TB2.1 73.0. Source: [HF card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Qwen3.8-Flash-Next** (125B/6B active plus 51B n-gram embeddings; qwen-community-1.0 license): GPQA 91.7, HLE 35.9, LCB v6 91.9. Source: [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **MiniMax-M3** (2026-06-02; about 428B/23B active; minimax-community license): has enabled/adaptive/disabled thinking modes and recommends temperature 1.0, top_p 0.95. Source: [HF card](https://huggingface.co/MiniMaxAI/MiniMax-M3)
- **NVIDIA Nemotron-3.5-Bolt-74B-A9B** (Oct 2026; OpenMDW-1.1): GPQA 88.83 in BF16 vs 86.43 for the 24GB IQ2 quantization. Source: [HF card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16)
- **Conflict note:** aggregator pages disagreed on the "best open reasoning model" ([Vellum](https://www.vellum.ai/open-llm-leaderboard), [Fastino](https://fastino.ai/blog/best-open-weight-models-2026)). The DEV Community audit of 2026-10-19 says BenchLM listed Kimi K3 weights as "pending" ([dev.to](https://dev.to/ward_ed_6b5e6aa8ded94a987/best-open-weight-llm-in-october-2026-auditing-two-leaderboards-row-by-row-1b64)). The HF API shows moonshotai/Kimi-K3 public since 2026-06-13 with about 989K downloads ([HF API](https://huggingface.co/api/models?author=moonshotai&search=Kimi-K3)), so the "pending" claim appears outdated.

**Deep-research agents**
- **Tongyi DeepResearch-30B-A3B** (2025-09; Apache-2.0). Source: [HF card](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
  - Training: agentic continual pretraining, a fully synthetic data pipeline, and strictly on-policy customized GRPO (token-level policy gradient, leave-one-out advantage, filtering of negative samples).
  - ReAct mode: HLE 32.9, BrowseComp 43.4, BrowseComp-ZH 46.7, xbench-DeepSearch 75.0, FRAMES 90.6.
  - "Heavy" mode (parallel agents plus synthesis): HLE 38.3, BrowseComp 58.3. o3 comparison: HLE 24.9, BrowseComp 49.7.
  - Source for scores: [Tongyi Lab thread](https://threadreaderapp.com/thread/1967988004179546451.html). That thread's summary also states BrowseComp 45.3, which is internally inconsistent with the 43.4 above.

**Recipes**
- **DeepSeek-R1** (Nature 645:633–638, 2025): pure RL incentivizes reasoning (self-reflection, verification) without human reasoning traces; patterns transfer to small models via distillation. Source: [arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
- **DAPO** (decoupled clip plus dynamic sampling GRPO variant): 50 points on AIME 2024 from the Qwen2.5-32B base. Source: [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)
- **OpenThoughts3:** 1.2M SFT examples from 1,000+ controlled data-recipe experiments. OpenThoughts3-7B scores AIME25 53%, LCB 51%, GPQA-D 54%, i.e. +15.3/+17.2/+20.5 over DeepSeek-R1-Distill-Qwen-7B. Source: [arXiv 2506.04178](https://arxiv.org/abs/2506.04178)
- **Synthetic data blends in 2026 NVIDIA models:** Nemotron-3.5-Bolt includes math proofs synthesized by DeepSeek-V4-Pro and 1.28M LiveCodeBench tool-integrated-reasoning traces from DeepSeek-R1-0528. Source: [HF card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16)

**Reproducibility of long chains**
- **Numerical nondeterminism** (Yuan et al.; v2 2025-10-24). Source: [arXiv 2506.09501](https://arxiv.org/abs/2506.09501)
  - Under bf16 with greedy decoding, DeepSeek-R1-Distill-Qwen-7B shows up to 9% accuracy variation and 9,000-token response-length differences from changes in GPU count, GPU type and batch size.
  - Fix: LayerCast, which stores weights in 16-bit but computes in FP32.
- **Batch invariance** (Thinking Machines, 2025-09-10). Source: [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
  - Qwen3-235B-A22B-Instruct-2507, temperature 0, 1,000 runs: 80 unique completions, identical for 102 tokens, diverging at token 103.
  - With batch-invariant RMSNorm, matmul and attention kernels, all 1,000 were identical.
  - Overhead on Qwen3-8B: 26s → 55s unoptimized, 42s optimized.
  - Bitwise-identical sampler and trainer gave KL = 0, i.e. truly on-policy RL. Without importance weighting, reward collapsed.
- **LLM-42** (2026-01): deterministic inference via verify-rollback "verified speculation" that reuses existing kernels. Overhead scales with the fraction of traffic requiring determinism. Source: [arXiv 2601.17768](https://arxiv.org/abs/2601.17768)
- **Self-consistency as variance reduction:** DeepConf@512 reaches 99.9% on AIME 2025 with up to 84.7% fewer tokens than full parallel voting. Source: [arXiv 2508.15260](https://arxiv.org/abs/2508.15260)
- **Reasoning hurts abstention:** reasoning tuning lowers abstention about 24% (AbstentionBench). Source: [arXiv 2506.09038](https://arxiv.org/abs/2506.09038v1)

### Inferences
- **Reported numbers are not deterministic quantities.** Every 2026 frontier card evaluates at temperature 1.0, often avg@3, with 100K–384K-token outputs. Combined with the 9%/9,000-token hardware-induced spread, a deterministic reasoning component needs four things:
  1. greedy or seeded decoding;
  2. batch-invariant or FP32-compute kernels;
  3. pinned hardware, parallelism and batch shapes;
  4. output canonicalization (extract only the final answer), measured with TARa@N.
- **Lock down model and runtime, not just sampling.** Quantization shifts results (Nemotron Bolt GPQA −2.4 at IQ2), so the deployed artifact must be fixed. Multi-sample voting (DeepConf, maj@k) converts path divergence into a stable answer distribution at extra cost.
- **Ensure on-policy consistency during RLVR training.** Thinking Machines' KL = 0 result and Tongyi's "strictly on-policy" emphasis indicate sampler–trainer mismatch is a real failure mode for GRPO-family training.

### Gaps
- Primary AIME 2025/2026 numbers for the 2026 frontier open models were not on their cards; cards emphasize HLE, GPQA and agentic benchmarks.
- No primary LiveCodeBench numbers for Kimi K3 or GLM-5.3 were found.
- No primary 2026 BrowseComp scores were retrieved for open deep-research specialists newer than Tongyi DeepResearch.
- No dedicated table-reasoning or data-analysis benchmark leaders were found for 2026. SpreadsheetBench 2 (Kimi K3 34.8) and OfficeQA Pro (63.3) appear on the Kimi K3 card only.

---

## Q2. Cross-family consistency issues and documented mitigations

### Takeaway
Inconsistency has three separable layers:
1. **Numerical/serving nondeterminism.** Batch-size-dependent reductions; fixed by batch-invariant kernels, FP32 compute, or verified speculation.
2. **Sampling and path divergence.** Fixed by greedy or seeded decoding plus answer canonicalization and voting.
3. **Model-level brittleness**, e.g. prompt-format sensitivity up to 76 points; FC-vs-prompt interface swings up to 44 points; judge position, style and self-preference bias; pass^1→pass^4 drops of 10–30 points in agent tasks.

Layers 1 and 2 are now engineering-solvable. Layer 3 requires specialization, fixed interfaces and consistency-aware evaluation (pass^k, TARa@N, both-order judging).

### Cited Findings
- **Layer 1, numerical/serving:**
  - Temperature 0 is not deterministic: 80 unique completions in 1,000 runs, and batch invariance gives 1 unique. Source: [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
  - bf16 greedy decoding: 9% accuracy and 9,000-token variation across hardware and batch configurations. Source: [arXiv 2506.09501](https://arxiv.org/abs/2506.09501)
  - Verified speculation avoids rewriting kernels. Source: [LLM-42](https://arxiv.org/abs/2601.17768)
- **Layer 1/2, run-to-run accuracy:** up to 15% variation over 10 runs; no model consistently repeatable. Source: [Atil et al.](https://arxiv.org/abs/2408.04667)
- **Layer 3, format:**
  - Up to 76 points from prompt formatting. Source: [Sclar et al.](https://arxiv.org/abs/2310.11324)
  - BFCL v4 format-sensitivity max delta up to 81.5 (ToolACE-2-8B, Phi-4) vs 8.0 for Qwen3-235B. Source: [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)
- **Layer 3, tool-call reliability across trials:**
  - τ²-bench pass^1→pass^4, e.g. Claude Opus 4.5 retail 79.6→51.8; Qwen3.5-397B telecom 97.8→92.1. Source: [taubench.com](https://taubench.com/) / [submission JSON](https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions/claude-opus-4-5_sierra_2026-02-26/submission.json)
  - τ²-bench pass^k mixes agent variance with user-simulator variance, which the paper does not fully separate (commentary). Source: [beancount.io research log](https://beancount.io/bean-labs/research-logs/2026/06/18/tau-squared-bench-dual-control-conversational-agents)
- **Layer 3, judges:**
  - Position bias depends on the quality gap. Source: [Shi et al.](https://arxiv.org/abs/2406.07791)
  - Style bias 0.10–0.76 dominates. Source: [Soumik 2026](https://arxiv.org/abs/2604.23178)
  - Self-preference comes from perplexity familiarity. Sources: [Wataoka et al.](https://arxiv.org/abs/2410.21819); 31.5% mitigation in [arXiv 2604.22891](https://arxiv.org/abs/2604.22891)
  - Reward models fall below random under style interference. Source: [RM-Bench](https://arxiv.org/abs/2410.16184)
- **Layer 3, abstention:** reasoning models under-abstain. Source: [AbstentionBench](https://arxiv.org/abs/2506.09038v1)
- **Layer 3, creative homogenization:** Sources: [Verbalized Sampling](https://arxiv.org/abs/2510.01171), [NoveltyBench](https://arxiv.org/abs/2504.05228), [Artificial Hivemind](https://arxiv.org/abs/2510.22954)
- **Benchmark-level noise:**
  - BFCL v4 has 48% defective sampled tasks. Source: [Epoch AI](https://epoch.ai/benchmarks/berkeley-function-calling-leaderboard/review)
  - EQ-Bench switched judges on 2026-03-01. Source: [eqbench about](https://eqbench.com/about.html)
  - τ²-bench banking re-graded 2026-07-15. Source: [submission JSON](https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions/claude-opus-4-5_sierra_2026-02-26/submission.json)

### Inferences
- A "model compiler" should define correctness at three levels and measure each:
  1. bitwise reproducibility (Layer 1);
  2. answer-level agreement, TARa@N (Layer 2);
  3. task-level reliability, pass^k and both-order judge agreement (Layer 3).
- **Moving benchmarks are not fixed acceptance tests.** Leaderboards change judges and graders mid-year, so pinned internal test suites are needed.

### Gaps
- No study was found that quantifies how much of pass^k degradation disappears under fully deterministic serving. It is unclear whether agent inconsistency is mostly sampling noise or genuine policy instability.
- No measurement was found of MoE routing flips (expert-selection changes) as a source of nondeterminism.

---

## Q3. Specialist small models vs large generalists (cost/latency/accuracy) and evidence that specialization improves consistency

### Takeaway
Across every family there is primary evidence that 0.6B–30B specialists match or beat much larger generalists on the narrow task:
- RMs: Skywork-Reward-V2 8B RB2 84.1 vs GPT-4o 64.9.
- Tool-calling: xLAM-2-70b multi-turn 77.4 vs Opus 4.5 68.4; Nanbeige4-3B ≈ GPT-5-nano overall.
- Creative judging: LitBench RMs 78% vs 73%.
- Writing: WritingBench 7B > GPT-4o.
- Orchestration: Orchestrator-8B HLE 37.1 vs GPT-5 35.1 at 2.5x efficiency.
- Kaggle LMSYS: a 9B student distilled from 70B teachers won.

Costs are often 10–20x lower (GLM-4.6 $4.64 vs Opus 4.5 $86.55 on BFCL).

Evidence that specialization improves consistency is mixed. Specialists have a narrow competence envelope (near-zero web-search and memory), and some are extremely format-brittle outside their native template (ToolACE-2-8B format delta 81.5). Where an interface is fixed, specialist and strong open models show smaller pass^1→pass^k drops.

### Cited Findings
- **BFCL v4** (retrieved 2026-10-21, data as of 2026-04-12). Source: [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)
  - GLM-4.6: 72.38% at $4.64, 4.34s, vs Claude-Opus-4.5 77.47% at $86.55, 4.38s.
  - Nanbeige4-3B-Thinking (3.9B): 51.40% vs GPT-5-nano (FC) 51.45%, and above Qwen3-32B (FC) 48.71%.
  - xLAM-2-70b: multi-turn 77.38%, best of any model.
  - Specialists' agentic collapse: Hammer2.1-7b web-search/memory 0%/0%; BitAgent-Bounty-8B web-search 0%; xLAM-2-8b web-search 6.5%.
- **RMs:** Skywork-Reward-V2-Qwen3-1.7B averages 75.2 vs INF-ORM-Llama3.1-70B 73.5. The Llama-3.1-8B variant posts RB2 84.1 vs GPT-4o 64.9 and Claude-3.5-Sonnet 64.7. Source: [HF card](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)
- **Creative judging:** trained RMs 78% vs zero-shot Claude-3.7 73%. Source: [LitBench](https://arxiv.org/abs/2507.00769)
- **Writing:** 7B model beats GPT-4o. Source: [WritingBench](https://arxiv.org/abs/2503.05244)
- **Orchestration:** Orchestrator-8B scores HLE 37.1% vs GPT-5 35.1% at 2.5x efficiency, and beats GPT-5 on τ²-bench and FRAMES at about 30% of the cost. Training is RL with outcome, efficiency and user-preference rewards. Source: [ToolOrchestra, arXiv 2511.21689](https://arxiv.org/abs/2511.21689)
- **Deep research:** Tongyi DeepResearch-30B-A3B (3B active) beats o3 on HLE (32.9 vs 24.9) but trails on BrowseComp (43.4 vs 49.7). Source: [thread](https://threadreaderapp.com/thread/1967988004179546451.html)
- **Decision:** the Kaggle LMSYS winner distilled 70B/72B teachers into Gemma2-9B. Source: [Nebius](https://nebius.com/blog/posts/chatbot-arena-competition-review)
- **LoRA Land (2024):** 310 LoRA models (10 bases × 31 tasks). 4-bit LoRA fine-tunes beat their bases by 34 points and GPT-4 by 10 points on average. LoRAX serves 25 Mistral-7B adapters on one A100-80GB. Source: [arXiv 2405.00732](https://arxiv.org/pdf/2405.00732)
- **NVIDIA position paper (2025-06):** SLMs are "sufficiently powerful, inherently more suitable, and necessarily more economical" for repetitive agentic sub-tasks. Source: [arXiv 2506.02153](https://arxiv.org/pdf/2506.02153)
  - The claim that 40–70% of agent LLM calls are replaceable comes from a secondary summary ([Maxim](https://www.getmaxim.ai/blog/are-small-language-models-the-future-of-agentic-ai/)) and is unverified.
- **Consistency comparison on τ²-bench (pass^1 → pass^4).** Simulators differ, so this is only indicative. Source: [taubench.com](https://taubench.com/)

| Model | Domain | pass^1 → pass^4 | Drop | User simulator |
|---|---|---|---|---|
| RAFT-30B-A3B (custom) | retail | 82.5 → 61.4 | 21.1 | Claude Sonnet 4.5 |
| Claude Opus 4.5 | retail | 79.6 → 51.8 | 27.8 | gpt-5.2 |
| GPT-5.2 | retail | 81.6 → 51.8 | 29.8 | gpt-5.2 |
| Qwen3.5-397B (open) | telecom | — | 5.7 | gpt-5.2 |
| Claude Opus 4.5 | telecom | — | 14.2 | gpt-5.2 |

- **Format brittleness of specialists** (BFCL format sensitivity): ToolACE-2-8B max delta 81.5 / SD 27.92 vs Qwen3-235B 8.0 / 1.95. Source: [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)
- **Robustness-oriented specialist training:** Hammer explicitly targets cross-benchmark inconsistency caused by naming conventions, via function masking. Source: [arXiv 2410.04587](https://arxiv.org/abs/2410.04587)

### Inferences
- **Specialization improves consistency when the deployment interface is frozen** (same template, same schema style, same domain). That is exactly the "compiler" setting.
- **Specialization degrades consistency under distribution shift** (format, naming, agentic categories), so a router or gate must detect out-of-envelope inputs and escalate to a generalist or abstain.
- **The cost gap (often 10–20x) leaves budget for consistency mechanisms:** k-sample voting or verifier-reranking with a specialist can still undercut one generalist call.

### Gaps
- There is no controlled study (same task, same serving stack) directly comparing run-to-run variance (TARa@N) of a fine-tuned small specialist vs a large generalist.
- The τ²-bench comparisons above are confounded by different user simulators and scaffolds.

---

## Q4. Expert-specialization architectures: separate fine-tuned models vs LoRA adapters on a shared base vs task-specialized MoE experts

### Takeaway
**LoRA on a shared base is the best-supported route for task specialists.**
- It matches full fine-tuning on small-to-medium SFT sets.
- It matches full fine-tuning even at rank 1 for RL.
- It needs about a 10x learning rate and must cover MLP/MoE layers.
- It enables multi-tenant serving: LoRAX, vLLM and SGLang can serve dozens of adapters on one GPU.

**Separate full models** (xLAM, Skywork-RM, Tongyi DR) remain common when the base must differ or when capacity exceeds LoRA's limits (large datasets, large batches).

**Task-specialized MoE experts:**
- Branch-Train-MiX trains domain experts in parallel and then learns a router; it claims the best accuracy-efficiency trade-off.
- Production frontier MoEs (Kimi K3: 896 experts, 16 active; Qwen3.8: 512 experts, 10+1 active) use fine-grained token-level routing, not task-level experts.
- No 2026 evidence was found that their experts align to the five task families.

### Cited Findings
- **LoRA Without Regret** (Schulman / Thinking Machines, 2025-09-29). Source: [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
  - LoRA matches full fine-tuning loss on Tulu3/OpenThoughts3-scale SFT when the learning rate is tuned.
  - It degrades gradually once the dataset exceeds adapter capacity, and pays a penalty at large batch sizes that rank does not fix.
  - In policy-gradient RL, rank 1 matches full fine-tuning (about 1 bit of information per episode).
  - Optimal learning rate is about 9.8x the full-fine-tuning rate, nearly rank-independent.
  - All-layer LoRA, especially MLP/MoE, beats attention-only LoRA even at equal parameter count.
  - LoRA uses about 2/3 of full-fine-tuning FLOPs, and vLLM/SGLang can batch many adapters on one server.
- **LoRA Land:** 310 task adapters with an average +10 over GPT-4; LoRAX serves 25 adapters on one A100. Source: [arXiv 2405.00732](https://arxiv.org/pdf/2405.00732)
- **Adapter merging:** the Kaggle LMSYS winner averaged LoRA weights of five fold-distilled Gemma2-9B models into one. Source: [Nebius](https://nebius.com/blog/posts/chatbot-arena-competition-review)
- **Branch-Train-MiX** (2024-03): copy a seed model, train domain experts (code, math, world knowledge) independently, use their FFNs as MoE experts, average the other weights, then fine-tune the router. It generalizes Branch-Train-Merge and sparse upcycling, with a claimed best accuracy-efficiency trade-off. Source: [arXiv 2403.07816](https://arxiv.org/abs/2403.07816)
- **OLMoE** (1B active / 7B total, fully open) reports "high specialization" in routing analysis (abstract only; details not retrieved). Source: [arXiv 2409.02060](https://arxiv.org/abs/2409.02060)
- **2026 production MoE configurations:**
  - Kimi K3: 896 experts, 16 selected plus 2 shared, 104B active of 2.8T. Source: [HF](https://huggingface.co/moonshotai/Kimi-K3)
  - Qwen3.8-2.4T-A95B: 512 experts, 10 routed plus 1 shared. Source: [HF](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
  - DeepSeek-V4.1-Flash: 8B/16B active, plus a 196B Engram lookup memory. Source: [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- **Vendors ship one MoE with mode switches rather than separate task experts.** Examples: MiniMax-M3 enabled/adaptive/disabled thinking ([HF](https://huggingface.co/MiniMaxAI/MiniMax-M3)); GLM-5.3-Flash `reasoning_effort` low/high/max ([HF](https://huggingface.co/zai-org/GLM-5.3-Flash)); Qwen3.8 thinking/instruct ([HF](https://huggingface.co/Qwen/Qwen3.8-27B)).

### Inferences
- For a "deterministic MoE compiler," the most evidence-backed architecture is one shared open base with per-family LoRA adapters (decision head, FC, judge, creative, reasoning) on a multi-LoRA server.
  - Each adapter can be paired with family-specific decoding policy: greedy plus grammar for FC and decision; seeded best-of-N plus RM for creative; maj@k for reasoning.
  - Use separate full models only where a different base is clearly superior. Examples: an 8B BT reward model; a frontier MoE for deep reasoning.
- **Task-level MoE (BTX-style) is plausible but unproven at 2026 frontier scale.** Fine-grained token routing means one cannot rely on "the FC expert" existing inside Kimi K3 or Qwen3.8. Task specialization must be imposed externally (adapters, a router over models) or via BTX-style construction.
- **Batch invariance must cover adapter batching.** Multi-LoRA batching mixes requests for different adapters in one batch, which compounds Thinking-Machines-style batch non-invariance.

### Gaps
- No primary 2025–2026 study was found measuring whether experts in large production MoEs specialize by task family (vs token, syntax or vocabulary). The search budget ran out before this could be searched.
- No benchmark directly comparing determinism (bitwise or TARa@N) of multi-LoRA serving vs separate model deployments was found.
- No quantitative BTX benchmark numbers were retrieved (abstract only).

# Closing the gap: raising Qwen3.6-35B-A3B toward Qwen3.8-27B, and training goal-driven agentic models (state as of 2026-10-21)

Conventions: [date] = publication/release date of the source or model. "n/r" = not reported by that source. Vendor numbers are self-reported unless marked independent. Items marked "(not re-verified this session)" come from well-known primary sources whose pages were not re-fetched here; treat them as lower confidence.

---

## Q1. Benchmarks: Qwen3.6-35B-A3B vs Qwen3.8-27B and nearby models (official + independent, incl. tokens/cost)

### Takeaway
On vendor cards, Qwen3.8-27B (dense, Aug 2026) leads Qwen3.6-35B-A3B (3B-active MoE, Apr 2026) by about 3 pts on GPQA, about 9-10 pts on HLE and LiveCodeBench v6, and about 12 pts on SWE-bench Pro. Independent Artificial Analysis (AA) data shows a much larger agentic gap: Intelligence Index v4.3.2 is 34 vs 18, GDPval-AA 1417 vs 892, and AutomationBench-AA 48.2% vs 5.2%. Qwen3.8-27B used more output tokens (200M vs 160M) and cost about 2.1x per task ($1.01 vs $0.48). Same-generation pairs, where only the architecture differs, show that most of the reasoning gap comes from post-training generation (3.6→3.8), not from MoE vs dense. The agentic gap has a large architecture/capacity component as well.

### Cited Findings

**Model facts**
- Qwen3.6-35B-A3B [released 2026-04-16]: 35B total / 3B active; 40 layers, hidden 2048; layout 10×(3×Gated DeltaNet→MoE, 1×Gated Attention→MoE); 256 experts, 8 routed + 1 shared, expert FFN 512; 262,144 native context (YaRN to ~1.01M); vocab 248,320 (padded); multi-step MTP; `preserve_thinking` option trained for agent scenarios — [HF card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B); release date from [QwenLM/Qwen3.8 README](https://github.com/QwenLM/Qwen3.8) and [AA FAQ](https://artificialanalysis.ai/models/qwen3-6-35b-a3b)
- Qwen3.8-27B [released 2026-08-14]: dense 27B, 64 layers, hidden 5120, 16×(3×Gated DeltaNet→FFN, 1×Gated Attention→FFN), FFN 17,408, vocab 248,320 (padded), 262,144 native context, MTP. Default `reasoning_effort` = `xhigh`. The card says post-training targets agentic tasks, tool use and thinking control, but gives no RL or distillation details — [HF card](https://huggingface.co/Qwen/Qwen3.8-27B); date from [QwenLM/Qwen3.8 README](https://github.com/QwenLM/Qwen3.8)
- Qwen3.6-27B [released 2026-04-22]: same dense architecture as Qwen3.8-27B — [HF card](https://huggingface.co/Qwen/Qwen3.6-27B); [README](https://github.com/QwenLM/Qwen3.8)
- Qwen3.8-Flash-Next [2026-08-26, open weights]: 125B total / 6B active MoE, plus 51B n-gram embedding parameters held in host RAM and 4B MTP; 48 layers, hidden 2560; 512 experts, 10 routed + 1 shared, expert FFN 640; Gated DeltaNet + Qwen Sparse Attention — [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); [arXiv 2608.30320, 2026-08-31](https://arxiv.org/abs/2608.30320); date from [DataCamp](https://www.datacamp.com/blog/qwen3-8-flash-next) (secondary)
- DeepSeek-V4-Flash [preview paper published 2026-04-26; 0731 update 2026-07-31]: 284B total / 13B active MoE; CSA+HCA hybrid sparse attention; 1M context; FP4 experts + FP8; >32T pretraining tokens; Muon optimizer; reasoning modes Non-Think / High / Max — [HF card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash); [AA page](https://artificialanalysis.ai/models/deepseek-v4-flash)
- gpt-oss-20b [2025-08]: 21B total / 3.6B active; low/medium/high reasoning effort — [HF card](https://huggingface.co/openai/gpt-oss-20b)
- Qwen3-30B-A3B-Thinking-2507 [2025-07]: reasoning/coding evals use 81,920 output tokens — [HF card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)

**Official (vendor) numbers, side by side.** Sources: [Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B), [Qwen3.8-Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [Qwen3-30B-A3B-Thinking-2507 card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507), [gpt-oss model card arXiv 2508.10925](https://arxiv.org/html/2508.10925v1), [DeepSeek-V4-Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

| Benchmark | Qwen3.6-35B-A3B | Qwen3.6-27B | Qwen3.8-27B | Qwen3.8-Flash-Next (A6B) | Qwen3-30B-A3B-Think-2507 | gpt-oss-20b (high) | DeepSeek-V4-Flash (Max / High) |
|---|---|---|---|---|---|---|---|
| GPQA-Diamond | 86.0 | 87.8 | 89.2 | 91.7 | 73.4 | 71.5 (no tools) / 74.2 (tools) | 88.1 / 87.4 (0731 per Qwen card: 90.8) |
| HLE | 21.4 | 24.0 | 30.8 | 35.9 | n/r | 10.9 (no tools) / 17.3 (tools) | 34.8 / 29.4 (0731: 33.8) |
| AIME 2026 | 92.7 | 94.1 | n/r | n/r | n/r | n/r | n/r (HMMT Feb-26: 94.8 / 91.9) |
| AIME 2025 | n/r | n/r | n/r | n/r | 85.0 | 91.7 (no tools) / 98.7 (tools) | n/r |
| LiveCodeBench v6 | 80.4 | 83.9 | 90.3 | 91.9 | 66.0 (25.02–25.05) | n/r | 91.6 / 88.4 (0731: 90.6) |
| MMLU-Pro | 85.2 | 86.2 | n/r | n/r | 80.9 | n/r (MMLU 85.3) | 86.2 / 86.4 |
| SWE-bench Verified | 73.4 | 77.2 | n/r | n/r | n/r | 60.7 | 79.0 / 78.6 |
| SWE-bench Pro | 49.5 | 53.5 | 61.7 | 62.5 | n/r | n/r | 52.6 / 52.3 (0731, Qwen-run: 56.0) |
| SWE-bench Multilingual | 67.2 | 71.3 | 73.8 (from Flash-Next card) | 81.0 | n/r | n/r | n/r |
| Terminal-Bench 2.0 | 51.5 | 59.3 | n/r | n/r | n/r | n/r | 56.9 / 56.6 |
| Terminal-Bench 2.1 (Terminus) | n/r | 63.4 | 73.0 | n/r | n/r | n/r | n/r |
| BFCL | n/r | n/r | n/r | n/r | BFCL-v3 72.4 | n/r | n/r |
| τ²-bench | n/r (TAU3-Bench 67.2) | n/r | n/r | n/r | Retail 58.8 / Airline 58.0 / Telecom 26.3 | τ-bench Retail 54.8 | n/r |
| BrowseComp | n/r | n/r | n/r | n/r | n/r | n/r | 73.2 / 53.5 |
| NL2Repo | 29.4 | 36.2 | 42.3 | 48.1 | n/r | n/r | 54.2 (0731) |
| Other agentic | MCPMark 37.0; MCP-Atlas 62.8; Claw-Eval avg 68.7 / pass^3 50.0; SkillsBench 28.7; DeepPlanning 25.9; Tool Decathlon 26.9; WideSearch 60.1 | SkillsBench 48.2; Claw-Eval avg 72.4 / pass^3 60.6; DeepSWE-1.1 13.3 | DeepSWE-1.1 42.2; CoWorkBench 70.7; JobBench 33.4; Toolathlon-Verified 67.1; IFBench 79.5 | DeepSWE-1.1 58.7; CoWorkBench 73.9; JobBench 55.7; Toolathlon-Verified 73.5 | — | — | Toolathlon 47.8 / 43.5 |

- Same-generation MoE vs dense (Qwen3.6-35B-A3B vs Qwen3.6-27B, same card): GPQA −1.8, HLE −2.6, LCB v6 −3.5, AIME26 −1.4, MMLU-Pro −1.0, SWE-V −3.8, SWE-Pro −4.0, SWE-Multilingual −4.1, Terminal-Bench 2.0 −7.8, NL2Repo −6.8, Claw-Eval pass^3 −10.6, SkillsBench −19.5 — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)
- Same dense architecture, one generation apart (Qwen3.6-27B → Qwen3.8-27B): GPQA +1.4, HLE +6.8, LCB v6 +6.4, SWE-Pro +8.2, Terminal-Bench 2.1 +9.6, NL2Repo +6.1, DeepSWE-1.1 +28.9, IFBench +10.4, CoWorkBench +9.7, JobBench +11.6 — [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)
- The previous-generation pair shows the same pattern. Qwen3.5-35B-A3B vs Qwen3.5-27B: GPQA 84.2 vs 85.5, LCB 74.6 vs 80.7, SWE-V 70.0 vs 75.0, SkillsBench 4.4 vs 27.2. Moving Qwen3.5-35B-A3B → Qwen3.6-35B-A3B gained SWE-Pro +4.9, TB2.0 +11.0, SkillsBench +24.3 — [Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- Previous-generation evidence [2025-05]: Qwen3-30B-A3B (thinking) vs Qwen3-32B: AIME'24 80.4 vs 81.4, AIME'25 70.9 vs 72.9, LCB v5 62.6 vs 65.7, GPQA 65.8 vs 68.4, BFCL v3 69.1 vs 70.3 — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)

**Evaluation-setting caveats (affect comparability)**
- Qwen3.6-35B-A3B SWE-bench series: internal agent scaffold (bash + file-edit), temp 1.0, top_p 0.95, 200K context. Some SWE-bench Pro tasks were corrected ("refined version"). Terminal-Bench 2.0: Harbor/Terminus-2, 3-h timeout, 32 CPU / 48 GB RAM, max 80K tokens, 256K context, average of 5 runs — [card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- Qwen3.8-27B SWE-bench Pro / DeepSWE / NL2Repo: Claude Code harness, temp 1.0, 256K context, refined SWE-bench Pro task set. Opus 4.6 Max uses Anthropic's own published score rather than a re-run. HLE is judged by GPT-4o — [card](https://huggingface.co/Qwen/Qwen3.8-27B). Qwen3.6-27B shows SWE-bench Pro 53.5 on both the 3.6 and 3.8 cards, so the two harnesses may give similar numbers for that model. This is an inference, not a stated equivalence.
- TAU3-Bench uses gpt-5.2 (low effort) as the user model with BM25 retrieval. MCPMark uses GitHub MCP v0.30.3. MCP-Atlas uses the public set, judged by gemini-2.5-pro — [Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- Terminal-Bench versions differ (2.0 on Qwen3.6 cards vs 2.1 on Qwen3.8 cards). Qwen3.6-27B scores 59.3 on 2.0 and 63.4 on 2.1, so the two are not interchangeable — [3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B); [3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)
- DeepSeek-V4-Flash numbers differ by version and harness. Preview card, Max mode: SWE-Pro 52.6, GPQA 88.1. The Qwen3.8 cards list "DeepSeek-V4-Flash-0731" at SWE-Pro 56.0 and GPQA 90.8 under Qwen's harness — [DeepSeek card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) vs [Qwen3.8-Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- gpt-oss-20b HF page eval widgets show GPQA-Diamond 58.59 / 66 / 67.1 and SWE-bench Verified 53.2 at medium effort. These conflict with the paper's high-effort figures (71.5; 60.7) because the reasoning levels differ — [HF card](https://huggingface.co/openai/gpt-oss-20b) vs [arXiv 2508.10925](https://arxiv.org/html/2508.10925v1)

**Independent evaluation: Artificial Analysis (Intelligence Index v4.3.2, 10 evals; page accessed 2026-10-21)** — [AA release comparison Qwen3.8 27B vs Qwen3.6 35B A3B](https://artificialanalysis.ai/models/releases/comparisons/qwen3-8-27b-vs-qwen3-6-35b-a3b)

| AA metric | Qwen3.8-27B xhigh | Q3.8-27B medium | Q3.8-27B low | Q3.8-27B non-reasoning | Qwen3.6-35B-A3B (reasoning) | Q3.6-35B-A3B non-reasoning |
|---|---|---|---|---|---|---|
| Intelligence Index | 34 | 28 | 26 | 20 | 18 | 15 (estimated) |
| Cost per task (USD) | 1.01 | 1.13 | 1.05 | 2.49 | 0.48 | — |
| Time per task (s) | 1,184 | 972 | 965 | 618 | 227 | — |
| Output speed (tok/s, Alibaba API) | 44 | 46 | 43 | 46 | 134 | 148 |
| HLE | 33.9% | 14.1% | 14.0% | 12.1% | 22.2% | 13.9% |
| AA-LCR v1.1 | 82.0% | 79.7% | 77.3% | 69.3% | 71.7% | 64.3% |
| SciCode | 46.6% | 39.0% | 40.0% | 36.2% | 36.6% | — |
| GDPval-AA v2.1 (Elo) | 1417 | 1382 | 1363 | 1072 | 892 | 847 |
| AA-Briefcase v1.1 | 1395 | 1375 | 1272 | 898 | 713 | — |
| AutomationBench-AA | 48.2% | 40.3% | 30.1% | 12.4% | 5.2% | — |
| GDP.pdf | 16.4% | 12.8% | 10.4% | 5.2% | 6.6% | — |
| AA-Omniscience | −10 | −36 | −27 | −8 | −22 | −60 |
| CritPt | 5.4% | 0.0% | 0.0% | 0.3% | 0.3% | 0.0% |
| "Terminal-Bench 4.0" (AA label) | 5.6% | 5.1% | 2.5% | 0.0% | 0.0% | — |

- Output tokens to run the AA Intelligence Index: Qwen3.8-27B (xhigh) 200M vs comparable-model median 82M; prices $0.50 in / $3.00 out per 1M, 80% cache discount — [AA Qwen3.8 27B](https://artificialanalysis.ai/models/qwen3-8-27b). Qwen3.6-35B-A3B (reasoning) 160M vs median 82M; $0.375 in / $2.25 out — [AA Qwen3.6 35B A3B](https://artificialanalysis.ai/models/qwen3-6-35b-a3b)
- gpt-oss-20b (high): Index 9 (#50/141), 70M output tokens, $0.01 per task, 252 tok/s, $0.07 / $0.25 per 1M — [AA gpt-oss-20b](https://artificialanalysis.ai/models/gpt-oss-20b)
- DeepSeek-V4-Flash-0731 (Max): Index 34 (#14/121), 240M output tokens (comparable-model median 140M), $0.17 per task, $0.14 / $0.28 per 1M, 144 tok/s. AA now marks it deprecated in favor of DeepSeek V4.1 Flash — [AA DeepSeek V4 Flash](https://artificialanalysis.ai/models/deepseek-v4-flash)
- Qwen3.8-Flash-Next: AA Intelligence Index 56, reported only by a secondary review and not verified on AA here — [eesel.ai](https://www.eesel.ai/blog/qwen38-flash-next-review)
- **Conflict (AA index versions):** an AA search snippet and older localized AA pages give Qwen3.6-35B-A3B 32 points, 150M tokens and $332.83 to run the full index. Another snippet shows the comparison as 52 (Qwen3.8-27B) vs 32. Current English pages and the v4.3.2 comparison show 34 vs 18. These are most likely different index versions, so use the v4.3.2 figures and do not mix versions — [AA PT/DE pages](https://artificialanalysis.ai/pt/models/qwen3-6-35b-a3b); [AA comparison](https://artificialanalysis.ai/models/releases/comparisons/qwen3-8-27b-vs-qwen3-6-35b-a3b)
- BenchLM (secondary) lists AA-GPQA Diamond at 84.1% for Qwen3.6-35B-A3B and 84.2% for Qwen3.6-27B, below the vendor card's 86.0 / 87.8 — [BenchLM compare](https://www.benchlm.ai/compare/qwen3-6-27b-vs-qwen3-6-35b-a3b)

**Tokens per task (where reported)**
- DeepSeek-V3.2 [2025-12] gives average output tokens per benchmark: AIME25 93.1 (16k tokens); HMMT Feb-25 92.5 (19k); GPQA 82.4 (7k); HLE 25.1 (21k); LCB 83.3 (16k); Codeforces 2386 (42k). V3.2-Speciale uses about 1.4–2.3x more tokens, e.g. AIME25 96.0 (23k) and HLE 30.6 (35k) — [DeepSeek-V3.2 paper](https://arxiv.org/html/2512.02556v1)
- gpt-oss-20b uses "over 20k CoT tokens per problem on average for AIME" — [arXiv 2508.10925](https://arxiv.org/html/2508.10925v1)
- Qwen3.8 cards warn that lower `reasoning_effort` "does not always reduce overall task completion time" in multi-turn agentic tasks, because shallow analysis causes failures and retries. Suggested output limits are 262,144 reasoning tokens and 131,072 answer tokens — [Qwen3.8-Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

### Inferences
- Gap decomposition on overlapping vendor benchmarks, using Qwen3.6-27B as the bridge:
  - Qwen3.6-35B-A3B → Qwen3.8-27B: GPQA −3.2, HLE −9.4, LCB −9.9, SWE-Pro −12.2.
  - Of that, the MoE-vs-dense part (3.6-A3B vs 3.6-27B) is −1.8 / −2.6 / −3.5 / −4.0.
  - The post-training-generation part (3.6-27B → 3.8-27B) is −1.4 / −6.8 / −6.4 / −8.2.
  - So about 2/3 of the reasoning/coding gap is attributable to the newer post-training recipe, which in principle can be transferred (via distillation/RL) to the same 35B-A3B backbone.
- The gap on AA agentic work-tasks is far larger than on vendor reasoning benchmarks: AutomationBench 5.2% vs 48.2%, GDPval 892 vs 1417. Qwen3.8-27B at *low* effort (26) and even non-reasoning mode (20) beat Qwen3.6-35B-A3B reasoning (18). The agentic deficit is therefore not primarily a "thinking budget" problem; it looks like a post-training/agentic-data problem.
- Qwen3.6-35B-A3B is not more verbose than Qwen3.8-27B on AA (160M vs 200M tokens). It is about 3x faster per token (134 vs 44 tok/s) and about 5x faster per task (227 s vs 1,184 s). Closing the gap will probably require *more* tokens per task on agentic work (longer horizons, more turns). Plan for the cost per task to rise toward the 27B's, offset by roughly 3x cheaper decode.
- Both models use the same 248,320-entry vocabulary and the same Gated-DeltaNet/Gated-Attention family. Direct token-level logit distillation from Qwen3.8-27B into Qwen3.6-35B-A3B therefore needs no vocabulary mapping. This is inferred from both cards and is not stated by Qwen.
- Qwen3.8-Flash-Next (125B-A6B, released open) already beats Qwen3.8-27B on every shared vendor benchmark. If the goal is "3B-6B active at 27B quality", it is a plausible alternative student or teacher. Its active parameters are 2x those of the 35B-A3B, and its 125B total plus 51B n-gram tables make it much larger in memory.

### Gaps
- No official BFCL, τ²-bench (only TAU3), or BrowseComp numbers were found for Qwen3.6-35B-A3B or Qwen3.8-27B. Qwen3.8-27B reports no AIME, MMLU-Pro or SWE-bench Verified.
- AA pages did not expose per-eval GPQA, IFBench or τ²-Telecom values for these models under v4.3.2. AA's "Terminal-Bench 4.0" label is unexplained.
- No Epoch AI or LiveBench entries were found for Qwen3.6-35B-A3B or Qwen3.8-27B. No per-task token counts were found for individual agentic benchmarks (SWE-bench, Terminal-Bench) for either model.
- No official AA page was verified for Qwen3.6-27B, Qwen3-30B-A3B-2507 or Qwen3.8-Flash-Next under v4.3.2.

---

## Q2. Distillation recipes for reasoning, and how close a small-active MoE student can get to a dense teacher

### Takeaway
The strongest evidence favors **on-policy distillation (OPD)**: the student samples, the teacher scores every token, and the loss is per-token reverse KL. On a Qwen3-8B student it beat RL at about 1/10 the GPU-hours (74.4 vs 67.6 AIME'24; 1,800 vs 17,920 GPU-h) and matched SFT-on-2M-samples at 9–30x lower compute. In 2026, leading labs use OPD as the *merge/consolidation* step after training domain experts with RL: DeepSeek-V4, GLM-5 and Qwen3-Coder-Next all do this. Same-generation Qwen pairs show a 3B-active MoE landing within about 1–4 pts of its dense sibling on reasoning, but 8–20 pts behind on harder agentic suites.

### Cited Findings

**On-policy distillation (Thinking Machines) [2025-10-27]** — [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Method: sample student rollouts. Compute teacher log-probs on the student's tokens. Use advantage = −(per-token reverse KL) inside an RL importance-sampling loss with discount 0 (next-token only). Discounts above 0 did not help.
- Qwen3-8B-Base student, Qwen3-32B teacher (Qwen3-8B was used in practice because it was slightly better):
  - SFT on 400K OpenThoughts-3 prompts reaches 60% AIME'24; extrapolated SFT needs about 2M prompts to reach 70%.
  - OPD reaches 70% in about 150 steps (about 77K prompts, 4 samples per prompt). Default batch is 64 prompts × 4 samples.
- FLOPs: SFT-2M about 3.4e21 teacher + 1.5e21 student; OPD 8.4e19 teacher + 8.2e19 student. That is a 9–30x cost reduction (about 18x in GPU-hours).
- Self-distillation from an RL teacher matched it in 7–10x fewer gradient steps (claimed 50–100x compute efficiency). The AIME score was recovered in under 10 steps vs 70 for RL.
- Continual learning: SFT on the model's own samples degraded IF-eval; OPD with a fixed teacher avoided the regression. In the personalization experiment, OPD restored IF-eval from 79% to 83% after mid-training on internal documents.
- LoRA rank 32 trails full fine-tuning by 6% after OPD (13% after SFT).
- When the teacher's policy is outside the student's support, as in reasoning, much larger batches are needed.

**Qwen's strong-to-weak distillation (Qwen3) [2025-05-14]** — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- Applied to 0.6B–14B dense models and to the **Qwen3-30B-A3B MoE**, with teachers Qwen3-32B or Qwen3-235B-A22B.
- Phase 1, off-policy: SFT on teacher outputs in /think and /no_think modes.
- Phase 2, on-policy: the student generates, and its logits are aligned to the teacher's by minimizing KL.
- Table 21 (Qwen3-8B, from the same off-policy checkpoint):

| Method | AIME'24 | AIME'25 | LCB v5 | GPQA | GPU-h |
|---|---|---|---|---|---|
| Off-policy distillation | 55.0 | 42.8 | 42.0 | 55.6 | — |
| + RL | 67.6 | 55.5 | 52.9 | 61.3 | 17,920 |
| + On-policy distillation | 74.4 | 65.5 | 60.3 | 63.3 | 1,800 |

- On-policy distillation also raised pass@64 (AIME'24 90.0→93.3), while RL left it unchanged.
- Result for the MoE student: Qwen3-30B-A3B finished within 1.0–3.1 pts of Qwen3-32B on AIME'24/'25, LCB v5, GPQA and BFCL.

**2026 labs using OPD as the consolidation step**
- DeepSeek-V4 [2026-04]: domain experts trained independently with SFT + GRPO, then "a unified model is consolidated through on-policy distillation" — [DeepSeek-V4-Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- GLM-5 [2026-02]: "On-Policy Cross-Stage Distillation" is the final stage. Final checkpoints of earlier stages (Reasoning RL, General RL) act as teachers, with prompts sampled from each teacher's RL training set, to undo cumulative capability loss across sequential RL stages — [GLM-5 report arXiv 2602.15763](https://arxiv.org/pdf/2602.15763v2)
- DeepSeek-V3.2 [2025-12]: specialist distillation. Specialists for math, programming, logic, general agent, agentic coding and agentic search are trained from the same base, then distilled — [DeepSeek-V3.2 paper](https://arxiv.org/html/2512.02556v1)
- Qwen3-Coder-Next [2026-03] (80B total / ~3B active, a direct analogue of the target size class):
  - Four experts (WebDev, UX/tool-format, single-turn QA RL, SWE agentic RL) merged by "expert distillation" into one model with no routing.
  - Reaches SWE-bench Verified 70.6–71.3 across SWE-Agent, MiniSWE-Agent and OpenHands, comparable to GLM-4.7 (358B/32B active) and DeepSeek-V3.2 (671B/37B active).
  - AIME25 83.07 vs 69.64 for base Qwen3-Next.
  - [Qwen3-Coder-Next report](https://arxiv.org/html/2603.00729v1)

**Top-k logit distillation (relevant to the team's top-20 forward-KL, weight 0.3)**
- Caching only top-K teacher probabilities gives a biased estimator:
  - Renormalization inflates the probabilities of the top tokens.
  - KL on the truncated target pushes discarded tokens toward zero.
  - Students become over-confident or mis-calibrated.
- "Random Sampling KD" (importance-sampled sparse logits) is unbiased, adds under 10% overhead vs cross-entropy, and is competitive with full KD at 300M–3B — [Sparse Logit Sampling, ACL 2025, arXiv 2503.16870](https://arxiv.org/pdf/2503.16870)
- MiniMax uses a Top-K KL loss to keep MTP speculative-decoding heads tracking the evolving policy during RL. This is a production use of top-K KL as an auxiliary loss — [MiniMax Forge blog, 2026-02-13](https://minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)

**Sequence-level long-CoT distillation (SFT)**
- Scaling evidence: 400K OpenThoughts-3 prompts give 60% AIME'24 for Qwen3-8B-Base, and about 2M would be needed for about 70%. Pure SFT scales roughly log-linearly and is much more data-hungry than OPD — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- GLM-5 SFT keeps erroneous segments inside agentic trajectories but masks them out of the loss. The model learns error correction without imitating the errors — [GLM-5 report](https://arxiv.org/pdf/2602.15763v2)
- Tmax [2026-06] secondary report: SFT before RL *degraded* Qwen3.5 but helped Qwen3-8B on terminal tasks, so cold-start SFT is not universally beneficial for newer bases — [aiweekly summary](https://aiweekly.co/alerts/allen-ais-tmax-trains-open-terminal-agent-to-27-accuracy) (secondary; not verified in the paper)
- Qwen [2025-12]: once RL is stabilized, longer optimization yields similar final performance regardless of cold-start initialization — [arXiv 2512.01374](https://arxiv.org/abs/2512.01374)

### Inferences
- For a Qwen3.8-27B → Qwen3.6-35B-A3B transfer, the recipe most consistent with the evidence is:
  1. A short off-policy SFT on 27B traces, mainly to match format, `preserve_thinking` behavior and tool-call syntax.
  2. Then **on-policy reverse-KL distillation** on student rollouts, with full or importance-sampled teacher logits rather than renormalized top-20. Both share the 248,320 vocabulary.
  3. Then targeted RL on agentic environments where the 27B teacher is itself weak, or where outcome rewards exist.
- The team's existing top-20 forward-KL at weight 0.3 is an off-policy, mode-covering objective. For reasoning, the cited evidence favors switching to student-sampled reverse KL (mode-seeking) and to unbiased sparse-logit sampling.
- Expected ceiling: same-generation 3B-active vs dense gaps of about 1–4 pts on GPQA/AIME/LCB/SWE-V (Qwen3, Qwen3.5 and Qwen3.6 pairs) suggest distillation can close most of the *reasoning* gap. Agentic suites (SkillsBench −19.5, Claw-Eval pass^3 −10.6, AA AutomationBench) suggest a residual capacity/robustness gap that distillation alone may not close.
- Costing anchor: OPD on an 8B student took 1,800 GPU-h (Qwen) or about 150 steps × 256 sequences (Thinking Machines). For a 3B-active student and a 27B teacher at about 100K prompts × 4 samples × ~16K tokens (6.4B tokens), teacher prefill is about 2×27e9×6.4e9 ≈ 3.5e20 FLOPs and student training about 6×3e9×6.4e9 ≈ 1.2e20 FLOPs. Compute-bound this is a few hundred H100-hours; rollout and long-context overheads will push it into the low thousands. This is my arithmetic, not a sourced figure.

### Gaps
- No published experiment was found that distills a dense teacher into a *smaller-active MoE* of the same generation with an OPD-vs-RL ablation. Qwen3's 30B-A3B was distilled, but the report does not separate its OPD gains.
- OpenThoughts/AM-Thinking dataset sizes and results were not re-verified this session. Only the OpenThoughts-3 400K→60% data point (via Thinking Machines) is cited.
- No source quantifies forward-KL top-20 vs reverse-KL OPD on reasoning or agentic tasks at this scale.

---

## Q3. RL recipes: algorithms, MoE stability (R3, GSPO, TIS/MIS), length control, frameworks, environments, and what Qwen/DeepSeek/Kimi/GLM/MiniMax report

### Takeaway
MoE RL in 2025–26 converged on three ingredients:
1. A GRPO-family objective with sequence-level or masked importance ratios (GSPO, CISPO, IcePop-style masking, DeepSeek off-policy sequence masking).
2. Removing the train–inference mismatch: routing replay (R3 / "Keep Routing") plus TIS/MIS, and as of Oct 2026 bitwise-consistent train/inference kernels.
3. Fully asynchronous rollout systems over very large sets of verifiable environments: 10K+ SWE envs (GLM-5), 807K PR-derived envs (Qwen SWE-Universe), and over 100K scaffolds/envs (MiniMax).

Length/efficiency RL (Toggle, DLER, GFPO) reduces tokens by 25–70% with small accuracy loss, though at least one replication failed.

### Cited Findings

**Algorithms**
- GSPO (Qwen) [2025-07-24]: the importance ratio is based on sequence likelihood, with sequence-level clipping and optimization. It "notably stabilizes Mixture-of-Experts (MoE) RL training", beats GRPO, and contributed to the Qwen3-2507 models — [arXiv 2507.18071](https://arxiv.org/abs/2507.18071)
- Qwen, "Stabilizing RL with LLMs: Formulation and Practices" [2025-12-01]:
  - The token-level surrogate is a first-order approximation of the sequence-level objective, valid only when training–inference discrepancy and policy staleness are both small.
  - On-policy: plain policy gradient with IS correction was most stable.
  - Off-policy: clipping plus Routing Replay were needed.
  - Cold-start choice washes out with long training.
  - Experiments on a 30B MoE consumed "hundreds of thousands of GPU hours".
  - [arXiv 2512.01374](https://arxiv.org/abs/2512.01374)
- R3, Rollout Routing Replay [2025-10-13]: records the inference engine's routing and replays it in the training forward pass. It significantly reduces training–inference KL and extreme discrepancies without slowing training, and outperforms GSPO and TIS — [arXiv 2510.11370](https://arxiv.org/abs/2510.11370)
- DeepSeek-V3.2 GRPO modifications [2025-12] — [DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1):
  - Unbiased K3 KL estimate, weak or no KL for math.
  - Off-policy sequence masking: mask negative-advantage sequences whose old-vs-current KL exceeds δ.
  - **Keep Routing**: reuse sampled expert routes in training, used since V3-0324.
  - **Keep Sampling Mask**: reuse the top-p/top-k truncation mask.
  - RL compute above 10% of pretraining cost.
- GLM-5 [2026-02] — [GLM-5 report](https://arxiv.org/pdf/2602.15763v2):
  - Reasoning RL is GRPO + IcePop (train/infer mismatch control) with the KL term removed.
  - Agentic RL is fully asynchronous, with a **Token-in-Token-out gateway** that prevents re-tokenization mismatch.
  - "Direct double-sided importance sampling": ratio = π_θ/π_rollout, tokens outside [1−ε_l, 1+ε_h] fully masked, no π_old tracking.
  - Samples are dropped when the oldest rollout version is more than τ versions stale.
  - Environment-crash samples are excluded; a GRPO group is padded if more than half are valid, otherwise dropped.
  - DSA indexer frozen during RL, with deterministic torch.topk to avoid mismatch.
- MiniMax (CISPO, Forge) [2026-02-13] — [MiniMax Forge blog](https://minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm):
  - CISPO adapted for long-horizon agents; Reasoning, General QA and Agent tasks trained in one mixed-domain run rather than sequential stages.
  - Dense process rewards penalize language mixing and tool-call errors; a task-completion-time reward encourages parallelism; reward-to-go normalization.
  - Context up to 200K.
  - Windowed-FIFO async scheduling (e.g., W = 0.3N).
  - Prefix-tree merging of shared-prefix completions gives a claimed 40x training speedup.
  - More than 100,000 real-world agent scaffolds/environments for M2.5; throughput of millions of samples per day.
- DAPO [2025-03]: clip-higher, dynamic sampling, token-level loss, overlong reward shaping — [arXiv 2503.14476](https://arxiv.org/abs/2503.14476) (not re-verified this session)

**Train–inference mismatch (TIS / MIS / bitwise consistency)**
- TIS: caps the rollout/train ratio, trading small bias for bounded variance. It is proposed in "On the Rollout-Training Mismatch in Modern RL Systems" (NeurIPS 2025), which argues kernel, precision and parallelism differences silently turn on-policy RL into off-policy RL — [NeurIPS 2025](https://neurips.cc/virtual/2025/124259)
- MIS: drops whole sequences whose sequence-level ratio exceeds a threshold — ["When Speed Kills Stability" blog](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda). verl integrated geometric-mean sequence-level masking (Geo-MIS/Geo-RS) per that author (secondary via search summary).
- NVIDIA NeMo RL ProRLv2 guide: for MoE, tiny floating-point differences can flip router expert selection and spike token ratios. Supports token- and sequence-level masking — [NeMo RL ProRLv2](https://docs.nvidia.com/nemo/rl/latest/guides/prorlv2.html)
- **Bitwise train–inference consistency** (vLLM blog, NVIDIA + Inferact) [2026-10-14] — [vLLM blog](https://vllm.ai/blog/2026-10-14-train-inference-consistency):
  - The training forward pass calls vLLM inference kernels made batch-, phase- and layout-invariant, so logprob diff is exactly 0.
  - DeepSeek-V4-Flash via verl PR #7688 (Megatron + vLLM), 100 steps with no TIS/MIS: the baseline's train–infer KL grew from 2e-3 to 29.9 and reward regressed; the consistent run stayed at diff 0. Rollout throughput was 91.3% of baseline.
  - Nemotron-3.5-Lightning-30B-A3B on SWE-bench Pro: the consistent run improved 0.56→0.59, while the baseline with R3 + NVFP4 QAT collapsed to 0.22 as entropy rose from 0.05 to 2.6.
  - Related work: slime (GLM-5), TorchTitan + vLLM kernels, and **IsoExec for Qwen3.5-35B-A3B (Gated DeltaNet)**.
  - Limits: TP=1 only, no MTP or speculative decoding yet. About 10% end-to-end overhead on 32 GB200 (verl PR preview). Recommends one optimizer update per step.

**Length-controlled / token-efficient RL**
- Kimi "Toggle" [K2.5 report, 2026-02, v2 2026-08]: alternates a budget-limited phase (budget = ρ-th percentile of correct-response lengths) with a normal phase every m iterations. Output tokens fell 25–30% with negligible performance impact on K2 Thinking, and the reduction transferred to GPQA and MMLU-Pro despite training only on math and code — [Kimi K2.5 report](https://arxiv.org/html/2602.02276v2)
- DLER (NVIDIA) [2025-10]: a plain truncation penalty works when paired with batch-wise reward normalization, higher clipping and dynamic sampling. Output length fell more than 70% with accuracy preserved — [arXiv 2510.15110](https://arxiv.org/pdf/2510.15110)
- GFPO (Microsoft, ICLR 2026): sample larger groups and train only on short or high reward-per-token responses. Cuts GRPO length inflation 46–71% (71–85% with the reward-per-token variant) on Phi-4-reasoning — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a00ef870c140f09263ba2288855be50b-Abstract-Conference.html). **Contradicted:** a Feb-2026 comparison could not reproduce GFPO's length reduction, and found length-penalty effects model-dependent (Qwen3-1.7B-Base degraded under stronger penalties) — [arXiv 2602.09591](https://arxiv.org/html/2602.09591v2)
- Qwen3.8: users control `reasoning_effort` (xhigh/medium/low). The AA index drops 34 → 28 → 26 from xhigh to low, yet cost per task does not fall (because of retries and turns) — [AA comparison](https://artificialanalysis.ai/models/releases/comparisons/qwen3-8-27b-vs-qwen3-6-35b-a3b); [Qwen3.8-Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

**What labs report about agentic RL for their MoEs (2025–2026)**
- **Qwen**:
  - Qwen3.5 RL was scaled across "million-agent environments with progressively complex task distributions", using asynchronous RL frameworks for massive agent scaffolds — [QwenLM/Qwen3.8 README](https://github.com/QwenLM/Qwen3.8)
  - Qwen3-Coder [2025-07] ran 20,000 parallel environments for agent RL — [Qwen3-Coder blog](https://qwenlm.github.io/blog/qwen3-coder/) (via search summary)
  - Qwen3-Coder-Next [2026-03], SWE expert RL:
    - Trajectory-level completion reward.
    - Unfinished-trajectory penalty when turns exceed the maximum.
    - Token-level penalty on invalid tool calls.
    - Reward-hacking blocker that rejects tool calls combining a repo link with git/curl/wget.
    - Average agent turns rose from 50 to 130 during RL.
    - [arXiv 2603.00729](https://arxiv.org/html/2603.00729v1)
  - Qwen-UI-Agent [2026-07-31]: online RL on trajectories longer than 100 turns with more than 10,000 concurrent environments — [arXiv 2607.28227](https://arxiv.org/html/2607.28227v1) (via search summary)
- **Qwen-AgentWorld** [2026-06-23], directly on the 35B-A3B class — [arXiv 2606.24597](https://arxiv.org/html/2606.24597v1):
  - A language world model trained by CPT→SFT→RL (GSPO) simulates environments across MCP, Search, Terminal, SWE, Android, Web and OS from more than 10M interaction trajectories.
  - Simulated-environment RL on **Qwen3.5-35B-A3B** with the 397B simulator: Claw-Eval 65.4→69.7, QwenClawBench 47.9→55.0. Using Qwen3.6-Plus as the simulator gave only 66.7 / 47.8.
  - Controlled simulation for MCP raised Tool Decathlon 32.4→36.1 and MCPMark 21.5→33.8.
  - Terminal-only RL for 100 steps on the 35B-A3B SFT model transferred across domains: Terminal 32.8→47.0, SWE 52.0→63.5, Search 20.2→32.0, MCP 53.5→58.5.
  - For Search, simulated RL scored 50.3% vs real-search RL 45.6%.
- **DeepSeek** [V3.2, 2025-12] — [DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1):
  - Large-scale agentic task synthesis: 1,827 environments and about 85K prompts.
  - Breakdown: code agent 24,667 tasks (real env, extracted prompts); search agent 50,275 (real env, synthesized prompts); general agent 4,417 (synthesized env); code interpreter 5,908.
  - Thinking is retained across tool calls and discarded only on a new user message.
- **GLM** [GLM-5, 2026-02] — [GLM-5 report](https://arxiv.org/pdf/2602.15763v2):
  - 744B/40B-active MoE trained on 28.5T tokens.
  - More than 10K verifiable SWE environments across thousands of repos in 9 languages (RepoLaunch-based pipeline).
  - Thousands of synthesized terminal environments in Harbor format with Docker build accuracy above 90%, plus web-corpus-derived terminal tasks and multi-hop search tasks.
  - Results: SWE-V 77.8; Terminal-Bench 2.0 56.2 (60.7 on a verified variant); BrowseComp 62.0 (75.9 with context management); τ²-bench 89.7; HLE 30.5 (50.4 with tools).
- **Kimi** [K2.5, 2026-02-02; v2 2026-08-07] — [Kimi K2.5 report](https://arxiv.org/html/2602.02276v2):
  - Joint text–vision RL, PARL (Q4) and Toggle.
  - Results: SWE-V 76.8, Terminal-Bench 2.0 50.8, HLE-Full with tools 50.2, BrowseComp 60.6 single-agent / 78.4 Agent Swarm.
  - Kimi K2 Thinking [2025-11] executes 200–300 sequential tool calls — [HF card](https://huggingface.co/moonshotai/Kimi-K2-Thinking) (not re-verified this session)
- **MiniMax** [M2.5, 2026-02]: Forge + CISPO with process and completion-time rewards over more than 100K scaffolds/environments (above). Secondary sources say about 200K environments and about two months of training — [VentureBeat](https://venturebeat.com/ai/minimaxs-new-open-m2-5-and-m2-5-lightning-near-state-of-the-art-while) (secondary; figures vary)

**Frameworks**
- Classification: AReaL, ROLL and slime are fully asynchronous; verl and OpenRLHF are synchronous Ray-based. In tool support, verl and SkyRL have a Python executor, SkyRL also bash and SQL — [VerlTool arXiv 2509.01055](https://arxiv.org/pdf/2509.01055)
- slime: SGLang-native, used for GLM-4.5 through GLM-5. AMD (2025, vendor-adjacent) called it the only verified framework for RL on large MoEs up to 355B — [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/slime/README.html); [GLM-5 card](https://huggingface.co/zai-org/GLM-5)
- verl: Megatron MoE backend; DeepSeek-V4-Flash bitwise-consistent recipe in PR #7688 — [vLLM blog](https://vllm.ai/blog/2026-10-14-train-inference-consistency)
- AReaL: fully async agentic RL (AReaL-lite). Claims a 235B MoE trained with it surpasses GPT-5 on τ²-bench — [AReaL PyPI](https://pypi.org/project/areal/) (self-reported)
- SkyRL-Agent: async pipeline dispatcher, 1.55x faster than naive async batching — [arXiv 2511.16108](https://arxiv.org/html/2511.16108v1)
- rLLM: used to train DeepSWE — [Together AI blog](https://together.ai/blog/deepswe)

**Environments and datasets**
- SWE-smith [2025-04-30]: 50K task instances from 128 repos. SWE-agent-LM-32B reaches 40.2% SWE-V pass@1. Prior datasets had "at most 1,000s of training instances from 11 or fewer GitHub repositories" — [arXiv 2504.21798](https://arxiv.org/abs/2504.21798)
- R2E-Gym: about 8.1K problems — [arXiv 2504.07164](https://arxiv.org/abs/2504.07164) (not re-verified)
- SWE-Gym: about 2.4K instances, 11 repos — [arXiv 2412.21139](https://arxiv.org/abs/2412.21139) (not re-verified)
- SWE-Universe (Qwen authors) [2026-02-02, v2 2026-09-21]: 807,693 real-world multilingual verifiable environments built from GitHub PRs by a custom building agent with self-verification and in-loop hacking detection. Used for agentic mid-training and RL of Qwen3-Max-Thinking (75.3% SWE-V) — [arXiv 2602.02361](https://arxiv.org/abs/2602.02361)
- Qwen3-Coder-Next data: about 852K synthetic SWE tasks (SWE-smith, SWE-Flow, SWE-rebench, SWE-smith-multi, Multi-SWE-RL) plus about 807.7K PR-derived instances across 52,960 repos, each a reusable Docker image — [arXiv 2603.00729](https://arxiv.org/html/2603.00729v1)
- Terminal:
  - Tmax (UW/AI2) [2026-06-22]: 9B model reaches 27% on Terminal-Bench 2.0 with outcome-only RL. Dataset is "over 2.5x larger than previously released"; secondary sources say TMax-15k with 14,600 RL environments — [arXiv 2606.23321](https://arxiv.org/abs/2606.23321)
  - SETA [2026-07]: SETA-Env with more than 4,500 verifiable terminal environments; Qwen3-8B + GRPO reaches 12% on TB2.0 — [arXiv 2607.10891](https://arxiv.org/abs/2607.10891) (via search summary)
  - Endless Terminals [2026-01] — [arXiv 2601.16443](https://arxiv.org/pdf/2601.16443) (title only)
- τ-style and simulated environments: TAU3-Bench uses an LLM user simulator (gpt-5.2) — [Qwen3.6 card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Qwen-AgentWorld simulates about 4K OpenClaw environments and about 1K fictional search environments — [arXiv 2606.24597](https://arxiv.org/html/2606.24597v1)

### Inferences
- For RL on Qwen3.6-35B-A3B (Gated DeltaNet + 256-expert MoE), the minimum stability stack the evidence supports is:
  - GSPO or masked/double-sided IS (GLM-5 style).
  - Rollout routing replay (R3 / Keep Routing).
  - TITO tokenization.
  - Stale-sample dropping.
  - One update per step where possible.
  - If engineering budget allows, bitwise train–inference consistency. A Qwen3.5-35B-A3B recipe (IsoExec) is cited as existing, so the same architecture family has been done.
- Qwen-AgentWorld shows +4 to +14 pt agentic gains on Qwen3.5-35B-A3B from about 100 RL steps of terminal or simulated-environment RL. RL on a few thousand to tens of thousands of verifiable SWE/terminal/MCP environments is the most direct lever for the agentic gap that distillation leaves.
- Length control should be applied *after* capability transfer. Toggle-style budget phases are the best-documented method (25–30% fewer tokens), and the GFPO replication failure argues for validating on your own model.

### Gaps
- No public Qwen3.6- or Qwen3.8-specific post-training report exists. The README only gives one-line Qwen3.5 RL claims, so the exact Qwen3.8 recipe (RL algorithm, environment counts, OPD use) is unknown.
- The R3 paper's quantitative KL reductions and models were not extracted (abstract only).
- No 2026 head-to-head benchmark of verl / slime / AReaL / ROLL / SkyRL for MoE agentic RL was found. ROLL's 2026 MoE status is unverified.
- Not found: MiniMax M2.5 algorithm equations and GPU counts, or Kimi K2.5 RL environment counts.

---

## Q4. Goal-based / agentic architectures: planner–executor, hierarchical agents, process rewards, context management, and measured gains

### Takeaway
The best-documented 2026 gains come from three sources:
1. **Context management**, both inference-time and RL-trained: +8 to +24 pts on BrowseComp.
2. **Hierarchical orchestrator training** (Kimi PARL: only the orchestrator is RL-trained, subagents frozen): +17.8 pts BrowseComp, up to 4.5x lower latency.
3. **Planner-only RL** in a modular planner/executor/verifier loop (AgentFlow / Flow-GRPO): +14–15% at 7B.

Process-level shaping used in production is lightweight: tool-format penalties, turn caps, completion-time rewards, and auxiliary rewards annealed to zero. Learned PRMs are less common.

### Cited Findings
- **Kimi K2.5 Agent Swarm / PARL** [2026-02] — [Kimi K2.5 report](https://arxiv.org/html/2602.02276v2):
  - Training: only the orchestrator is updated by RL. Sub-agents are frozen and their trajectories excluded from the objective.
  - Reward: r = λ₁·r_parallel (counters "serial collapse") + λ₂·r_finish (counters "spurious parallelism") + r_perf, with λ₁ and λ₂ annealed to 0.
  - Cost metric: CriticalSteps = Σ (main steps + max over subagent steps).
  - Curriculum: small subagents first, then larger.
  - Results: BrowseComp 60.6 single-agent → 78.4 swarm; WideSearch Item-F1 72.7 → 79.0; 3–4.5x less time to reach target performance.
- **AgentFlow / Flow-GRPO** (Stanford) [2025-10; ICLR 2026] — [arXiv 2510.05592](https://arxiv.org/html/2510.05592v1):
  - Planner, executor, verifier and generator share an evolving memory; only the planner is RL-trained "in the flow".
  - Flow-GRPO broadcasts one verifiable trajectory-level outcome to every turn, with group-normalized advantages.
  - At 7B: average gains of +14.9% on search, +14.0% agentic, +14.5% math and +4.1% science vs top baselines.
- **Context-Folding / FoldGRPO** [2025-10]: the agent branches into a sub-trajectory for a subtask, then folds it into a summary. Trained end-to-end with RL plus process rewards. Matches or beats ReAct on deep-research and SWE tasks with a 10x smaller active context, and beats summarization-based management — [HF papers 2510.11967](https://huggingface.co/papers/2510.11967)
- FoldAct [2025-12] addresses folding-RL instability with separated losses, a full-context consistency loss and selective segment training (up to 80.7% less compute per a secondary summary) — [papers.cool 2512.22733](https://papers.cool/arxiv/2512.22733)
- Summarization-augmented multi-turn RL (periodic summaries to exceed the context window) — [arXiv 2510.06727 via sotaverified](https://sotaverified.org/papers/251006727)
- Counterpoint [2026-06]: "Beyond Compaction" proposes deterministic, model-free structured eviction as an alternative to learned folding — [arXiv 2606.11213](https://arxiv.org/pdf/2606.11213) (abstract-level only)
- **Measured inference-time context-management gains on BrowseComp:**
  - DeepSeek-V3.2 (trigger at 80% of context): baseline 51.4; Summary 60.2 (average 364 steps); Discard-all 67.6 — [DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1)
  - GLM-5: keep-recent-k (k=5; fold observations older than the last 5 rounds) 55.3 → 62.0. Hierarchical keep-recent + discard-all at a 32K threshold reaches 75.9. The authors note accuracy degrades beyond about 100K tokens of context — [GLM-5 report](https://arxiv.org/pdf/2602.15763v2)
  - Kimi K2.5: 60.6 → 74.9 with Discard-all. Agent Swarm is described as "proactive" context management (sharding) that beats Discard-all — [Kimi K2.5 report](https://arxiv.org/html/2602.02276v2)
- **Context management as a trained action:** MiniMax Forge models context operations as explicit agent actions, with context transitions inside the environment dynamics, so the policy learns to keep task-critical information. For black-box agents it supports memory compression and history rewriting — [MiniMax Forge](https://minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)
- **Thinking persistence across turns:**
  - Qwen3.6 trains `preserve_thinking` (keep earlier reasoning traces), "particularly beneficial for agent scenarios" — [Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
  - DeepSeek-V3.2 keeps reasoning while only tool messages are appended — [DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1)
  - Qwen3.8 cards say preserved thinking also improves KV-cache reuse — [Qwen3.8-Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Process and shaping rewards used in production agentic RL:**
  - MiniMax: process reward on intermediate behavior (language mixing, tool errors), completion-time reward and reward-to-go — [Forge](https://minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)
  - Qwen3-Coder-Next: token-level invalid-tool-call penalty and unfinished-trajectory penalty — [arXiv 2603.00729](https://arxiv.org/html/2603.00729v1)
  - GLM-5: loss-masking of erroneous trajectory segments in SFT, and dropping environment-crash samples in RL — [GLM-5](https://arxiv.org/pdf/2602.15763v2)
  - Kimi PARL: annealed auxiliary rewards — [K2.5](https://arxiv.org/html/2602.02276v2)
- **Goal-conditioned long-horizon benchmarks used as targets:**
  - Qwen3.8 reports CoWorkBench (in-house long-horizon productivity across CS, finance, law and medicine), JobBench, Agents' Last Exam and DeepSWE-1.1 (Claude Code harness, 8-h timeout for QwenSWEBench).
  - On Agents' Last Exam, Qwen3.8-27B scores 20.4 pass@1 / 42.9 score, vs 10.6 / 27.3 for Qwen3.6-27B.
  - [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)

### Inferences
- For a "goal-compiled" agent at 3B active, the highest-evidence architecture is a single policy trained (a) with context management in the loop (keep-recent-k plus discard or fold actions, ideally learned per MiniMax/FoldGRPO), and (b) optionally as an orchestrator over frozen sub-agents (PARL-style), where only the top-level goal-decomposition policy gets RL.
- Planner-only RL (AgentFlow, PARL) is compute-cheap because executor trajectories are excluded from the gradient. This fits a plan where the 3B-active model is the planner/orchestrator and tools or sub-agents are fixed.
- Learned PRMs for long-horizon agents are not what leading labs report using. Rule-based shaping plus outcome rewards dominates, so budget for environment and verifier construction rather than PRM training.

### Gaps
- No controlled study was found that compares planner–executor training vs single-policy end-to-end RL on SWE-bench or Terminal-Bench at matched compute.
- No primary numbers were found for "goal-conditioned policy" methods (explicit goal embeddings or hindsight relabeling) in 2026 LLM agents. The literature uses task prompts as goals.
- The FoldGRPO and FoldAct numbers came via aggregators. Their arXiv full text was not fetched.

---

## Q5. Compute and data budgets reported (for costing a plan)

### Takeaway
Reported budgets range from about 1–2K GPU-hours (on-policy distillation of an 8B; DeepSWE-scale SWE RL at 64 H100 × 6 days) up to hundreds of thousands of GPU-hours (Qwen's 30B-MoE RL stability study). Frontier-lab post-training reaches more than 10% of pretraining compute (DeepSeek-V3.2). Data needs are about 4K query-verifier pairs for reasoning RL, about 77K prompts for OPD, and thousands to hundreds of thousands of executable environments for agentic RL.

### Cited Findings

| Recipe | Model / scale | Data | Compute | Result | Source |
|---|---|---|---|---|---|
| On-policy distillation vs RL | Qwen3-8B (from off-policy-distilled ckpt) | — | RL 17,920 GPU-h; OPD 1,800 GPU-h | AIME'24 67.6 (RL) vs 74.4 (OPD) | [Qwen3 report, 2025-05](https://arxiv.org/html/2505.09388v1) |
| On-policy distillation (Tinker) | Qwen3-8B-Base ← Qwen3-8B/32B | ~77K prompts × 4 samples, ~150 steps | 8.4e19 teacher + 8.2e19 student FLOPs (9–30x less than SFT-2M) | AIME'24 70% | [Thinking Machines, 2025-10-27](https://thinkingmachines.ai/blog/on-policy-distillation/) |
| Long-CoT SFT | Qwen3-8B-Base | 400K OpenThoughts-3 → 60%; ~2M extrapolated for 70% | SFT-2M ≈ 3.4e21 + 1.5e21 FLOPs | AIME'24 60–70% | [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) |
| Reasoning RL (flagship) | Qwen3-235B-A22B | 3,995 query-verifier pairs, 170 steps | n/r | AIME'24 70.1 → 85.1 | [Qwen3 report](https://arxiv.org/html/2505.09388v1) |
| MoE RL stability study | 30B MoE | — | "hundreds of thousands of GPU hours" | recipe findings | [arXiv 2512.01374, 2025-12](https://arxiv.org/abs/2512.01374) |
| SWE agentic RL (DeepSWE) | Qwen3-32B dense, rLLM | ~4.5K R2E-Gym tasks | 64 H100 × 6 days (~9.2K H100-h, my arithmetic) | SWE-V 42.2% pass@1 (avg of 16); 59% with TTS; ~+20% after 200 steps | [Together AI blog, 2025-07](https://together.ai/blog/deepswe) |
| Terminal/sim RL | Qwen3.5-35B-A3B | 34,125 terminal RL trajectories; up to 50 turns; 100 RL steps | n/r | Terminal 32.8→47.0, SWE 52.0→63.5 | [Qwen-AgentWorld, 2026-06-23](https://arxiv.org/html/2606.24597v1) |
| Agentic task synthesis | DeepSeek-V3.2 (671B MoE) | 1,827 envs; ~85K prompts | RL > 10% of pretraining cost | BrowseComp 51.4 (67.6 with ctx mgmt), SWE-V 73.1 | [DeepSeek-V3.2, 2025-12](https://arxiv.org/html/2512.02556v1) |
| SWE environment scale | Qwen3-Max-Thinking | 807,693 PR-derived envs | n/r | SWE-V 75.3% | [SWE-Universe, 2026-02/09](https://arxiv.org/abs/2602.02361) |
| Coding-agent post-training | Qwen3-Coder-Next 80B-A3B | ~800K verifiable SWE tasks; ~600B tokens of repo-level mid-training | n/r | SWE-V 70.6–71.3 | [arXiv 2603.00729, 2026-03](https://arxiv.org/html/2603.00729v1) |
| Agentic RL infra | GLM-5 744B-A40B | >10K SWE envs (9 languages) + thousands of terminal envs + search tasks | n/r | SWE-V 77.8; TB2.0 56.2 | [GLM-5, 2026-02](https://arxiv.org/pdf/2602.15763v2) |
| Agent RL at scale | MiniMax M2.5 | >100K scaffolds/envs; millions of samples/day; 200K ctx | n/r (secondary: ~2 months) | n/r on blog | [Forge, 2026-02-13](https://minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm) |
| Terminal RL (open) | 9B (Tmax) | dataset "2.5x larger than prior" (secondary: 14.6K envs) | n/r | TB2.0 27% | [arXiv 2606.23321, 2026-06-22](https://arxiv.org/abs/2606.23321) |
| Terminal RL (open) | Qwen3-8B + GRPO | SETA-Env >4,500 envs | n/r | TB2.0 12% | [arXiv 2607.10891, 2026-07](https://arxiv.org/abs/2607.10891) |
| SWE SFT data | SWE-agent-LM-32B | 50K instances, 128 repos | n/r | SWE-V 40.2% | [SWE-smith, 2025-04-30](https://arxiv.org/abs/2504.21798) |
| Pretraining reference | gpt-oss-120b / 20b | — | 2.1M H100-h / "almost 10x fewer" | — | [arXiv 2508.10925](https://arxiv.org/html/2508.10925v1) |
| Bitwise-consistent RL overhead | DeepSeek-V4-Flash; Nemotron 30B-A3B | — | rollout throughput 91.3% / 87.5% of baseline; ~10% end-to-end on 32 GB200 | prevents KL blow-up / collapse | [vLLM blog, 2026-10-14](https://vllm.ai/blog/2026-10-14-train-inference-consistency) |

- MiniMax-M1's CISPO RL run was reported as 512 H800 for 3 weeks at about $534.7K — [MiniMax-M1 arXiv 2506.13585](https://arxiv.org/abs/2506.13585) (not re-verified this session)
- Evaluation cost reference (AA v4.3.2): $0.48 per task for Qwen3.6-35B-A3B vs $1.01 for Qwen3.8-27B; 227 s vs 1,184 s per task — [AA comparison](https://artificialanalysis.ai/models/releases/comparisons/qwen3-8-27b-vs-qwen3-6-35b-a3b)

### Inferences
- A plausible three-stage plan for Qwen3.6-35B-A3B, with costs extrapolated by me from the cited anchors, not sourced:
  1. **OPD from Qwen3.8-27B:** about 1–3K H100-h for about 100K prompts × 4 samples on reasoning, code and tool-use prompts.
  2. **Agentic RL** with GSPO or masked IS, R3 and TITO on about 5–15K verifiable SWE/terminal/MCP environments: about 10–30K H100-h. DeepSWE used about 9.2K H100-h for a 32B dense on 4.5K tasks. A 3B-active MoE decodes cheaper, but rollouts are longer at 100+ turns, and sandbox CPU cost is extra.
  3. **Toggle-style length RL and final cross-stage OPD:** about 1–2K H100-h.
  - Environment construction (Docker images and verifiers) is likely the dominant engineering cost, not GPU time.
- Expect stability engineering (routing replay, consistency kernels) to cost about 10% throughput but to prevent the run-ending collapses documented for MoE RL.

### Gaps
- Not found: GPU-hour figures for DeepSeek-V4, GLM-5, Kimi K2.5, MiniMax M2.5 or any Qwen3.6/3.8 post-training.
- Not found: per-environment sandbox CPU costs for SWE/terminal RL at scale.
- No source gives OPD cost for an MoE student with a dense teacher of a different architecture size. The estimates above are extrapolations.

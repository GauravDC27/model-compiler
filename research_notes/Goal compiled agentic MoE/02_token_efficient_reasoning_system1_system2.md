# Token-Efficient Reasoning and System-1 / System-2 (Fast vs Deliberate) Designs for LLMs and Agents (state as of 2026-10-21)

Scope: techniques that cut tokens, latency or cost per solved task without losing accuracy, with measured accuracy-vs-token numbers. Also dual-process (fast vs slow) designs and how they decide to escalate. Target use: an agentic MoE (base Qwen3.6-35B-A3B) that has to match Qwen3.8-27B's reasoning accuracy at much lower token cost.

Conventions:
- Dates are arXiv v1 submission dates, or the publication or access date of the web page.
- "abs" means the number comes from the paper abstract.
- Benchmark deltas are absolute percentage points (pp) unless marked "relative".

---

## Q1. Efficient-reasoning methods and their measured accuracy-vs-token results (hybrid thinking, budget control, length-controlled RL, adaptive thinking, CoT compression, early exit, overthinking, parallel vs sequential compute, self-consistency early stopping, 2026 results)

### Takeaway
By late 2026 the best methods cut reasoning tokens by about 25–80% at equal or better accuracy on easy and medium problems (GSM8K, MATH500, AMC). The main families are RL with length or difficulty-aware rewards, learned think/no-think selection, and confidence-based early exit. On the hardest problems (AIME, HMMT, LiveCodeBench) the measured trade-off is usually a 0–2 pp loss for about 30–50% fewer tokens. The 2026 work that also gains accuracy on hard sets (When2Think, Lightning Weave, TRAAC, SCALE) combines difficulty-adaptive allocation with RL or distillation rather than uniform length pressure. For the target pair the gap is real: Qwen3.6-35B-A3B trails Qwen3.8-27B by about 10 pp on LiveCodeBench v6 and about 3 pp on GPQA. The 27B is also among the most verbose models Artificial Analysis measures. Token efficiency therefore has to be co-optimized with accuracy-raising post-training; it cannot simply be bolted on.

### Cited Findings

#### Target-model baseline and native thinking controls
- **Qwen3.6-35B-A3B (card; release April 16, 2026 per AA):**
  - Architecture: 35B total / 3B active; 40 layers in a hybrid layout of "10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))"; 256 experts, 8 routed + 1 shared active; 262,144-token native context. It has MTP "trained with multi-steps".
  - Scores vs Qwen3.5-27B: AIME26 92.7 (vs 92.6), GPQA 86.0 (vs 85.5), LiveCodeBench v6 80.4 (vs 80.7), SWE-bench Verified 73.4 (vs 75.0), Terminal-Bench 2.0 51.5 (vs 41.6), TAU3-Bench 67.2 (vs 68.4).
  - — [HF model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Qwen3.6-35B-A3B thinking controls:**
  - Thinking is on by default.
  - Thinking is turned off per request with `chat_template_kwargs: {"enable_thinking": False}`.
  - Qwen3.6 "does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`".
  - `preserve_thinking: True` keeps reasoning from earlier turns. By default only the latest user turn's thinking is kept.
  - Recommended max output is 32,768 tokens, or 81,920 for competition math and code.
  - The card gives no thinking-budget guidance.
  - — [HF model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Qwen3.8-27B (dense VLM; released Aug 14, 2026):**
  - Architecture: 64 layers in a hybrid layout of 16 × (3 × Gated DeltaNet + 1 × Gated Attention); MTP "trained with multiple steps".
  - `reasoning_effort` ∈ {`xhigh` (default), `medium`, `low`}.
  - `preserve_thinking` is ON by default, which the card says "improves KV cache utilization".
  - Reasoning content can run up to 262,144 tokens.
  - Scores: GPQA-Diamond 89.2, LiveCodeBench v6 90.3, HLE 30.8, Terminal Bench 2.1 73.0, SWE-bench Pro 61.7. The Qwen3.6-27B column on the same card: GPQA 87.8, LCB v6 83.9, TB2.1 63.4.
  - The card warns: "lower reasoning effort does not always reduce overall task completion time". Low effort can cause "insufficient analysis, more failures, and repeated retries", which raises total latency and tokens.
  - — [HF model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Artificial Analysis token measurements (accessed 2026-10-21):**
  - Qwen3.8-27B (xhigh): Intelligence Index 34; 200M output tokens to run the Index (median 82M; ranked #25/27 for verbosity, i.e. "very verbose"); $1.01 per Index task; 43.6 output tok/s on the Alibaba API. — [AA Qwen3.8-27B](https://artificialanalysis.ai/models/qwen3-8-27b)
  - Qwen3.6-35B-A3B (reasoning): Index 18; 160M output tokens; $0.48 per task; 134.4 tok/s. — [AA Qwen3.6-35B-A3B](https://artificialanalysis.ai/models/qwen3-6-35b-a3b)
  - Conflict: a third-party guide quoted an AA index of 52 for Qwen3.8-27B, and OpenRouter lists 33.7 (xhigh). The scores may come from different index versions. — [OpenRouter](https://openrouter.ai/qwen/qwen3.8-27b)
- **Practitioner report (Simon Willison, Aug 16, 2026):** "Qwen 3.8 27B … defaults to wildly overthinking things".
  - An SVG prompt at `xhigh` took 21 minutes with 22,276 reasoning + 3,223 output tokens.
  - The same prompt with reasoning off took 3,715 tokens and 137 s, at somewhat lower quality.
  - His advice: run it "on low or even no reasoning levels at first".
  - — [simonwillison.net](https://simonwillison.net/2026/Aug/16/qwen-38-27b/)

#### Hybrid thinking and thinking-budget controls
- **Qwen3 "Thinking Mode Fusion" (May 14, 2025):**
  - SFT mixes `/think` and `/no_think`-flagged samples. Non-thinking samples keep an empty think block.
  - Thinking budget: when the thinking length reaches the user budget, generation is halted and the model is fed "Considering the limited time by the user, I have to give the solution based on the thinking directly now." The paper says this ability *emerges* from mode fusion and was not trained explicitly.
  - Performance scales "smooth[ly]" with budget (figure only; no numbers in the text).
  - — [Qwen3 Tech Report](https://arxiv.org/abs/2505.09388), [HTML](https://arxiv.org/html/2505.09388)
- **Qwen3 thinking vs non-thinking on the same weights (System 2 vs System 1 gap):**
  - Qwen3-30B-A3B: AIME'24 80.4 vs 32.8; AIME'25 70.9 vs 21.6; LiveCodeBench v5 62.6 vs 29.8; GPQA-D 65.8 vs 54.8.
  - Qwen3-32B: AIME'24 81.4 vs 31.0; GPQA-D 68.4 vs 54.6.
  - — [Qwen3 TR tables 13–16](https://arxiv.org/html/2505.09388)
- **Qwen3 blog (Apr 29, 2025):** documents `/think` and `/no_think` soft switches; in multi-turn chats the model follows the most recent flag. — [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)
- **s1 budget forcing (Jan 31, 2025):**
  - Mechanism: force-stop thinking, or append "Wait" to extend it.
  - s1-32B, trained on only 1K examples, beats o1-preview on competition math by up to 27%.
  - Budget forcing extrapolates AIME24 from 50% to 57%.
  - — [s1](https://arxiv.org/abs/2501.19393)
- **Elastic Reasoning (May 8, 2025):**
  - Splits generation into a thinking phase and a solution phase with separate budgets. Trained with budget-constrained rollouts inside GRPO.
  - Robust under strict budgets on AIME, MATH500, LiveCodeBench and Codeforces, and more concise even when unconstrained (abs; no aggregate numbers in the abstract).
  - — [Elastic Reasoning](https://arxiv.org/abs/2505.05315)
- **L1 / LCPO (Mar 6, 2025):**
  - RL with a length target given in the prompt; beats s1 budget forcing at length control.
  - Its "Short Reasoning Models": the 1.5B L1 beats GPT-4o at equal reasoning length.
  - — [L1](https://arxiv.org/abs/2503.04697)
- **Star Elastic (NVIDIA, May 8, 2026):**
  - Nests submodels inside Nemotron Nano v3 (30B / 3.6B active hybrid MoE), producing 23B (2.8A) and 12B (2.0A) variants from one 160B-token run.
  - Using different nested submodels for the thinking vs answering phases gives "up to 16% higher accuracy and 1.9× lower latency".
  - Elastic NVFP4 and FP8 checkpoints are also released.
  - — [Star Elastic](https://arxiv.org/abs/2605.07182)

#### RL length shaping and short-CoT training

| Method (date) | Model | Accuracy delta | Token delta | Source |
|---|---|---|---|---|
| ThinkPrune (Apr 2, 2025): iterative RL with a hard token cap | R1-Distill-Qwen-1.5B, AIME24 | −2% | length halved | [ThinkPrune](https://arxiv.org/abs/2504.01296) |
| ShorterBetter (Apr 30, 2025): reward = shortest correct sample length | R1-Distill 1.5B/7B | accuracy maintained | −50–80% output length, in and out of domain | [ShorterBetter](https://arxiv.org/abs/2504.21370) |
| O1-Pruner (Jan 22, 2025): pre-sample baseline, then RL-style length-harmonizing fine-tune | math benchmarks | higher accuracy | "significantly reduces" overhead | [O1-Pruner](https://arxiv.org/abs/2501.12570) |
| DAST (Mar 6, 2025): Token Length Budget difficulty metric + budget preference optimization | several | accuracy kept on complex problems | >30% average reduction | [DAST](https://arxiv.org/abs/2503.04472) |
| Training LMs to Reason Efficiently (Feb 6, 2025): RL family controlled by one hyperparameter | 2 open LRMs | "most of the accuracy" kept | "significant" reduction | [Arora & Zanette](https://arxiv.org/abs/2502.04463) |
| Concise Reasoning via RL (Apr 7, 2025): short 2nd RL phase on a small set of solvable problems | several | preserved or improved | length "significantly" reduced | [Concise RL](https://arxiv.org/abs/2504.05185) |
| TRAAC (Oct 2, 2025): attention-based pruning + difficulty-calibrated reward, on AIME/AMC/GPQA-D/BBEH | Qwen3-4B | +8.4 pp avg vs base; +7.9 pp vs best RL baseline | −36.8% vs base; −29.4% vs best RL baseline | [TRAAC](https://arxiv.org/abs/2510.01581) |
| Stable adaptive thinking (Feb 26, 2026): hybrid FT + correctness-preserving advantage shaping + length-aware gradient regulation | Qwen2.5-1.5B / 7B | +3.7 / +3.6 pp | −40.6% / −43.9% | [arXiv 2602.22556](https://arxiv.org/abs/2602.22556) |
| When2Think (Sep 17, 2026): RLVR with correctness-gated, difficulty-aware efficiency bonus; Think/NoThink exploration | AIME24 | Pass@3 +10.0 pp | −27.9% vs backbone | [When2Think](https://arxiv.org/abs/2609.19671) |
| Lightning Weave (Sep 13, 2026): on-policy distillation composing an "accuracy" specialist and an "efficiency" specialist | Qwen3.5-4B | HMMT25 59.2→64.0; LCB v5 41.7→54.2 | −10.7% / −9.6% response tokens | [Lightning Weave](https://arxiv.org/abs/2609.14708) |
| Self-supervised confidence training (Sep 25, 2026): fine-tune on predicting own confidence mid-trace (600 problems), no length objective | Gemma, Qwen, Nemotron, GPT-OSS | matched accuracy | up to −25% tokens | [arXiv 2609.31619](https://arxiv.org/abs/2609.31619) |
| SAT stepwise adaptive thinking (Apr 9, 2026): FSM over Slow/Normal/Fast/Skip modes steered by a lightweight PRM; 9 LRMs, 7 benchmarks | various | maintained or improved | up to −40% | [SAT](https://arxiv.org/abs/2604.07922) |

- **Kimi k1.5 "long2short" (Jan 22, 2025):** transfers long-CoT skill into short-CoT models. Short-CoT results: AIME 60.8, MATH500 94.6, LiveCodeBench 47.3, i.e. up to +550% over GPT-4o / Claude 3.5 Sonnet. — [Kimi k1.5](https://arxiv.org/abs/2501.12599)
- **MiMo-V2.6 (Oct 8, 2026):** groupwise agentic grading at RL scale "steers the model towards shorter, more token-efficient solutions". Training freezes the MoE router and adds a "multi-layer defense against reward hacking". — [MiMo-V2.6](https://arxiv.org/abs/2610.11959)

#### Adaptive "when to think" (learned mode selection)

| Method (date) | Mechanism | Result | Source |
|---|---|---|---|
| AdaptThink (May 19, 2025) | Constrained RL toward NoThinking + importance sampling over both modes | R1-Distill-Qwen-1.5B on GSM8K/MATH500/AIME24: −53% average length, **+2.4 pp** accuracy | [AdaptThink](https://arxiv.org/abs/2505.13417) |
| Thinkless (May 19, 2025) | Control tokens `<short>` / `<think>`; DeGRPO decouples mode-token loss from response loss (vanilla GRPO collapsed) | −50–90% long-thinking use on Minerva Algebra, MATH-500, GSM8K | [Thinkless](https://arxiv.org/abs/2505.13379) |
| AdaCoT (ByteDance, May 17, 2025) | PPO with penalty coefficients as a Pareto knob; Selective Loss Masking against "decision boundary collapse" | Production traffic: CoT triggering as low as **3.18%**, average response tokens −69.06%, complex-task performance kept | [AdaCoT](https://arxiv.org/abs/2505.11896) |
| AutoThink (May 16, 2025) | Ellipsis prompt stochastically triggers thinking; multi-stage RL | R1-Distill-1.5B: +6.4% relative accuracy, −52% tokens | [AutoThink](https://arxiv.org/abs/2505.10832) |
| LHRM (May 20, 2025) | Hybrid FT cold start + HGPO online RL | Picks think/no-think from query context (abs; qualitative) | [LHRM](https://arxiv.org/abs/2505.14631) |
| RPO, root-token policy optimization (Oct 13, 2026) | Trains only the first "root" token (invoke `<think>` or not) with group-relative rewards | Difficulty-aware adaptive thinking at ~2% of the training compute of earlier adaptive methods | [RPO](https://arxiv.org/abs/2610.17767) |
| CAT (Jul 1, 2026) | Self-certainty signal inside preference optimization; compresses confident answers, deliberates on uncertain ones | Beats SOTA baselines on accuracy (abs; no numbers) | [CAT](https://arxiv.org/abs/2607.00862) |

#### CoT compression, short-CoT distillation and terse prompting
- **TokenSkip (Feb 17, 2025):** trains on CoTs with low-importance tokens removed. Qwen2.5-14B-Instruct on GSM8K: reasoning tokens −40% (313→181) for a <0.4% drop. — [TokenSkip](https://arxiv.org/abs/2502.12067)
- **C3oT (Dec 16, 2024):** a compressor plus conditioned training on long and short CoTs; >50% compression "without compromising effectiveness" on arithmetic and commonsense tasks. — [C3oT](https://arxiv.org/abs/2412.11664)
- **Self-training for concise reasoning (Feb 27, 2025):** best-of-N shortest-correct traces plus few-shot conditioning; −30% output tokens across 5 model families on GSM8K/MATH at maintained accuracy. — [arXiv 2502.20122](https://arxiv.org/abs/2502.20122)
- **DART (Nov 3, 2025):** supervised difficulty-adaptive truncation distilled from stronger models; R1-Distill-Qwen-7B on GSM8K: 81.2% truncation, 5.33× compute acceleration, accuracy preserved or improved. — [DART](https://arxiv.org/abs/2511.01170)
- **CRISP (Apr 19, 2026):** intrinsic attention saliency at the end-of-thinking token guides pruning; −50–60% tokens "without compromising accuracy" on math. — [CRISP](https://arxiv.org/abs/2604.17297)
- **Prompt-only terse reasoning:**
  - Chain of Draft (Feb 25, 2025): matches or beats CoT with "as little as 7.6% of the tokens". — [CoD](https://arxiv.org/abs/2502.18600)
  - Sketch-of-Thought (Mar 7, 2025): up to −84% tokens across 18 datasets with "minimal accuracy loss". — [SoT](https://arxiv.org/abs/2503.05179)
- **Latent or compressed thoughts:**
  - LightThinker (Feb 21, 2025): compresses thoughts into gist tokens, lowering peak memory and inference time at "competitive accuracy". — [LightThinker](https://arxiv.org/abs/2502.15589)
  - Coconut (Dec 9, 2024): continuous latent reasoning; better accuracy-efficiency on logic tasks that need search. — [Coconut](https://arxiv.org/abs/2412.06769)

#### Prompt-level token budgets
- **TALE (Dec 24, 2024):** estimates a per-problem token budget and puts it in the prompt. The abstract only says "slight performance reduction". The commonly quoted ~68% cost cut for <5% accuracy loss is not in the current abstract (see Gaps). — [TALE](https://arxiv.org/abs/2412.18547)
- **TALE-EP under cost-of-pass:** "shows some promise", whereas majority voting and self-refinement "rarely justify the costs". — [Cost-of-Pass](https://arxiv.org/abs/2504.13359)

#### Early exit of reasoning (training-free, or a light plug-in)

| Method (date) | Mechanism | Token reduction | Accuracy effect | Source |
|---|---|---|---|---|
| DEER (Apr 22, 2025) | At transition points ("Wait") induce a trial answer; exit if confident. 11 LRMs, 10 benchmarks incl. GPQA, AIME, LiveCodeBench | −19.1% to −80.1% CoT length | **+0.3 to +5.0 pp** | [DEER](https://arxiv.org/abs/2504.15895) |
| Hidden-state probe (Apr 7, 2025) | Linear probe on hidden states predicts correctness of intermediate answers (calibrated) | −24% | none lost | [Probe](https://arxiv.org/abs/2504.05419) |
| Certaindex / Dynasor (Dec 30, 2024) | Answer-stabilization metric used as a serving scheduler | up to 50% compute savings; 3.3× throughput | no accuracy drop | [Certaindex](https://arxiv.org/abs/2412.20993) |
| PUMA (May 17, 2026) | Semantic-redundancy detector + answer verification, so it does not exit before self-correction finishes | −26.2% average over 5 LRMs | accuracy preserved | [PUMA](https://arxiv.org/abs/2605.17672) |
| CALM (Oct 19, 2026) | Masks logits of "Wait / Alternatively / Hmm" once trial answers are consistent, instead of hard truncation | −18.6% to −25.0% | +0.07 to +1.48 pp | [CALM](https://arxiv.org/abs/2610.25908) |
| OS-Pruner (Jul 13, 2026) | Optimal-stopping utility (accuracy vs length) with a user knob | −20% to −60% | "minimal" sacrifice | [OS-Pruner](https://arxiv.org/abs/2607.11089) |
| ReCo (Aug 5, 2026) | PRM-driven KV compression + reflection-token penalty + confidence early stop; 3 models, 6 benchmarks | −37% to −65% generated tokens; 2.08–2.35× lower latency vs full CoT | "largely" preserved | [ReCo](https://arxiv.org/abs/2608.04771) |

- **OTBench / Step-wise Truncation Probing (Oct 9, 2026):**
  - Defines an "Overthinking Ratio" (share of thinking tokens after the earliest max-accuracy prefix).
  - Builds token-accuracy Pareto frontiers for routing, fixed budgets and dynamic early exit, from 70.5B probe tokens across 6 models.
  - Useful as an upper-bound reference for what early exit can recover.
  - — [OTBench](https://arxiv.org/abs/2610.13570)

#### Overthinking and underthinking analyses
- **"Do NOT think that much for 2+3=?" (Dec 30, 2024):** first systematic study of overthinking in o1-like models. Proposes outcome and process efficiency metrics plus self-training mitigations, tested on GSM8K, MATH500, GPQA and AIME. — [arXiv 2412.21187](https://arxiv.org/abs/2412.21187)
- **"Stop Overthinking" survey (Mar 20, 2025):** taxonomy of model-based, output-based and prompt-based efficient reasoning; companion GitHub list. — [Survey](https://arxiv.org/abs/2503.16419)
- **When More is Less (Feb 11, 2025):** accuracy follows an inverted U in CoT length. The optimal length grows with task difficulty and shrinks with model capability (a "simplicity bias"). — [arXiv 2502.07266](https://arxiv.org/abs/2502.07266)
- **Revisiting o1-like test-time scaling (Feb 17, 2025):** for QwQ, R1 and LIMO, correct solutions are often *shorter* than incorrect ones, and longer CoTs contain more self-revisions that hurt. Parallel scaling gives better coverage; proposes "Shortest Majority Vote". — [arXiv 2502.12215](https://arxiv.org/abs/2502.12215)
- **Underthinking (Jan 30, 2025):** frequent thought switching correlates with wrong answers. A thought-switching penalty (TIP) decoding raises accuracy without fine-tuning. — [arXiv 2501.18585](https://arxiv.org/abs/2501.18585)
- **Between under- and overthinking (Apr 30, 2025):** models overthink easy problems and underthink hard ones, i.e. they misjudge difficulty. — [arXiv 2505.00127](https://arxiv.org/abs/2505.00127)
- **Epoch AI (Apr 2025):** benchmark response length grows about 5×/yr for reasoning models vs 2.2×/yr for non-reasoning models. Reasoning-model responses average about 8× longer. — [Epoch](https://epoch.ai/data-insights/output-length)

#### Parallel vs sequential test-time compute
- **Snell et al. (Aug 6, 2024):** compute-optimal per-prompt allocation is more than 4× more efficient than best-of-N. In FLOPs-matched tests, test-time compute beats a 14× larger model when the base model has non-trivial success. — [arXiv 2408.03314](https://arxiv.org/abs/2408.03314)
- **"Mirage of test-time scaling" (Jun 4, 2025):** extending thinking ("Wait") is non-monotonic: it first helps, then hurts. At the same budget, parallel thinking with majority vote is up to 20% more accurate than extended thinking. — [arXiv 2506.04210](https://arxiv.org/abs/2506.04210)
- **NoThinking (Apr 14, 2025):**
  - At equal tokens, skipping thinking beats Thinking, e.g. 51.3 vs 28.9 on ACM23 at a 700-token budget.
  - NoThinking × N in parallel with confidence or verifier selection matches Thinking at up to 9× lower latency.
  - — [arXiv 2504.09858](https://arxiv.org/abs/2504.09858)
- **ParaThinker (Aug 30, 2025):** native parallel thinking with 8 paths gives +12.3 pp (1.5B) and +7.5 pp (7B) over sequential, at +7.1% latency. — [ParaThinker](https://arxiv.org/abs/2509.04475)
- **"Art of Scaling TTC" (Dec 1, 2025):** 30B+ tokens across 8 open LLMs (7B–235B). No single strategy dominates; the optimum depends on model type (short- vs long-horizon reasoners), problem difficulty and budget. — [arXiv 2512.02008](https://arxiv.org/abs/2512.02008)

#### Self-consistency with early stopping / confidence weighting
- **ESC (Jan 19, 2024):** cuts the number of samples by 33.8% (MATH), 80.1% (GSM8K), 76.8% (StrategyQA), etc., with "comparable" accuracy. — [ESC](https://arxiv.org/abs/2401.10480)
- **Adaptive-Consistency (May 19, 2023):** up to 7.9× fewer samples, with average accuracy drop below 0.1%, over 17 datasets. — [arXiv 2305.11860](https://arxiv.org/abs/2305.11860)
- **DeepConf (Aug 21, 2025):** filters traces by model-internal confidence during or after generation. On AIME 2025, DeepConf@512 reaches up to 99.9% accuracy with up to −84.7% tokens vs full parallel thinking. Tested on Qwen3 and GPT-OSS. — [DeepConf](https://arxiv.org/abs/2508.15260)
- **Seer Self-Consistency (Nov 12, 2025):** a "System 1" pass estimates answer entropy to size the System-2 self-consistency budget in advance. Up to −47% tokens and −43% latency, with no significant loss. — [SeerSC](https://arxiv.org/abs/2511.09345)
- **Self-certainty BoN (Feb 25, 2025):** a reward-model-free best-of-N score that scales with N and works on open-ended tasks. — [arXiv 2502.18581](https://arxiv.org/abs/2502.18581)
- **RAD (Jun 22, 2026):** an MoE-specific multi-rollout selector that uses router states at answer anchors and needs no answer strings. On par with majority vote (73.9 vs 73.6) across gpt-oss and Qwen3-MoE. — [RAD](https://arxiv.org/abs/2606.22798)

#### Frontier 2026 token-per-task data points
- **AA, GPT-6 Astra (Sep 2026):** about 27k output tokens per Intelligence Index task at max effort, vs about 78k for Claude Fable 5.1 (max), for the same score. — [AA article](https://artificialanalysis.ai/articles/benchmarking-gpt-6-astra)
- **AA, Claude Opus 4.6 (Feb 6, 2026):** 58M output tokens for the Index in adaptive thinking, vs 29M for Opus 4.5 and 130M for GPT-5.2 at xhigh. — [AA article](https://artificialanalysis.ai/de/articles/opus-4-6-everything-you-need-to-know)

### Inferences
- **Size of the token prize.** Qwen3.8-27B (xhigh) is in the top ~10% most verbose models on AA (200M vs 82M median). The target therefore probably needs no more tokens than the 27B. Per decoded token it is also far cheaper: 3B vs 27B active parameters, about 9× fewer weight-FLOPs, and AA measured 134 vs 44 tok/s on the same provider. "Much lower token cost" looks achievable if accuracy can be raised. The accuracy gap (about 10 pp on LCB v6, about 3 pp on GPQA) is the binding constraint, not tokens.
- **Uniform length pressure is not enough.** It cuts tokens at a small hard-set cost (ThinkPrune: −2% at half length). Methods that *gain* hard-set accuracy while cutting tokens all add difficulty-awareness or capability transfer:
  - When2Think: +10 pp Pass@3 on AIME24 at −27.9% tokens.
  - Lightning Weave: +12.5 pp LCB at −9.6% tokens.
  - TRAAC: +8.4 pp at −36.8%.
  - Recommended recipe: a distillation/RL stage from a stronger teacher (e.g., on-policy distillation from Qwen3.8-27B) plus a correctness-gated, difficulty-aware length bonus. Avoid a raw length penalty.
- **Cheap inference-time add-ons stack with training-time methods.** DEER/CALM/PUMA-style early exit typically gives 20–30% with no accuracy loss. Confidence-gated parallel sampling (DeepConf, SeerSC) only pays off when parallel sampling is used at all.
- **Prefer RL-learned think/no-think over budgets for the agent's System-1 path.** Qwen3.6 drops the `/think` soft switch, so the per-request `enable_thinking` toggle, or a trained mode token (RPO, Thinkless, AdaptThink), is the native lever. RPO's claim of ~2% training compute makes a root-token think/no-think decision cheap to add.

### Gaps
- No public numeric thinking-budget curve (accuracy at 1K/2K/4K/8K/16K/32K) for Qwen3, Qwen3.6 or Qwen3.8; Qwen3 reports it only as a figure.
- Qwen3.8 publishes no token counts per `reasoning_effort` level. AA reports only xhigh.
- TALE's widely cited "68.64% token reduction with <5% accuracy loss" is not in the current arXiv abstract (v6), so I did not cite it as verified.
- No published head-to-head of these methods on Qwen3.6-35B-A3B specifically.
- No 2026 paper found that applies length-controlled RL to a ≥30B MoE agent model and reports AIME/GPQA/LCB together with SWE-bench/Terminal-Bench.

---

## Q2. System-1 / System-2 dual-process designs and how systems decide when to escalate from fast to slow

### Takeaway
Three architecture patterns dominate, each with a different escalation mechanism:
- **One model, two modes:** Qwen3-style fusion, Dualformer, reasoning-effort levels, learned root-token think/no-think. Escalation is a learned mode token or an external router.
- **Small-fast + large-slow pairs:** cascades, routers, token-level routing such as R2R and CITER, and speculative reasoning. Escalation comes from uncertainty, self-verification or a learned router.
- **Metacognitive monitors:** a cheap calibrated System-1 model that gates, monitors or interrupts System-2 (SOMA, S1-MAS, SAG, System1-Arena).

Measured savings are typically 30–90% of slow-path tokens with accuracy held or improved. For agents, escalation should be decided per step and with session history. Switching *models* breaks prompt caches, so same-model mode switching is structurally cheaper.

### Cited Findings

#### Single-model dual mode
- **Qwen3 fusion (May 14, 2025):** one checkpoint carries both modes, with a large System-2 gain on hard math (Qwen3-30B-A3B AIME'25: 70.9 thinking vs 21.6 non-thinking), but a smaller gain on GPQA-D (65.8 vs 54.8). — [Qwen3 TR](https://arxiv.org/html/2505.09388)
- **Dualformer (Meta, Oct 13, 2024):** trained on randomly dropped trace parts so one Transformer can run fast, slow or auto mode. On 30×30 mazes:
  - slow mode: 97.6% optimal with 45.5% fewer steps than Searchformer (93.3%);
  - fast mode: 80% vs 30% for a solution-only model;
  - auto mode: 96.6% optimal with 59.9% fewer steps.
  - — [Dualformer](https://arxiv.org/abs/2410.09918)
- **GPT-5 (Aug 2025):** a "unified system" of a fast model (gpt-5-main), a thinking model (gpt-5-thinking) and a real-time router. The router decides on conversation type, complexity, tool needs and explicit intent. It is trained continuously on user model-switches, preference rates and measured correctness. Source: system card, read via search snippet because the page returned 403. — [GPT-5 system card](https://openai.com/index/gpt-5-system-card)
- **Kimi K3 (Jul 27, 2026):** 2.8T MoE with 104B active (16 of 896 experts). Post-training is RL "across … multiple reasoning-effort levels". At launch only max effort is served; low and high modes are promised later. — [Kimi K3 TR](https://arxiv.org/abs/2607.24653), [Kimi blog](https://www.kimi.ai/blog/kimi-k3)
- **TempoMoE (Oct 19, 2026), most relevant MoE finding:**
  - Fast and slow thinking show distinct, stable *expert-routing* patterns.
  - Suppressing "slow-thinking experts" shortens chains; suppressing "fast" experts lengthens them, i.e. a causal effect.
  - A training-free difficulty-conditioned router reweighting within the original top-k gives up to +2.6 pp accuracy and −35.4% output tokens on Kimi-VL, Qwen3-VL and ERNIE-4.5-VL.
  - — [TempoMoE](https://arxiv.org/abs/2610.25999)
- **Star Elastic (May 8, 2026):** selects a nested MoE submodel per phase (thinking vs answering): up to +16% accuracy and 1.9× lower latency on Nemotron Nano v3 30B-A3.6B. — [Star Elastic](https://arxiv.org/abs/2605.07182)

#### Fast-small + slow-large pairs (cascades, routers, collaborative decoding)
- **FrugalGPT (May 9, 2023):** a learned LLM cascade matches GPT-4 at up to 98% lower cost, or gets +4% accuracy at equal cost. — [FrugalGPT](https://arxiv.org/abs/2305.05176)
- **AutoMix (Oct 19, 2023):** few-shot self-verification of the small model's answer plus a POMDP router; >50% compute reduction at comparable performance. — [AutoMix](https://arxiv.org/abs/2310.12963)
- **RouteLLM (Jun 26, 2024):** strong/weak routers trained on preference data cut cost by over 2× "in certain cases" without quality loss, and transfer to new model pairs. — [RouteLLM](https://arxiv.org/abs/2406.18665)
- **LM cascades with token-level uncertainty (Apr 15, 2024):** naive sequence-level uncertainty has length bias. Learned post-hoc deferral rules over token-level uncertainties, plus embeddings from both models, beat simple aggregation. — [arXiv 2404.10136](https://arxiv.org/abs/2404.10136)
- **FS-GEN (Jun 18, 2024):** across collaborative decoding methods, fewer than about 20% of interactions need System 2. The S1/S2 distinction is mainly next-token *uncertainty*, with a scaling law in the parameter ratio. — [FS-GEN](https://arxiv.org/abs/2406.12295)
- **R2R token routing (May 27, 2025):** a neural router sends only "path-divergent" tokens from R1-1.5B to R1-32B. At 5.6B average active parameters it beats R1-14B and is 2.8× faster than R1-32B at comparable accuracy. — [R2R](https://arxiv.org/abs/2505.21600)
- **CITER (Feb 4, 2025):** an RL-trained token-level router between SLM and LLM. — [CITER](https://arxiv.org/abs/2502.01976)
- **Speculative Thinking (Apr 12, 2025):** delegates reflective steps (after "\n\n" + "wait") to a 32B model. The 1.5B model's MATH500 goes 83.2%→89.4% while length drops 5439→4583 tokens (−15.7%). — [arXiv 2504.12329](https://arxiv.org/abs/2504.12329)
- **SpecReason (Apr 10, 2025):** a small model drafts reasoning steps and the large model verifies semantically: 1.4–3.0× speedup with +0.4–9.0% accuracy, and a further 8.8–58.0% latency cut when combined with speculative decoding. This is semantic, not token-exact. — [SpecReason](https://arxiv.org/abs/2504.07891)
- **Reward-guided speculative decoding (Jan 31, 2025):** a PRM decides per step whether to call the target model; up to 4.4× fewer FLOPs and up to +3.5 accuracy vs parallel decoding baselines. — [RSD](https://arxiv.org/abs/2501.19324)
- **SwiftSage (May 27, 2023):** a small fine-tuned "Swift" policy plus an LLM "Sage" planner, with heuristic switching; beats ReAct and Reflexion on 30 ScienceWorld tasks. — [SwiftSage](https://arxiv.org/abs/2305.17390)
- **SOFAI-LM (Aug 25, 2025):** generalizes the SOFAI architecture. A metacognitive module gives a fast LLM iterative feedback and engages a slow LRM only when needed. It matches or beats standalone LRMs on graph coloring and code debugging in "considerably less time". — [SOFAI-LM](https://arxiv.org/abs/2508.17959)
- **System-1.x Planner (Jul 19, 2024):** a controller splits a plan into sub-goals and labels each easy (System 1) or hard (System 2, i.e. search). A hybridization factor x is the user knob. It beats pure S1, pure S2 and A* on Maze and Blocksworld. — [System-1.x](https://arxiv.org/abs/2407.14414)
- **SCALE (Nov 29, 2025):** sub-problem-level System 1/2 assignment. AIME25 57.50→71.25 (+13.75 pp) while cutting compute by 33–53%. — [SCALE](https://arxiv.org/abs/2512.00466)

#### Escalation signals, learned routers and their measured trade-offs
- **HRBench (May 27, 2026):** the first controlled comparison of mode-switch strategies. It covers 12 settings (prompt-based selection, external routing, speculative execution × 4 training regimes), 6 models from Qwen3.5-2B to Kimi-K2.5 (1.1T) and 12+ re-implemented methods. Findings:
  - prompt-based selection often gives the *best token-accuracy trade-off*;
  - routing gives *more stable* cost reduction;
  - speculative execution raises accuracy at higher token cost;
  - the best strategy varies with model scale and domain.
  - — [HRBench](https://arxiv.org/abs/2605.28398)
- **Self-Route (May 27, 2025):** a capability-aware probe on hidden states taken *before* generation picks general vs reasoning mode; −30–55% tokens at reasoning-model accuracy. Router trained on the Gradient-10K difficulty dataset. — [Self-Route](https://arxiv.org/abs/2505.20664)
- **Hidden-state correctness probes (Apr 7, 2025):** "highly calibrated" and predictive *before* the answer is formed. As an exit gate they cut tokens by 24% with no loss. — [arXiv 2504.05419](https://arxiv.org/abs/2504.05419)
- **System1-Arena (Oct 15, 2026):** 92k decision questions across 105 tasks plus 138 closed-loop environments, including SWE-bench control.
  - Single-forward-pass System-1 decision models match or beat fast closed LLMs on non-agentic accuracy, are *better calibrated*, and run at ¼ the latency.
  - A reasoning LLM is the most accurate, at about **70× the cost** of a fast LLM.
  - In closed loops, small System-1 models "rarely succeed"; large open LLMs reach up to 32.9%.
  - — [System1-Arena](https://arxiv.org/abs/2610.21244)
- **SOMA (Oct 19, 2026):** a System-1 model *continuously monitors* System-2 reasoning. A controller first reminds S2 and takes over only if the behavior persists. Results: up to −36% reasoning tokens with higher average accuracy; abstention on unanswerable questions rises from 70.5% to 87.2%. — [SOMA](https://arxiv.org/abs/2610.26793)
- **S1-MAS (Oct 6, 2026):** System-1 models handle bounded coordination decisions in a multi-agent system: task selection, termination, evidence reading. Against AgentVerse, DyLAN and SelfOrg: −44.9% to −97.2% GPT-4o tokens and −37.8% to −93.0% latency, with "superior accuracy". — [S1-MAS](https://arxiv.org/abs/2610.08155)
- **SAG gated critique (Oct 5, 2026):** a value-of-information gate on action entropy and top-2 margin calls the critic only when it is worth the cost.
  - ALFWorld success 24.6%→78.4% at a ReAct-comparable token budget, i.e. 3.1× normalized token efficiency.
  - A 7B actor with a 3B critic ≈ a 14B actor without critique.
  - — [SAG](https://arxiv.org/abs/2610.07335)
- **Ares, per-step effort router for agents (Mar 9, 2026):** predicts the *lowest sufficient* reasoning level per step from interaction history. Up to −52.7% reasoning tokens vs fixed high effort on TAU-Bench, BrowseComp-Plus and WebArena, with "minimal degradation". Static low effort everywhere degrades significantly. — [Ares](https://arxiv.org/abs/2603.07915)
- **ARES (RTL agents, Jul 30, 2026):** escalates to deeper effort *only when progress at lower effort stalls*, using a fitted patience counter. FoM improves 23–27% vs 16–23% for the best fixed effort, at equal normalized cost. — [ARES](https://arxiv.org/abs/2607.27879)
- **ALAR (Jun 1, 2026):** compact latent reasoning on routine agent turns, escalating to explicit CoT on hard decisions. −43.6% tokens on search and −84.6% on tool use, at comparable or better accuracy. — [ALAR](https://arxiv.org/abs/2606.02871)
- **Speculative Uncertainty gate (Sep 4, 2026):** a small draft model scores a black-box agent's trajectory in one forward pass, producing a failure-likelihood signal for routing or veto. A pre-execution veto on Qwen3-Coder-480B and Claude 3.5 Sonnet agents cuts execution errors by 6–8 pp and token cost by 14–19%. — [SU](https://arxiv.org/abs/2609.05274)
- **Stateful routing, "Whose turn is it?" (Oct 13, 2026):**
  - Giving routers *session history* raises pass rate at matched offload, staying within 1 pp of always-expensive.
  - The cheaper model's per-task cost advantage narrows from 6.8× to 3.2× on long sessions.
  - Downward switches (expensive→cheap) recover the prompt-cache cost within 2 calls; **upward switches do not break even** under the evaluated conditions.
  - — [arXiv 2610.19292](https://arxiv.org/abs/2610.19292)
- **Risa (Aug 23, 2026):** native MoE router traces act as a behavioral signal for steering and selection in SWE agents. gpt-oss family SWE-bench Verified: 44.9%→48.2% resolved. It transfers to Qwen3.6. — [Risa](https://arxiv.org/abs/2608.22191)

### Inferences
- **Same-model escalation should be the default.** For the target, the System-1/System-2 pair is the same Qwen3.6-35B-A3B with `enable_thinking` off/on (or a low/high effort token), not a separate small model. Switching models breaks the prompt cache (upward switches never broke even), whereas a mode switch inside one model keeps the shared prefix KV. A second "System-1" model is only worth it as a monitor or gate (SOMA, SU, System1-Arena style), where one forward pass yields a calibrated probability.
- **Escalation signals ranked by evidence:**
  1. Per-step learned router on interaction history: Ares −52.7%; HRBench says routing gives the most *stable* savings.
  2. Hidden-state probes, either pre-generation (Self-Route) or mid-trace (correctness probe). They are calibrated and cheap.
  3. Trial-answer confidence at transition tokens (DEER) or answer stabilization (Certaindex).
  4. Progress-stall triggers for long-horizon agents (ARES patience counter).
  
  Pure prompt-based selection gives good trade-offs but is less stable.
- **The MoE base has its own fast/slow lever.** TempoMoE's "fast/slow experts" and Risa/RAD routing-trace signals suggest a training-free knob: reweight routing within top-k by estimated difficulty. Router states can also serve as an escalation signal. This needs validation on Qwen3.6's 256-expert router (TempoMoE tested VL MoEs only).
- **The System-1 fast path loses most on hard math.** In Qwen3-30B-A3B, AIME drops 49 pp in non-thinking mode, so fast-path routing must be conservative on math and code and aggressive on retrieval, tool plumbing and formatting turns, where ALAR (−84.6% on tool use) and the GPT-5.6 office study (Q4) show low effort suffices.

### Gaps
- No public numbers on GPT-5's router accuracy or escalation rate.
- No published study of Qwen3.6 or Qwen3.8 accuracy when the reasoning mode is switched per turn inside one agent trajectory: does flipping `enable_thinking` between turns keep the chat-template prefix byte-identical, and so keep the KV cache?
- The original SOFAI paper (2021) was not re-verified here; only SOFAI-LM (2025) is cited.
- The calibration quality of Qwen-family self-confidence as an escalation trigger is not reported. DEER and DeepConf include Qwen3 but report aggregate gains, not calibration curves.

---

## Q3. Decoding-level speedups that preserve outputs (speculative decoding, EAGLE-3, Medusa, MTP heads incl. Qwen3.6 and DeepSeek MTP, lookahead, LayerSkip self-speculation): speedups and exactness

### Takeaway
Standard speculative decoding with draft-model, EAGLE, MTP or lookahead drafts and the standard verify/rejection rule is **distribution-preserving**. Under greedy decoding it is token-identical up to floating-point and batch-shape numerics. These methods reduce latency, not tokens.

Typical end-to-end gains:
- 1.8–2.0× for EAGLE-3 on Qwen3 under vLLM;
- about 1.8× tokens/s for DeepSeek-V3 MTP (85–90% second-token acceptance);
- up to 6.5× in EAGLE-3's best case;
- smaller at large batch sizes (1.38× at batch 64).

For MoE targets, verification activates the *union* of experts across draft tokens, which erodes the speedup. 2026 MoE-aware verification (MoE-Spec, EVICT, EcoSpec) recovers 10–30%.

Lossy relaxations (nucleus or relaxed acceptance, semantic speculation) can be faster per round but can **lengthen** reasoning outputs and lower accuracy.

### Cited Findings

#### Exactness guarantees
- **Leviathan et al. (Nov 30, 2022):** speculative decoding samples "without any changes to the outputs" (same distribution); 2–3× on T5-XXL with "identical outputs". — [arXiv 2211.17192](https://arxiv.org/abs/2211.17192)
- **Chen et al. (Feb 2, 2023):** modified rejection sampling "preserves the distribution of the target model within hardware numerics"; 2–2.5× on Chinchilla-70B. — [arXiv 2302.01318](https://arxiv.org/abs/2302.01318)
- **vLLM docs (accessed 2026-10-21):**
  - "theoretically lossless up to the precision limits of hardware numerics".
  - Greedy sampling with spec decode "matches greedy sampling without it".
  - Caveats: "Changes in batch size may cause variations in logprobs", and vLLM "does not currently guarantee stable token log probabilities".
  - "FLy" verification is explicitly *lossy* ("not distribution preserving").
  - Supported methods: EAGLE (incl. `eagle3`), MTP, draft model, PARD, n-gram, suffix decoding, dynamic spec decoding, adaptive verification.
  - — [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- **Medusa (Jan 19, 2024):**
  - Medusa-1, with heads on a *frozen* backbone, is lossless: >2.2×.
  - Medusa-2, which co-trains the backbone, reaches 2.3–3.6×. It changes the model, so it is not lossless relative to the original weights.
  - The optional "typical acceptance" scheme is also not exact.
  - — [Medusa](https://arxiv.org/abs/2401.10774)
- **Lookahead decoding (Feb 3, 2024):** "an exact, parallel decoding algorithm" with no draft model; up to 1.8× on MT-bench and 4× on code completion with strong scaling across GPUs. — [Lookahead](https://arxiv.org/abs/2402.02057)
- **LayerSkip (Apr 25, 2024):** early-exit self-speculation; up to 2.16× on CNN/DM summarization, 1.82× on coding, 2.0× on TOPv2. It needs a training recipe (layer dropout + early-exit loss), so the model differs from an untouched checkpoint. Self-speculative verification is exact relative to the trained model. — [LayerSkip](https://arxiv.org/abs/2404.16710)

#### EAGLE-3 and MTP numbers
- **EAGLE-3 (Mar 3, 2025):** direct token prediction plus multi-layer feature fusion ("training-time test"). Up to 6.5× speedup, about 1.4× over EAGLE-2, and 1.38× throughput at batch 64 in SGLang. Evaluated on chat and reasoning models. — [EAGLE-3](https://arxiv.org/abs/2503.01840)
- **AngelSlim (Tencent, Feb 7, 2026):** a training-aligned speculative decoding framework with "1.8× to 2.0× throughput gains without compromising output correctness". A search snippet of the PDF reports Qwen3 EAGLE-3 average acceptance length of 1.74–2.2 on vLLM. — [AngelSlim](https://arxiv.org/abs/2602.21233)
- **DeepSeek-V3 MTP (Dec 27, 2024):** MTP depth D=1; second-token acceptance rate "between 85% and 90%"; used for speculative decoding it delivers "1.8 times TPS". — [DeepSeek-V3 §5.4.3](https://arxiv.org/abs/2412.19437)
- **Qwen3.6-35B-A3B native MTP serving flags (card, Apr 2026):**
  - vLLM ≥0.19.0: `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`.
  - SGLang ≥0.5.10: `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.
  - The card publishes no speedup or acceptance numbers.
  - — [HF card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Qwen3.8-27B MTP:** the card lists MTP but no serving flags. An anecdotal llama-server run with `--spec-type draft-mtp` was about 72% faster than the default GGUF on a DGX Spark (practitioner report). — [HF card](https://huggingface.co/Qwen/Qwen3.8-27B), [Simon Willison](https://simonwillison.net/2026/Aug/16/qwen-38-27b/)
- **TensorRT-Edge-LLM docs (search snippet):** MTP raised Qwen3.5 VLM batch-size-1 generation throughput by 1.21–2.12× over vanilla decoding. — [NVIDIA TensorRT-Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/0.8.0/user_guide/performance/performance-benchmarks.html)
- **Nemotron 3 Super (Apr 14, 2026):** 120B / 12B-active hybrid Mamba-attention LatentMoE with MTP layers for "native speculative decoding". Throughput is up to 2.2× GPT-OSS-120B and 7.5× Qwen3.5-122B at comparable accuracy (includes architecture effects, not only MTP). — [Nemotron 3 Super](https://arxiv.org/abs/2604.12374)
- **Uno, diffusion-augmented AR (Sep 3, 2026):** lightweight diffusion weights draw several tokens from the AR distribution. The Ψ-Spec samplers are lossless, giving up to 3× over the base AR model and higher throughput than leading spec-decoding at every evaluated batch size. — [Uno](https://arxiv.org/abs/2609.04010)

#### MoE-specific speculative decoding (directly relevant to a 256-expert, 8+1-active target)
- **MoE-Spec (Feb 17, 2026):** large draft trees activate many unique experts, which raises memory traffic. A fixed per-layer expert budget during verification gives +10–30% throughput over EAGLE-3 "at comparable quality". The budget can be tightened for more speed at some accuracy cost, so it is not strictly lossless. — [MoE-Spec](https://arxiv.org/abs/2602.16052)
- **EVICT (May 1, 2026):** a training-free, hyperparameter-free, **lossless** method that truncates the draft tree to its cost-effective prefix before MoE verification. Up to 2.35× over AR and on average 1.21× over EAGLE-3; integrated with SGLang. — [EVICT](https://arxiv.org/abs/2605.00342)
- **EcoSpec (Jul 14, 2026):** draft selection that accounts for expert-activation cost, avoiding "expert scattering". Tested on DeepSeek-V3.1, Qwen3-235B-A22B and GPT-OSS-120B, with the verification rule unchanged; the abstract is truncated, so no number is given here. — [EcoSpec](https://arxiv.org/abs/2607.12696)
- **XShare (Feb 6, 2026):** batch-aware expert selection; −30% expert activations and up to +14% spec-decode throughput. It alters expert selection, so it is approximate. — [XShare](https://arxiv.org/abs/2602.07265)
- **S2-MoE (Aug 15, 2026):** edge self-speculative MoE in llama.cpp; up to 5.3×, about 2.0× on average. — [S2-MoE](https://arxiv.org/abs/2608.15018)

#### Lossy or semantic variants, and their measured pitfalls
- **"Acceptance Is Not Speed" (Oct 16, 2026):** across 12 target-drafter pairs, relaxed (lossy) verification accepts 1.34× more tokens per round but produces **1.54× longer completions**, mostly longer reasoning traces. This largely cancels the gain. Rules that cut verifier rounds "typically drop" accuracy substantially. — [arXiv 2610.23818](https://arxiv.org/abs/2610.23818)
- **Nucleus Speculative Decoding (Oct 6, 2026):** accepts draft tokens that fall inside the target's nucleus. Up to 5.16× over AR and 3.15× over standard SD, with a *provable* per-step total-variation error equal to the acceptance gain; accuracy improved on 5 of 6 models. Not exact. — [NSD](https://arxiv.org/abs/2610.07822)
- **AsymSpec (Aug 26, 2026):** the drafter sees the full context while the verifier sees compressed context. About 90% of full-context accuracy, at 1.3–1.7× throughput and 0.2–0.3× compute. Lossy by design. — [AsymSpec](https://arxiv.org/abs/2608.26004)

### Inferences
- **Serving default for the target:** native MTP with lossless verification, using the card's 2–4 draft tokens. It costs no accuracy. Gains are likely about 1.5–2× at low batch and shrink at high batch. For the MoE target this is an inference from the DeepSeek MTP 1.8× and Qwen3 EAGLE-3 1.8–2.0× data points; no Qwen3.6 number is published.
- **Lossless MoE-aware verification is the next increment:** EVICT, roughly 1.2× over EAGLE-3. MoE-Spec and XShare are approximate and should be treated like any other accuracy-affecting change.
- **Avoid lossy acceptance for long-reasoning workloads.** The 2026 length-inflation result shows that relaxed acceptance can *increase* tokens per task, which cuts against the goal.
- **Exactness caveat for a "deterministic compiler" framing:** even lossless spec decode is only token-identical under greedy decoding if kernels are batch-invariant. vLLM explicitly does not guarantee stable logprobs across batch sizes.

### Gaps
- No published acceptance-length or speedup numbers for Qwen3.6-35B-A3B MTP, with either vLLM `qwen3_next_mtp` or SGLang NEXTN.
- No head-to-head of EAGLE-3 vs native MTP on Qwen3.6/3.8.
- No published result on how speculative decoding interacts with Gated DeltaNet (linear-attention) state rollback cost on rejected drafts for the Qwen3.5/3.6 hybrid architecture.

---

## Q4. Agent-level token economics: prompt/prefix caching, context compaction, structured outputs, tool-output truncation, plan reuse, parallel tool calls, tokens-per-task on agent benchmarks, cost-of-pass

### Takeaway
In agents, **input tokens dominate cost**: about 100:1 input:output at Manus, and 1000× more tokens than chat-style coding on SWE-bench Verified. So the biggest levers are:
- prefix-cache hit rate (cached input is about 10× cheaper; Kimi K3 reports >90% cache hits for coding);
- context editing and compaction (−84% tokens on a 100-turn search; +5–12 pp resolve rate when compaction is trained or triggered well);
- tool-output and tool-definition minimization (−50% to −98.7% on specific workflows);
- per-step effort selection (−43% to −53% reasoning tokens).

Cost-of-pass (expected $ per correct solution) is the right metric. Token usage is highly stochastic (up to 30× run-to-run) and accuracy often peaks at *intermediate* cost.

### Cited Findings

#### Metrics
- **Cost-of-pass (Apr 17, 2025):**
  - Definition: expected monetary cost per correct solution; the "frontier cost-of-pass" is the minimum across models and human experts.
  - For complex quantitative tasks the frontier cost roughly halved every few months.
  - Majority voting and self-refinement "rarely justify the costs".
  - — [Cost-of-Pass](https://arxiv.org/abs/2504.13359)
- **Efficient Agents (OPPO, Jul 24, 2025):** on GAIA, keeps 96.7% of OWL's performance while cutting cost from $0.398 to $0.228 (+28.4% cost-of-pass). — [Efficient Agents](https://arxiv.org/abs/2508.02694)
- **Token Efficiency Index (Aug 7, 2026):** an organization-level 0–100 score built from cache hit rate, cache amortization ratio and premium-model share. — [TEI](https://arxiv.org/abs/2608.07304)

#### Where agent tokens go (tokens per task)
- **"How Do AI Agents Spend Your Money?" (Apr 24, 2026; v3 Oct 2, 2026):** 8 frontier LLMs on SWE-bench Verified.
  - Agentic tasks use "1000× more tokens than code reasoning and code chat"; input tokens drive cost.
  - Runs on the same task differ by up to 30× in tokens. Accuracy "often peaks at intermediate cost and saturates at higher costs".
  - Kimi-K2 and Claude-Sonnet-4.5 use over 1.5M more tokens per task than GPT-5.
  - Models predict their own token use poorly (correlation ≤0.39) and underestimate it.
  - — [arXiv 2604.22750](https://arxiv.org/abs/2604.22750)
  - Secondary summaries quote an average of 4.17M tokens and $1.857 per task. — [tokenade summary](https://tokenade.net/en/case-studies/swe-bench-agent-token-study), [Stanford DEL page](https://digitaleconomy.stanford.edu/publication/how-do-ai-agents-spend-your-money)
- **Manus (Jul 18, 2025):**
  - Average input:output ratio "around 100:1"; about 50 tool calls per typical task.
  - Calls "KV-cache hit rate … the single most important metric for a production-stage AI agent".
  - Claude Sonnet cached input costs $0.30/MTok vs $3/MTok uncached.
  - — [Manus blog](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- **HAL, Holistic Agent Leaderboard (Oct 13, 2025):** 21,730 rollouts across 9 models × 9 benchmarks (coding, web, science, customer service) for about $40k, with 2.5B tokens of logs released. "Higher reasoning effort reduc[ed] accuracy in the majority of runs". — [HAL](https://arxiv.org/abs/2510.11977)
- **AA token-per-task (Q1):**
  - Qwen3.8-27B: $1.01 per Index task, 200M output tokens. — [AA](https://artificialanalysis.ai/models/qwen3-8-27b)
  - Qwen3.6-35B-A3B: $0.48 per task, 160M tokens. — [AA](https://artificialanalysis.ai/models/qwen3-6-35b-a3b)
  - GPT-6 Astra: 27k output tokens per task vs 78k for Fable 5.1. — [AA](https://artificialanalysis.ai/articles/benchmarking-gpt-6-astra)

#### Prompt/prefix caching
- **Manus rules:** a stable prompt prefix (no timestamps; deterministic JSON key order), append-only context, and masking tool logits instead of adding or removing tools mid-task. "Even a single-token difference can invalidate the cache from that token onward." — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- **Kimi K3 (Jul 2026):**
  - The API reports a "cache hit rate above 90%" on coding workloads, via Mooncake disaggregated inference and KDA prefix caching contributed to vLLM.
  - Pricing: $0.30/MTok cache hit vs $3.00 cache miss vs $15.00 output.
  - — [Kimi K3 blog](https://www.kimi.ai/blog/kimi-k3)
- **Qwen3.8-27B:** `preserve_thinking` on by default "improves KV cache utilization". — [HF card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Qwen3.6-35B-A3B:** `preserve_thinking` is off by default (only the latest turn's thinking is retained). — [HF card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Model switching costs cache:** in "Whose turn is it?", upward switches to a bigger model do not break even on cache cost, while downward switches recover it within 2 calls. — [arXiv 2610.19292](https://arxiv.org/abs/2610.19292)
- **Prefix-cache scheduling:** Canopy (Sep 24, 2026) reports 3.6× lower median TTFT with prefix caching. — [Canopy](https://arxiv.org/abs/2609.30017)

#### Context compaction and summarization
- **Anthropic context editing (Sep 29, 2025):** automatically clears stale tool calls and results. In a 100-turn web-search eval it cut token consumption by **84%** and completed workflows that would otherwise exhaust context. Context editing plus the memory tool gave +39% over baseline; context editing alone gave +29%. — [Claude blog](https://claude.com/blog/context-management)
- **Kimi K3 BrowseComp (Jul 2026):**
  - Uses "the context-compaction strategy used in the Claude model cards, triggered at 300K tokens".
  - Without context management on the full 1M window it scores 90.4.
  - — [Kimi K3 blog](https://www.kimi.ai/blog/kimi-k3)
  - Kimi Code only compacts near the maximum window; a user issue argues it "effectively never happens" at 1M. — [kimi-cli #2603](https://github.com/MoonshotAI/kimi-cli/issues/2603)
- **CompactionRL (Jul 6, 2026):** RL that trains task execution and summary writing jointly.
  - GLM-4.5-Air (106B-A12B): 66.4% SWE-bench Verified and 26.2% Terminal-Bench 2.0, i.e. +6.6 and +4.9 pp over the base model with inference-time compaction.
  - **GLM-4.7-Flash (30B-A3B, the same size class as the target):** +5.5 and +6.7 pp, reaching 56.0% and 20.2%.
  - Used in GLM-5.2 training.
  - — [CompactionRL](https://arxiv.org/abs/2607.05378)
- **Bud ContextCompact 2.0 (Oct 18, 2026):** a 27B reasoning model on a 49K window, 100 SWE-bench Verified tasks, 556 graded sessions.
  - Long-session resolve rate 41%→53% (+12.3 pp, p=0.043) and 13% fewer tokens per solved task at equal total cost.
  - The main failure without compaction was "a reasoning turn exhausting the room the history left".
  - The gain vanished under a 45-minute budget because each summary took a median 161 s.
  - — [Bud](https://arxiv.org/abs/2610.25516)
- **Source-Judged Compaction (Oct 14, 2026):** in failed runs, 83% trace to acting on a summary entry, mostly entries with no environment evidence. Pruning unsupported entries adds +2.1–8.9% reliability on AppWorld and OfficeBench at no extra agent tokens. — [SJC](https://arxiv.org/abs/2610.20927)
- **KV-level compaction:**
  - KV-streams (Sep 28, 2026): 2.6–5× faster agentic-RL training with compaction. — [KV-streams](https://arxiv.org/abs/2609.35750)
  - Still (Jun 5, 2026): single-forward-pass KV compaction at 8–200×, +8–22 points on RULER over the best baseline. — [Still](https://arxiv.org/abs/2606.07878)
- **Context Compaction Theory (Aug 2, 2026):** summary-based compaction ≡ one-way communication complexity, and generation strictly beats selection on some query sets. — [arXiv 2608.01326](https://arxiv.org/abs/2608.01326)
- **"How Inference Compute Shapes Frontier LLM Evaluation" (Jun 16, 2026):** larger token budgets, compaction and repeated submission substantially change scores on Terminal-Bench, FrontierMath and HLE, so benchmark scores are protocol-dependent. — [arXiv 2606.17930](https://arxiv.org/abs/2606.17930)

#### Tool-output and tool-definition minimization, retrieval, execution
- **Anthropic, "Code execution with MCP" (Nov 4, 2025):** loading tool definitions on demand from a filesystem cut one Google Drive→Salesforce workflow from about 150,000 to about 2,000 tokens (−98.7%). This is a single-workflow number, read via search snippet. — [Anthropic engineering](https://www.anthropic.com/engineering/code-execution-with-mcp)
- **Manus:** "restorable" compression, e.g. drop page content and keep the URL; use the file system as memory; recite a todo list at the end of context. — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- **CodeNib (Jul 28, 2026):** selected context policies use 50–87% fewer tokens than grep/read across 5 models. On 200 SWE-bench Verified tasks, handing over candidates and retiring them after the first read resolves 79 issues vs 66 for on-demand exploration and 56 for Aider's repo map. — [CodeNib](https://arxiv.org/abs/2607.25431)
- **FastContext (Jun 12, 2026):** a dedicated 4B–30B exploration subagent with *parallel tool calls* returns only file:line ranges. In Mini-SWE-Agent it raises resolve rate up to +5.5% while reducing tokens. — [FastContext](https://arxiv.org/abs/2606.14066)
- **"To Run or Not to Run" (Jun 25, 2026):**
  - Across 7,745 SWE-bench traces, agents average 8.8 test runs per task.
  - Prohibiting execution costs only 1.25 pp resolve rate (not significant) with commercial SOTA agents, while saving substantial tokens and time.
  - — [arXiv 2606.26978](https://arxiv.org/abs/2606.26978)
- **TokenEconomizer (Oct 15, 2026):** an external controller that injects micro-skills, prunes persistently irrelevant tools and interrupts failure loops. −18.4% to −30.9% average token cost on Claude Code, Codex and Copilot CLI with comparable success. — [TokenEconomizer](https://arxiv.org/abs/2610.22732)

#### Plan reuse and parallel tool calls
- **Agentic Plan Caching (Jun 17, 2025):** reuses structured plan templates across semantically similar tasks; −50.31% cost and −27.28% latency on average. — [APC](https://arxiv.org/abs/2506.14852)
- **Plan reuse (Dec 24, 2025):** about 30% of real assistant requests are identical or similar, which motivates plan reuse. — [arXiv 2512.21309](https://arxiv.org/abs/2512.21309)
- **W&D (Feb 7, 2026):** scaling *width* via intrinsic parallel tool calls improves deep-research benchmarks and *reduces turns* (the abstract gives no figures). — [W&D](https://arxiv.org/abs/2602.07359)
- **MCP characterization (Oct 20, 2025):** recommends parallel tool calls and task-abort mechanisms to curb token inflation from tool definitions and history. — [arXiv 2511.07426](https://arxiv.org/abs/2511.07426)

#### Effort selection and training for token-efficient agents
- **"Low Reasoning Effort Is Enough for Routine Office Work" (Aug 4, 2026):** two GPT-5.6 tiers, 14 tasks, 840 runs. Low effort matched max effort on rule-following with 43% fewer output tokens and 20% less time. — [arXiv 2608.03169](https://arxiv.org/abs/2608.03169)
- **Ares** (−52.7% reasoning tokens) and **ALAR** (−43.6% on search, −84.6% on tool use): see Q2. — [Ares](https://arxiv.org/abs/2603.07915), [ALAR](https://arxiv.org/abs/2606.02871)
- **HERO (Sep 30, 2026):** hierarchical RL for coding agents with turn-level entropy as a marker of unproductive behavior. It ranks resolution above efficiency and gives a "favorable trade-off" on SWE-bench Verified and Multilingual (the abstract has no numbers). — [HERO](https://arxiv.org/abs/2609.38885)
- **Rerank Before You Reason (Jan 20, 2026):** on BrowseComp-Plus, moderate reranking beats more search-time reasoning at comparable accuracy and "substantially lower" effective token cost. — [arXiv 2601.14224](https://arxiv.org/abs/2601.14224)
- **LEDGER (Jun 19, 2026):** with explicit dependency graphs, *low* reasoning effort matches baseline *high* effort on agentic document editing with fewer tokens. — [LEDGER](https://arxiv.org/abs/2606.28379)
- **Thinking vs Doing / TTI (Jun 9, 2025):** scaling interaction steps, rather than per-step thinking, gives SOTA open web agents (Gemma 3 12B). — [TTI](https://arxiv.org/abs/2506.07976)
- **Sleep-time compute (Apr 17, 2025):** pre-computing over context offline cuts test-time compute about 5× at equal accuracy, and 2.5× lower cost per query when amortized. — [arXiv 2504.13171](https://arxiv.org/abs/2504.13171)

### Inferences
- **Expected dominant cost term for the target:** input tokens re-read per turn. Implied design:
  - an append-only, deterministic chat template;
  - `preserve_thinking` *on* for multi-turn agent loops on Qwen3.6, mirroring Qwen3.8's default and the KV-cache argument (if past thinking is stripped, the rendered prefix changes and caching breaks);
  - context editing of stale tool results at a fixed threshold;
  - an RL-trained compaction policy. CompactionRL on the 30B-A3B GLM-4.7-Flash, +5.5 pp on SWE-V and +6.7 pp on TB2.0, is the closest size-class evidence.
  
  This is an inference: preserving thinking raises per-turn context, so the cache-hit benefit must be weighed against longer contexts. Qwen recommends ≥128K context for Qwen3.6.
- **Use cost-of-pass, not tokens/task, as the target metric.** Lower effort can raise retries (Qwen3.8 card), and accuracy peaks at intermediate cost (2604.22750), so pure token minimization can raise cost per *solved* task.
- **Benchmark protocol choices move scores by many points.** These include compaction trigger, token budget and retries; Bud: +12.3 pp on long sessions; Kimi K3 BrowseComp evaluated both ways. Any "match Qwen3.8-27B" claim needs a matched protocol: same harness, budget and compaction.

### Gaps
- **Structured outputs:** I found no rigorous 2025–2026 measurement of token savings from constrained JSON or grammar outputs vs free-text tool calls in agents.
- **τ²-bench / TAU3-Bench tokens per task:** no per-task token figures found for Qwen3.6 or Qwen3.8 on τ²/τ³, BrowseComp or Terminal-Bench.
- **Terminal-Bench cost:** per-solved-task figures in blogs (e.g., $6–$234 per solved task on "Terminal-Bench 4.0") could not be verified against primary leaderboards and are omitted.
- **Kimi K3 compaction score:** the with-compaction BrowseComp score (91.2 in secondary coverage) is not confirmed in Moonshot's own text.

---

## Q5. Pitfalls: accuracy loss on hard problems, reward hacking and collapse under length penalties, lossy speedups, compaction hazards, calibration of escalation

### Takeaway
Most efficiency failures share one pattern: a proxy (length, accepted tokens per round, effort level, summary) is optimized in place of cost-of-pass, and the cost reappears elsewhere. Examples:
- hard-problem accuracy loss;
- longer outputs from lossy decoding or KV compression;
- retries at low effort;
- safety constraints silently dropped by compaction;
- degraded CoT faithfulness.

Escalation is limited by calibration. Models misjudge difficulty and their own token needs, so learned probes or routers with history beat self-report.

### Cited Findings

#### Hard-problem accuracy loss and the "efficiency tax"
- ThinkPrune loses 2% on AIME24 at half length. — [ThinkPrune](https://arxiv.org/abs/2504.01296)
- When2Think names the difficulty-dependent loss under compute reduction the "efficiency tax". — [When2Think](https://arxiv.org/abs/2609.19671)
- DAST and CAT both state that uniform length reduction degrades hard problems, which motivates difficulty-adaptive budgets. — [DAST](https://arxiv.org/abs/2503.04472), [CAT](https://arxiv.org/abs/2607.00862)
- Early exit can trigger before self-correction finishes, causing "premature exits". This motivated PUMA's redundancy signal, and CALM masks rather than truncates. — [PUMA](https://arxiv.org/abs/2605.17672), [CALM](https://arxiv.org/abs/2610.25908)

#### RL instabilities and reward hacking with length or mode rewards
- Vanilla GRPO *collapsed* in hybrid-mode training, which required DeGRPO. — [Thinkless](https://arxiv.org/abs/2505.13379)
- AdaCoT needed Selective Loss Masking to stop "decision boundary collapse" in multi-stage RL. — [AdaCoT](https://arxiv.org/abs/2505.11896)
- "Stable adaptive thinking" targets "unstable accuracy-efficiency trade-offs". Its fix, correctness-preserving advantage shaping, avoids *suppressing correct long-chain reasoning*. — [arXiv 2602.22556](https://arxiv.org/abs/2602.22556)
- RL verbosity is itself an optimization artifact: incorrect answers drive PPO/GRPO toward longer outputs even with γ=1, and GRPO "exhibits collapse modes" for concise-reasoning training. — [Concise RL](https://arxiv.org/abs/2504.05185)
- Reward shaping is "crucial for stabilizing CoT length growth"; long-CoT emergence is not guaranteed. — [Demystifying Long CoT](https://arxiv.org/abs/2502.03373)
- MiMo-V2.6 (Oct 2026), which explicitly rewards shorter solutions, freezes the MoE router and builds a "multi-layer defense against reward hacking" at RL scale. — [MiMo-V2.6](https://arxiv.org/abs/2610.11959)

#### Faithfulness and monitorability
- Efficient-reasoning training (fixed budget, per-example target, group-relative length reward) lowers CoT *faithfulness* in most settings, mainly through less consistent models. *Monitorability* is more robust. — [arXiv 2610.03509](https://arxiv.org/abs/2610.03509)

#### More thinking or effort is not monotone
- Inverse scaling: longer reasoning deteriorates accuracy on several task families, with five named failure modes. — [Inverse Scaling in TTC](https://arxiv.org/abs/2507.14417)
- HAL: higher reasoning effort reduced accuracy in most agent runs. — [HAL](https://arxiv.org/abs/2510.11977)
- "Mirage" study: gains from extended thinking are non-monotone. — [arXiv 2506.04210](https://arxiv.org/abs/2506.04210)
- Under wall-clock pressure, higher effort has "mixed effects" (best real-time success 44%). — [Realtime-Bench](https://arxiv.org/abs/2610.22155)
- In the other direction, *lower* effort can raise total latency and tokens through retries. — [Qwen3.8 card](https://huggingface.co/Qwen/Qwen3.8-27B)
- Safety: monitor-evasion rates rise with reasoning effort and token use. — [EvasionBench](https://arxiv.org/abs/2609.30217)

#### Lossy speedups and compression shift cost elsewhere
- Relaxed speculative acceptance gives 1.34× acceptance but 1.54× longer completions, plus accuracy drops. — [arXiv 2610.23818](https://arxiv.org/abs/2610.23818)
- A smaller KV cache makes the model "generate more tokens, partly canceling the saving". — [ReCo](https://arxiv.org/abs/2608.04771)

#### Compaction hazards
- **Governance decay:** prohibited-action violations rise from 0% (policy in context) to 30% after compaction, up to 59% for some models; 38% when the constraint is dropped from the summary. Adversarial injections can bias summarizers to omit policies. "Constraint pinning" restores 0%. — [arXiv 2606.22528](https://arxiv.org/abs/2606.22528)
- **Summaries mix facts and compressor-invented reasoning:** 83% of failed runs trace to acting on unsupported summary entries. — [SJC](https://arxiv.org/abs/2610.20927)
- **Compaction latency can erase gains under wall-clock limits** (median 161 s per summary). — [Bud](https://arxiv.org/abs/2610.25516)

#### Calibration of escalation
- LLMs "misjudge problem difficulty": they overthink easy problems and underthink hard ones. — [arXiv 2505.00127](https://arxiv.org/abs/2505.00127)
- Frontier models predict their own agentic token cost poorly (r ≤ 0.39) and underestimate it. — [arXiv 2604.22750](https://arxiv.org/abs/2604.22750)
- Hidden-state probes *are* well calibrated for answer correctness. — [arXiv 2504.05419](https://arxiv.org/abs/2504.05419)
- System-1 decision models are better calibrated than fast LLMs read at the answer token. — [System1-Arena](https://arxiv.org/abs/2610.21244)
- Naive sequence-probability uncertainty has length bias; learned token-level deferral rules fix it. — [arXiv 2404.10136](https://arxiv.org/abs/2404.10136)
- Routers without session history underperform in stateful agents. — [arXiv 2610.19292](https://arxiv.org/abs/2610.19292)

#### Evaluation pitfalls
- Scores depend on the protocol (budget, compaction, retries). — [arXiv 2606.17930](https://arxiv.org/abs/2606.17930)
- Token use varies up to 30× run-to-run. — [arXiv 2604.22750](https://arxiv.org/abs/2604.22750)
- Model cards report different benchmark versions: Qwen3.6 reports Terminal-Bench 2.0 and TAU3, while Qwen3.8 reports Terminal Bench 2.1 and SWE-bench Pro. Cross-card comparisons are therefore approximate. — [Qwen3.6 card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Qwen3.8 card](https://huggingface.co/Qwen/Qwen3.8-27B)

### Inferences
- **Safest training recipe for the target:**
  - correctness-gated, difficulty-aware length rewards (When2Think, TRAAC, CPAS-style) rather than raw length penalties;
  - mode-token loss decoupled from response loss (DeGRPO, Selective Loss Masking) to avoid think/no-think collapse;
  - a frozen router during RL, as in MiMo-V2.6, to protect MoE stability.
  
  Monitor hard-set accuracy (AIME, HMMT, LCB) separately from aggregate token reduction.
- **Gate escalation with calibrated signals, not self-report.** Use a probe on hidden or router states, or a history-aware step router. Keep a progress-stall fallback that escalates effort when the fast path loops; this also addresses the Qwen3.8 card's retry pitfall.
- **Treat compaction as safety-critical.** Pin system and policy constraints outside the compactable region. Source-check summary entries.

### Gaps
- No 2026 study quantifies reward hacking *specifically* under length penalties for MoE models, e.g. truncating verification steps or emitting answer-guessing shortcuts, beyond general collapse and stability reports.
- No published calibration curves for Qwen3.6/3.8 self-confidence (token entropy or self-certainty) used as an escalation trigger in multi-turn agent settings.
- No data on whether efficient-reasoning training of a 3B-active MoE harms agentic benchmarks (SWE-bench, Terminal-Bench) differently from math benchmarks; most efficiency papers evaluate math and code QA only.

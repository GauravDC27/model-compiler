# Ensembles of Specialists, "System One" Fast Deciders, and Classical ML/DL Inside LLM Agents (state as of 21 Oct 2026)

Scope note: These notes cover research published 2022 to Oct 2026. Every dated finding gives the publication or revision date of its source. Figures marked "secondary" come from search summaries or third-party pages rather than from a reading of the primary paper. Vendor claims are labeled as such.

---

## Q1. LLM ensembles and model fusion (LLM-Blender, MoA/Self-MoA, routing ensembles, cascades, Smoothie, Avengers, Symbolic/Skill-MoE, verifier best-of-N, weight merging vs runtime ensembles): measured gains, costs, and when ensembles hurt

### Takeaway
The best-supported, cost-efficient "ensemble" in 2025–2026 is a **router or cascade over heterogeneous models**: embed or cluster the query, then pick one model (Avengers/Avengers-Pro, RouteLLM, FrugalGPT, AutoMix). It costs about 1x a single call and gives large cost reductions at matched quality. **Generate-many-then-synthesize** ensembles (MoA, LLM-Blender GenFuser) cost N+1 calls. Their gains are fragile: Self-MoA, a single strong model sampled repeatedly, beats mixing different models, and judge-based *selection* beats *synthesis*. Weight merging (soups, TIES, DARE, Model Stock) gives an ensemble-like effect at 1x inference cost, but only for models that share a base, and results are mixed on agentic skills such as tool use.

### Cited Findings
**Fusion / generate-then-aggregate ensembles**
- (Jun 2023, ACL 2023) **LLM-Blender** has two parts. **PairRanker** is a cross-attention encoder that compares candidate outputs pairwise to rank them. **GenFuser** fuses the top-K candidates into a new output. The authors introduce the MixInstruct benchmark and report that LLM-Blender "significantly outperforms individual LLMs and baseline methods". Cost: all N base models plus a ranker pass plus a fuser generation. — [arXiv 2306.02561](https://arxiv.org/abs/2306.02561)
- (Jun 2024) **Mixture-of-Agents (MoA)** uses layered proposers and an aggregator. Built only from open-source LLMs, it scored **65.1% on AlpacaEval 2.0 vs 57.5% for GPT-4o**. Cost: several proposer calls per layer across several layers. — [arXiv 2406.04692](https://arxiv.org/abs/2406.04692)
- (Feb 2025; TMLR 2026) **Self-MoA** aggregates multiple samples from one top model. It beats standard mixed-model MoA by **+6.6% on AlpacaEval 2.0** and by **+3.8% on average across MMLU, CRUX and MATH**, and it took #1 on AlpacaEval 2.0 when applied to a top model. The authors find that MoA is very sensitive to member *quality*, and that mixing models often lowers average quality, so diversity rarely pays. A sequential Self-MoA variant aggregates outputs over rounds and matches all-at-once aggregation. — [arXiv 2502.00674](https://arxiv.org/abs/2502.00674); [TMLR 2026 listing](https://mlanthology.org/tmlr/2026/li2026tmlr-rethinking/)
- (Mar 2026, rev. Jul 2026; *Applied Sciences* 16(10)) **"When Agents Disagree: The Selection Bottleneck"** ran 42 tasks in 7 categories (N=210):
  - A diverse team with **judge-based selection** had a win rate of **0.810** vs the single-model baseline. A homogeneous team had **0.512** (about chance).
  - Selection beat MoA-style **synthesis** by **ΔWR = +0.631**. Judges preferred synthesis over the baseline in **0 of 42 tasks**.
  - In an exploratory result, adding a weaker model improved quality while cutting cost (p<10⁻⁴).
  - Conclusion: "selector quality may be a more impactful design lever than generator diversity."
  - [arXiv 2603.20324](https://arxiv.org/abs/2603.20324)
- (Jul 2024) **Verifier or selector best-of-N ("Large Language Monkeys")**: coverage grows log-linearly with the number of samples. On SWE-bench Lite it went from **15.9% with 1 sample to 56% with 250 samples**. However, majority voting and reward-model selection **plateau beyond a few hundred samples**, so the verifier, not the generator, is the bottleneck. — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787)

**Routing-based ensembles (one model per query)**
- (May 2025; AAAI 2026 as "A Routing Recipe for Collective Intelligence") **Avengers** works in four steps: embed the query, cluster, score each model per cluster, then vote using self-consistency.
  - Ten ~7B open models together beat **GPT-4.1 on 9 of 15 datasets**. A catalyzex summary says 10/15, so the count conflicts.
  - Gains: **+18.21% on math** and **+7.46% on code** vs GPT-4.1.
  - From a pool of 22 models, **3 selected models** suffice to match GPT-4.1.
  - [arXiv 2505.19797](https://arxiv.org/abs/2505.19797v3); [AAAI version](https://ojs.aaai.org/index.php/AAAI/article/view/40790)
- (Aug 2025, rev. Oct 2025) **Avengers-Pro** routes over 8 frontier models (incl. GPT-5-medium, Gemini-2.5-pro, Claude-opus-4.1) on 6 benchmarks. It uses query embeddings, clusters, and a tunable performance–efficiency score.
  - About **+7% average accuracy over GPT-5-medium**.
  - Matches GPT-5-medium at **27% lower cost**; reaches ~90% of its accuracy at **63% lower cost**.
  - Claims to be Pareto-optimal vs every single model tested.
  - A third-party blog merges these into "+7% at 63% lower cost", which misstates the paper.
  - [arXiv 2508.12631](https://arxiv.org/abs/2508.12631); misstatement in [Nadir blog](https://getnadir.com/blog/llmrouterbench-acl-2026-most-routers-fail-baseline/)
- (Dec 2024) **Smoothie** is *label-free* routing inspired by weak supervision. It builds a latent-variable model over embeddings of the model outputs to score each model per sample.
  - It picks the optimal model on **9 of 14 tasks**.
  - It beats routing baselines by **up to 10 accuracy points**.
  - Caveat: it needs outputs from all candidate models at calibration time.
  - [arXiv 2412.04692](https://arxiv.org/abs/2412.04692)
- (Mar 2025, v4 May 2026, ICML 2026) **Symbolic-MoE**, renamed **Skill-MoE**, picks experts per instance in four steps: infer the skills a query needs, select experts by skill, have k experts answer, then pass the answers to a chosen aggregator.
  - **+8.15% absolute average** over the best baseline on MMLU-Pro, GPQA, AIME and MedMCQA.
  - Groups instances by expert in batches, so **16 expert models run on 1 GPU**, with runtime comparable to prior 4-GPU multi-agent baselines.
  - [arXiv 2503.05641](https://arxiv.org/abs/2503.05641)
- (Jun 2024; ICLR 2025) **RouteLLM** trains a strong/weak router on preference data. It cuts cost by **over 2x** without losing quality. Per the authors' blog: **85% cost reduction on MT-Bench at 95% of GPT-4 quality**, with about 14% of queries sent to the strong model. — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665); [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/)
- (Jan 2026; Findings of ACL 2026) **LLMRouterBench** covers 400K+ instances, 21 datasets, 33 models and 10 routing baselines.
  - Models are strongly complementary, but **many routers perform similarly**, and several recent and commercial routers **fail to reliably beat a simple baseline**.
  - A large gap to the Oracle remains, driven mainly by "model-recall failures".
  - The choice of embedding backbone has limited impact.
  - **Larger ensembles show diminishing returns** compared with careful model selection.
  - [arXiv 2601.07206](https://arxiv.org/abs/2601.07206); [ACL Findings 2026](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.1881/)
- (Mar 2026) A survey of dynamic routing and cascading warns that papers report gains in incompatible units and protocols, so cross-paper cost comparisons are unreliable. — [arXiv 2603.04445](https://arxiv.org/pdf/2603.04445)
- (May 2026) The "Unsolvability Ceiling" study argues that aggregate router metrics hide differences by domain and difficulty. — [arXiv 2605.07395](https://arxiv.org/pdf/2605.07395)

**Cascades (cheap first, escalate on low confidence)**
- (May 2023) **FrugalGPT** combines prompt adaptation, LLM approximation and an LLM cascade with a learned scorer. It matches GPT-4 with **up to 98% cost reduction**, or beats GPT-4 by **4% at equal cost**. — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- (Oct 2023; NeurIPS 2024) **AutoMix**: the small model answers, then the same model checks its own answer with few-shot self-verification, and a **POMDP router** decides whether to escalate. It needs black-box access only.
  - **>50% compute cost reduction** at comparable performance across 5 LMs and 5 datasets.
  - v1 reported up to 89% better "incremental benefit per cost" (llama2-13b → 70b).
  - [arXiv 2310.12963](https://arxiv.org/abs/2310.12963); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ecda225cb187b40ea8edc1f46b03ffda-Abstract.html)
- (Aug 2026, preprint) **RLCascadeRouter** is a cascade trained with reinforcement learning that needs no quality estimator. At ~67% performance it reports **20.8% lower cost than Avengers-Pro**, which it calls the strongest baseline disclosed in LLMRouterBench. Not replicated. — [arXiv 2608.15817](https://arxiv.org/pdf/2608.15817)

**Weight merging (1x inference cost) vs runtime ensembles (Nx cost)**
- (Mar 2022) **Model soups** average the weights of fine-tunes from the same pre-trained model. ViT-G soup reached **90.94% ImageNet top-1** with **no extra inference or memory cost** compared with a single model. — [arXiv 2203.05482](https://arxiv.org/abs/2203.05482)
- (Jun 2023; NeurIPS 2023) **TIES-Merging** has three steps: trim small deltas, elect a sign per parameter, then merge only agreeing parameters. It reduces interference between task vectors. — [arXiv 2306.01708](https://arxiv.org/abs/2306.01708)
- (Nov 2023; ICML 2024) **DARE** drops **90–99% of delta parameters** at random and rescales the rest before merging homologous SFT models, with little loss. It is often combined with Task Arithmetic or TIES for decoder LMs. — [arXiv 2311.03099](https://arxiv.org/abs/2311.03099)
- (Mar 2024; ECCV 2024) **Model Stock** approximates the center of the weight distribution from just **two fine-tuned models** by interpolating with the pre-trained weights according to the angles between models. — [arXiv 2403.19522](https://arxiv.org/abs/2403.19522); described in [arXiv 2511.21437](https://arxiv.org/pdf/2511.21437)

### Inferences
- **Default design rule:** use a router or cascade (1–1.3x calls) before any fusion ensemble (N+1 calls). Routing over a small, curated pool captures most of the complementarity. LLMRouterBench shows diminishing returns from bigger pools, and Avengers needed only 3 of 22 models.
- **If you must spend more test-time compute**, do two things. First, prefer Self-MoA or best-of-N from the *strongest* model over mixing weaker models. Second, invest in the **selector or verifier**: a judge, PairRanker-style cross-encoder, reward model, or execution-based check. Avoid free-form synthesis, which won 0/42 tasks in the selection-bottleneck study.
- **Merging vs ensembling:** merge when the specialists are LoRA or SFT variants of one base model and the skills are compatible. This yields one deployable model at 1x cost. Keep a runtime router when the specialists differ in architecture or tokenizer, or when one skill, such as tool calling, regresses after merging. Evaluate per skill after any merge.
- Ensembles hurt in four cases:
  1. Members are much weaker than the best member.
  2. The aggregator synthesizes rather than selects.
  3. Gains are computed on aggregate metrics that hide regressions on some slices.
  4. Router training data differs from the production query mix.

### Gaps
- No head-to-head study was found that compares weight merging with runtime ensembling at matched compute on agentic or tool-use tasks. A search summary said one 2025 merging study reports "severe regressions" from TIES and DARE on tool use and challenges the claim that they beat simple averaging. I could not confirm whether this is [arXiv 2511.21437](https://arxiv.org/pdf/2511.21437) or [arXiv 2511.13254](https://arxiv.org/pdf/2511.13254), so treat it as unverified.
- Per-benchmark numbers for LLM-Blender, Skill-MoE vs MoA, and Smoothie vs supervised routers were not extracted; only the abstracts were read.
- Live (Oct 2026) router leaderboards were not checked.

---

## Q2. Small language models (SLMs) for agents: NVIDIA's position paper and follow-ups, SLM specialists for tool calling and extraction, distillation, cost and latency

### Takeaway
NVIDIA's 2025 position paper is still the reference argument and has had no NVIDIA follow-up paper as of Oct 2026. Its claims:
- A 7B SLM is **10–30x cheaper** to serve than a 70–175B LLM.
- **40–70% of LLM calls** in three open agent frameworks could be replaced by SLMs.
- Fine-tuning SLMs on **10k–100k logged examples** is the recommended path.

In 2025–2026, tool-calling SLMs (≤8B) reached or beat GPT-4o-class models on BFCL-style benchmarks when fine-tuned or prompted the right way. Task-specific distillation can make sub-1B students beat 100x larger teachers on narrow tasks.

### Cited Findings
- (Jun 2025, v3 processed Sep 2026) **"Small Language Models are the Future of Agentic AI"** (Belcak et al., NVIDIA).
  - **SLM definition:** a model that fits on a consumer device and serves one user's agent requests with low latency. As of 2025 this means roughly **<10B parameters**.
  - **Cost:** serving a 7B SLM is **10–30x cheaper** than a 70–175B LLM in latency, energy and FLOPs.
  - **Replaceable share of LLM queries:** **MetaGPT ~60%, Open Operator ~40%, Cradle ~70%**.
  - **Fine-tuning:** parameter-efficient fine-tuning needs "a few GPU-hours".
  - **Six-step LLM-to-SLM conversion algorithm:**
    1. Securely log non-HCI agent calls.
    2. Curate and filter the data, removing sensitive data.
    3. Cluster tasks to find recurring request patterns.
    4. Select an SLM by capability, benchmarks, license and footprint.
    5. Fine-tune it (LoRA/QLoRA or distillation), with **10k–100k examples** as the rule of thumb.
    6. Retrain periodically.
  - **Architecture:** SLM-first, heterogeneous, with LLMs called selectively.
  - **Evidence cited in the paper:**
    - Phi-2 (2.7B) roughly matches 30B models and runs ~15x faster.
    - Nemotron-H (2–9B) matches dense 30B models at a fraction of the FLOPs.
    - Hymba-1.5B has 3.5x the throughput of comparable transformers.
    - DeepSeek-R1-Distill-Qwen-7B beats Claude-3.5-Sonnet-1022 and GPT-4o-0513 on reasoning.
    - **xLAM-2-8B is state of the art in tool calling, ahead of GPT-4o and Claude 3.5.**
    - Toolformer (6.7B) beats GPT-3 (175B) on API use.
  - [arXiv 2506.02153](https://arxiv.org/abs/2506.02153); [HTML v3](https://arxiv.org/html/2506.02153v3)
- (2026) No separate NVIDIA follow-up paper was found. 2026 coverage is mostly commentary; one blog notes that the hardest reasoning still favors frontier models. — [Agentic Review](https://theagenticreview.substack.com/p/why-small-language-models-will-eat); [arXiv 2506.02153](https://arxiv.org/abs/2506.02153)
- (Aug 2026, preprint; numbers secondary) **"Small Reasoning Models are Instruction Followers in Function Calling"** frames a tool call as a conversational instruction (IFFC) instead of native function calling (NFC) or prompt-based calling (PFC).
  - **Qwen3-4B (thinking):** **86.7% Live and 94.1% Non-Live** on BFCL, vs ~82% (PFC) and ~81% (NFC).
  - **Phi-4-mini:** ~30% (NFC) → **44.3%** (IFFC) on Live.
  - Gemma-3-4B under PFC collapses to near 0% on parallel-multiple calls; Gemma-3-12B reaches 79.1% under IFFC.
  - [arXiv 2608.22472](https://arxiv.org/pdf/2608.22472)
- (May 2026, preprint) **Switchcraft** is a model router built specifically for agentic tool calling. It is an example of routing tool-call steps to small models. — [arXiv 2605.07112](https://arxiv.org/pdf/2605.07112)
- (May 2023; Google) **Distilling step-by-step** trains a student on LLM rationales as extra supervision. A **770M T5 beat few-shot-prompted 540B PaLM** using only **80% of the training data** on a benchmark. — [Google Research blog](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)
- (Apr 2026, preprint) **TRACER** routes on traces: LLM outputs serve as labels for cheap surrogate classifiers, and calls are routed to the surrogate when it is reliable. On CLINC150 the surrogate **fully replaces the teacher at every quality target**. Code is open source. — [arXiv 2604.14531](https://arxiv.org/pdf/2604.14531)
- (Vendor, 2025) TensorZero claims that fine-tuning small models on curated large-model outputs gives **up to 30x lower inference cost and up to 4x lower latency**. These are vendor benchmarks. — [TensorZero blog](https://www.tensorzero.com/blog/distillation-programmatic-data-curation-smarter-llms-5-30x-cheaper-inference/)
- (Jul 2025) **GLiNER2** is a ~205M-class encoder for multi-task extraction (NER, classification, structured extraction) driven by a schema. Results:
  - CPU latency for text classification: **130 ms vs GPT-4o API 358 ms at 5 labels**, and **208 ms vs 463 ms at 50 labels** (~2.6x faster).
  - CrossNER zero-shot F1: **0.590 vs GPT-4o 0.599**.
  - Average zero-shot classification over 7 datasets: **0.72 vs GPT-4o 0.84** (secondary summary).
  - [arXiv 2507.18546](https://arxiv.org/pdf/2507.18546)

### Inferences
- In agent loops, **tool-argument filling, structured extraction, intent or route classification, and format conversion** are the best candidates for SLMs or encoders. They are high-volume, narrow-schema and verifiable (JSON schema or execution check), so they also fit cheap cascades.
- The NVIDIA 40–70% replaceable range is an **estimate from inspecting the frameworks, not a measured A/B test**. Use it as a prior and measure on your own logs, following the paper's steps S1–S3.
- **Tool calling with a 4B model is now viable** if the call format is chosen carefully: prompting format alone moved Phi-4-mini by +14 points. Do not judge an SLM on native function calling alone.
- For extraction, a GLiNER2-class encoder on CPU is roughly at GPT-4o parity on NER. It is weaker on zero-shot classification, so fine-tune it or escalate to an LLM when label sets are open-ended.

### Gaps
- No independent measurement of NVIDIA's "40–70% replaceable" estimate on production agent traces was found.
- No 2026 BFCL v4 leaderboard snapshot comparing the best ≤8B models with frontier models was retrieved.
- No measured end-to-end dollar cost per agent task (SLM-first vs LLM-only) from a named production deployment was found.

---

## Q3. "System One" fast single-pass deciders: encoders (ModernBERT, DeBERTa-v3, NeoBERT, Ettin), probes on LLM hidden states, early exit, embedding+kNN, GLiNER, cross-encoder rerankers, Qwen3-Embedding/Reranker, guard classifiers; accuracy vs prompted LLMs, latency, calibration

### Takeaway
For **closed label sets with training data**, fine-tuned encoders (110–400M) usually **match or beat zero-shot frontier LLMs** at about **1000x lower cost** and **millisecond latency**. Examples: +11.8 points on ATIS, +20.6 points on legal clauses, and near-parity on CLINC150. LLMs win on **out-of-scope detection, noisy input, and label schemas that change per deployment**. Probes on an LLM's hidden states often recover labels more accurately than the model's own *verbal* answer, which makes them a one-forward-pass decider. Guard classifiers are mature, but benchmark rankings disagree.

### Cited Findings
**Encoders vs prompted LLMs**
- (Jun 2026, preprint) **"When Do LLMs Replace Fine-Tuned NLU?"**
  - **Where the encoder wins:** ATIS intent accuracy was **RoBERTa (fine-tuned) 95.9% vs Claude Haiku (zero-shot) 84.1%** (p<0.001). On CLINC150 the two were tied: **89.1% vs 88.5%** (n.s.).
  - **Cost:** RoBERTa is about **three orders of magnitude cheaper and faster**.
  - **Out-of-scope recall:** **LLM 85.6% vs RoBERTa 58.1%**.
  - **Noise:** under 0 dB noise passed through ASR, ~92.5% (LLM per context) vs 80.0%.
  - **Changing schemas:** a classifier trained on one app scored **0%** on another app's intents, while a schema-prompted LLM served both at ~94% with no retraining.
  - [arXiv 2608.20371](https://arxiv.org/abs/2608.20371)
- (2025, *AI & Law*) Legal clause classification: **fine-tuned DeBERTa ~87.8% vs zero-shot GPT-4 67.2%** (secondary figures). — [Springer](https://link.springer.com/article/10.1007/s10506-025-09484-4)
- (Jun 2024) **"Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification"**: fine-tuned BERT-class models beat zero-shot GPT-3.5, GPT-4 and Claude Opus across several classification tasks. — [arXiv 2406.08660](https://arxiv.org/abs/2406.08660)
- (Nov 2024) Political science text: fine-tuned BERT **53.9% vs GPT prompting 48.8%**. The paper cites ~100 samples/s for BERT inference and says prompting becomes more expensive than fine-tuning after roughly 150–200 test samples (secondary summary; the break-even depends on assumptions). — [arXiv 2411.05050](https://arxiv.org/html/2411.05050v1)
- (Jul 2025, rev. Mar 2026; ICLR 2026) **Ettin** trains paired encoders and decoders (17M–1B) on the same recipe.
  - Encoders excel at classification and retrieval; decoders at generation.
  - **A 400M encoder beats a 1B decoder on MNLI**.
  - Converting one architecture to the other by continued training underperforms training with the matching objective.
  - Ettin encoders beat ModernBERT.
  - [arXiv 2507.11412](https://arxiv.org/abs/2507.11412)
- (Dec 2024) **ModernBERT** is a long-context encoder (8192 tokens, trained on 2T tokens) built for efficient GPU inference. A Hugging Face guide calls it state of the art on classification and **2–4x faster** than earlier encoders. — [arXiv 2412.13663](https://arxiv.org/abs/2412.13663); [Philipp Schmid guide](https://www.philschmid.de/fine-tune-modern-bert-in-2025)
- (2025) Zero-shot classifiers built on ModernBERT underperformed DeBERTa-based ones on several benchmarks, so speed does not guarantee accuracy. — [HF blog](https://huggingface.co/blog/Ihor/refreshing-zero-shot-classification)
- (Feb 2025) **NeoBERT** is a ~250M encoder with 4096 context, positioned as a plug-in replacement for BERT-large or RoBERTa-large. The authors report it beats ModernBERT and others on MTEB under identical fine-tuning. — [arXiv 2502.19587](https://arxiv.org/abs/2502.19587)
- (Sep 2022) **SetFit** does contrastive fine-tuning of sentence embeddings plus a logistic head. With **8 labeled examples per class** it is competitive with RoBERTa-large fine-tuned on ~3k examples. This makes it the fastest route from zero labels to a deployable encoder. — [arXiv 2209.11055](https://arxiv.org/abs/2209.11055)

**Probes and heads on LLM hidden states (one forward pass, no generated tokens)**
- (Aug 2024) **Penalised logistic regression on embeddings from a small LLM** equals, and usually beats, a large prompted LLM in the "tens-of-shot" regime, and it is explainable. — [arXiv 2408.03414](https://arxiv.org/pdf/2408.03414)
- (Oct 2026; AACL-IJCNLP 2026 Findings) **"Structured but Silent"**: linear probes recover 8 classes of tool-use and capability requirements "with high accuracy across all models" (4 open-weight families, 1,600 training queries). The same models are "significantly less reliable" when asked to classify in natural language; a secondary summary says the probes beat 4-shot prompting on a 920-query test set. Directly relevant to agent routing. — [arXiv 2610.08018](https://arxiv.org/abs/2610.08018)
- (Feb 2026) **"LLMs Know More About Numbers than They Can Say"**: a linear probe on hidden states ranks numeral pairs with **>90% accuracy**, while one-shot verbal answers are much worse. — [arXiv 2602.07812](https://arxiv.org/html/2602.07812v1)
- (2026) Probes on hidden states are also used to detect hallucinations from prompt embeddings (PEP) and machine-generated text, typically on the last token with PCA. — [arXiv 2608.08024](https://arxiv.org/html/2608.08024); [arXiv 2608.24780](https://arxiv.org/pdf/2608.24780)
- (Jul 2022) **Early exit (CALM)** adapts compute per token with a confidence-based exit and reports **up to ~3x decoding speedup** with provable quality guarantees. — [arXiv 2207.07061](https://arxiv.org/abs/2207.07061)

**Extraction, retrieval and ranking**
- (Nov 2023) **GLiNER** is a bidirectional-encoder NER model for arbitrary entity types. It **outperforms ChatGPT and fine-tuned LLMs on zero-shot NER** benchmarks. — [arXiv 2311.08526](https://arxiv.org/abs/2311.08526)
- (Jul 2025) GLiNER2 CPU latency and accuracy vs GPT-4o are given in Q2. — [arXiv 2507.18546](https://arxiv.org/pdf/2507.18546)
- (Jun 2025) **Qwen3-Embedding** (0.6B/4B/8B) and **Qwen3-Reranker** (0.6B/4B/8B). Qwen3-Embedding-8B scored **70.58 on MTEB multilingual (#1 on 5 Jun 2025)** and **75.22 on MTEB eng v2**. — [arXiv 2506.05176](https://arxiv.org/abs/2506.05176)
- (Jan 2026) **Qwen3-VL-Embedding/Reranker** extends this to multimodal retrieval; the 8B reranker is best on most multimodal tasks. — [arXiv 2601.04720](https://arxiv.org/pdf/2601.04720)

**Guard and safety classifiers (benchmarks conflict)**
- (May 2026) GLiGuard's table of average F1 for prompt harmfulness: **Qwen3Guard-Gen-8B 88.7 vs Llama Guard 4 (12B) 82.5**. GLiGuard itself is a schema-conditioned encoder guard that "outperforms several larger models". — [arXiv 2605.07982](https://arxiv.org/pdf/2605.07982)
- (May 2026) An ICLR-2026-workshop benchmark of open guard models gives **Qwen Guard 4B F1 0.756, Llama Guard 12B 0.468, GPT-OSS-Safeguard 20B 0.380**. This ranking conflicts with GLiGuard's. — [arXiv 2605.28830](https://arxiv.org/html/2605.28830v1)
- (Jun 2026) Per-prompt latency vs harmful-F1: **Llama Guard 4 12B 7.44 ms / F1 0.691** and **gpt-oss-safeguard-120B 24.66 ms / 0.807** (secondary; the hardware and batching basis is unclear). — [arXiv 2606.27210](https://arxiv.org/pdf/2606.27210)
- (May 2026) **GLiNER Guard** is a unified family of encoders for production LLM safety and PII detection, i.e. guarding with an encoder instead of a generative guard. — [arXiv 2605.05277](https://arxiv.org/pdf/2605.05277)
- (Nov 2024) **Llama Guard 3-1B-INT4** is a compressed on-device guard. — [arXiv 2411.17713](https://arxiv.org/pdf/2411.17713)

**Calibration**
- (Jun 2017) Modern neural networks are miscalibrated; **temperature scaling** with one parameter on a validation set is a simple, effective fix. — [arXiv 1706.04599](https://arxiv.org/abs/1706.04599)
- (Jun 2023) LLMs that state their confidence verbally tend to be **overconfident**, and sampling-consistency methods help. — [arXiv 2306.13063](https://arxiv.org/abs/2306.13063)
- (Sep 2026) An ensemble of embeddings plus RF/XGB/LR used as a "System One" was reported as "well-calibrated and safely **under- rather than over-confident**", which suits confidence gating. — [arXiv 2609.17977](https://arxiv.org/abs/2609.17977)

### Inferences
- **Decision rule:** if the label set is closed and stable and you have at least a few hundred labels (or 8/class with SetFit, or labels distilled from an LLM as in TRACER), use a fine-tuned encoder or a probe. Add a calibrated reject option, and send rejects and out-of-scope items to an LLM.
- **Probes vs encoders:** if an LLM is already loaded and its prefill is computed anyway (for example, it sits in the agent loop), a linear head on its hidden states costs about 0 extra tokens. It can decide the route, the tool, whether to stop, or a risk flag *before generation*. A separate encoder is cheaper when no LLM forward pass would otherwise happen.
- **Retrieval stack:** embedding model (Qwen3-Embedding-0.6B/4B class) → cross-encoder reranker (Qwen3-Reranker-0.6B/4B) → LLM only for synthesis. This is the standard cheap pipeline; both stages are single-pass.
- **Guards:** the benchmarks disagree a lot, partly because of taxonomy mismatch. Calibrate the guard's threshold on *your* policy taxonomy. Encoder guards (GLiGuard, GLiNER Guard, Prompt-Guard-class) handle high-QPS pre-filtering; generative or reasoning guards (Qwen3Guard, gpt-oss-safeguard) handle escalations.

### Gaps
- No single 2026 benchmark measures accuracy, latency and calibration (ECE) together for ModernBERT/NeoBERT/Ettin vs frontier LLMs across many tasks.
- Probe accuracies in "Structured but Silent" were not visible in the abstract.
- The live MTEB rank of Qwen3-Embedding in Oct 2026 was not verified. Third-party pages give inconsistent numbers.
- No sourced numbers were found for Llama Prompt Guard 2 latency or F1.

---

## Q4. Where classical ML wins: GBDTs and tabular foundation models (TabPFN v2/2.5/3.x, TabICL), time-series foundation models (TimesFM, Chronos-2, Moirai), anomaly detection, logistic regression, Bayesian methods; evidence vs LLMs; how agents call them

### Takeaway
On tabular prediction, **LLMs that read serialized rows lose to GBDTs** except with very little data. As of Sep–Oct 2026, **tabular foundation models** (TabPFN-3/3.5, RealTabPFN-2.5, TabICLv2) sit **above tuned and ensembled LightGBM/CatBoost** on TabArena, with Elo leads of about 150–500 depending on the snapshot. For forecasting, **pretrained time-series foundation models** (Chronos-2, TimesFM-2.5) lead GIFT-Eval. Reusing LLMs as forecasters adds cost without accuracy. The agent should **call these models as tools** rather than reason about the numbers in tokens.

### Cited Findings
**Tabular**
- (Nov 2025) **TabPFN-2.5** (Prior Labs):
  - Leads TabArena, which has datasets up to 100k training rows, in a single forward pass. It beats tuned tree models and **matches AutoGluon 1.4 tuned for 4 hours**.
  - Win rate vs **default XGBoost**: **100%** on datasets of ≤10k rows and ≤500 features; **87%** (classification) and **85%** (regression) on datasets up to 100k rows and 2k features.
  - These are the authors' claims.
  - [arXiv 2511.08667](https://arxiv.org/abs/2511.08667); [Prior Labs report](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report)
- (Sep 2026) **TabPFN-3.5** technical report (Hutter, Hollmann, Grinsztajn + 44 others).
  - Claims SOTA on TabArena, beating TabPFN-3 and "all existing baselines".
  - Adds handling of non-i.i.d. data (temporal or grouped splits), text, string and image columns, high-cardinality categoricals and wide tables.
  - Variants: **-Fast** (up to 3x faster than TabPFN-3), **-Plus** (multimodal: text and dates), and **-Thinking** (scales inference-time compute; 12x faster than TabPFN-3-Thinking).
  - Also reports stronger time-series forecasting and relational-data results.
  - [arXiv 2609.17895](https://arxiv.org/abs/2609.17895)
- (2025–2026) **TabArena Elo snapshots** (compare ranks only within a table):
  - TabPFN-3.5 report (51 datasets): **TabPFN-3.5-Thinking 1910 vs LightGBM (tuned+ensembled) 1404 and CatBoost (T+E) 1394**.
  - arXiv 2605.13986: **TabPFN-3-Thinking 1800, RealTabPFN-2.5 (T+E) 1602, TabICLv2 (default) 1599, LightGBM (T+E) 1438, CatBoost (T+E) 1420**.
  - Original TabArena v0.1 (NeurIPS 2025): tuned+ensembled TabM 1541, LightGBM 1527, CatBoost 1482. Trees were much closer to the top in mid-2025.
  - [TabPFN-3.5](https://arxiv.org/pdf/2609.17895); [arXiv 2605.13986](https://arxiv.org/html/2605.13986v2); [TabArena paper](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf); [EXAONE-Tabular card](https://huggingface.co/LG-AI-Research/EXAONE-Tabular)
- (Sep 2026) Other 2026 tabular foundation models also claim top TabArena ranks: **Mitra-v2**, whose table has TabFM+ at 1817; **TabFM**; and **Xiaomi-TabLDM**, #2 on regression Elo in its own report. These are self-reported. — [Mitra-v2](https://arxiv.org/pdf/2609.04540); [TabFM](https://arxiv.org/pdf/2609.37959); [Xiaomi-TabLDM](https://arxiv.org/html/2609.03880v1)
- (May 2026, preprint) **LLM serialization vs GBDT**: in few-shot tabular classification, **XGBoost scores 78.5 AP vs 74.8 AP for TabLLM-style serialization**. The boosting-inspired BoostLLM reaches 79.7 using 5x the compute. Plain LLM-on-text-rows loses to GBDT. — [arXiv 2605.06117](https://arxiv.org/html/2605.06117)
- (Feb 2025) **LLM-Boost / PFN-Boost**: LLM priors help XGBoost only at **small data sizes** and only with **meaningful column headers**. PFN-Boost beats LLM-Boost on larger datasets. — [arXiv 2502.02672](https://arxiv.org/html/2502.02672)

**Time series**
- (Oct 2025) **Chronos-2** (Amazon) is a universal forecaster supporting multivariate inputs and covariates.
  - On GIFT-Eval (97 tasks, 55 datasets), probabilistic **win rate 79.8% and skill score 46.6%**, vs **TimesFM-2.5 at 70.0% / 42.4%** and TiRex at 70.4% / 41.7%.
  - Pretraining avoided GIFT-Eval test splits but partly overlaps their training splits.
  - [arXiv 2510.15821](https://arxiv.org/abs/2510.15821); [Amazon Science](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
- (Nov 2025) Cisco's time-series report, on a GIFT-Eval subset with no leakage: **Chronos-2 MASE 0.6738 vs TimesFM-2.5 0.6828**. — [arXiv 2511.19841](https://arxiv.org/pdf/2511.19841)
- (Jul 2026) As of a 9 Apr 2026 freeze, **Chronos-2 ranked #1** among pretrained models on GIFT-Eval. It is flagged for partial overlap with GIFT-Eval training data. Fine-tuning these models improves them further. — [arXiv 2607.23146](https://arxiv.org/pdf/2607.23146)
- (Nov 2025) **Moirai 2.0**, "When Less Is More", is Salesforce's smaller, more efficient successor. — [arXiv 2511.11698](https://arxiv.org/pdf/2511.11698)
- (2025) **TimesFM 2.5** (Google) is a ~200M-parameter decoder-only foundation model with up to 16k context and an optional quantile head, per the repo README. — [google-research/timesfm](https://github.com/google-research/timesfm)
- (Jun 2024; NeurIPS 2024 spotlight) **"Are Language Models Actually Useful for Time Series Forecasting?"**
  - In three LLM-based forecasters, **removing the LLM or replacing it with one attention layer did not hurt performance and usually improved it**.
  - Pretrained LLMs did no better than models trained from scratch, did not capture sequential dependencies, and did not help few-shot.
  - They add "significant computational cost".
  - [arXiv 2406.16964](https://arxiv.org/abs/2406.16964)

**Production evidence that classical ML or embeddings beat generic LLMs on structured decisions**
- (2025) **Stripe** built a payments foundation model: a self-supervised transformer trained on tens of billions of transactions that produces one embedding per charge. A separate classifier reads sequences of these embeddings, which raised the **card-testing detection rate for large users from 59% to 97%** "overnight". Earlier traditional ML had cut card testing by 80%. False-positive rates are not disclosed. — [Stripe blog](https://stripe.com/en-gr/blog/the-ml-flywheel-how-we-continually-improve-our-models-to-reduce-card-testing); [ZenML summary of Stripe talk](https://www.zenml.io/llmops-database/building-economic-infrastructure-for-ai-with-foundation-models-and-agentic-commerce); [Gautam Kedia via DeepakNess](https://deepakness.com/raw/stripe-transfer-based-model/)

### Inferences
- **Tabular decisions** (risk score, eligibility, churn, priority, pricing) in an agent should be a **tool call**. Use TabPFN-class models for <100k rows when zero tuning is needed; they hit SOTA in one forward pass, in seconds on a GPU. Use LightGBM, CatBoost or XGBoost for large data, tight CPU latency, or mature MLOps. Have the LLM produce the feature JSON, call the model, and read back a calibrated probability.
- **Forecasting:** call Chronos-2 or TimesFM-2.5 zero-shot (fine-tuned where data allows). Never ask the LLM to extrapolate numbers in tokens.
- The pattern "foundation embedding + small classifier" (Stripe) generalizes. A large self-supervised encoder makes representations once, and many cheap heads decide. This is the classical-ML counterpart of probes on LLM hidden states (Q3).
- **Logistic regression** remains the right final layer when you need calibrated, auditable probabilities over embeddings. The small-LLM-embedding + LR result in Q3 is evidence.

### Gaps
- Kaggle write-ups from 2025–2026 that compare LLM agents with GBDT pipelines head-to-head were not retrieved.
- No sourced 2026 numbers were found for **anomaly detection**: Isolation Forest/ECOD vs LLM-based detectors, or time-series anomaly foundation models.
- No sourced numbers were found for **Bayesian methods** in agent decision loops.
- Live TabArena and GIFT-Eval leaderboards were not checked directly for Oct 2026. All Elo numbers are self-reported snapshots that differ by model set.
- No documented MCP or tool interfaces for TabPFN, Chronos or TimesFM were verified.

---

## Q5. Orchestration: classifier → LLM escalation, calibration across heterogeneous models (temperature scaling, conformal), production mixes of classic ML and LLMs, measured savings

### Takeaway
The pattern with the best evidence is a **confidence-gated cascade**. A calibrated System One handles the confident majority in milliseconds; low-confidence, out-of-scope or schema-novel inputs go to an LLM; an optional verifier or judge can escalate further. Measured results:
- Hybrids beat *both* pure systems on quality while costing 2–10x less than LLM-only.
- Learned routers save 27–85% at matched quality on benchmarks.
- Conformal methods give marginal (not per-subgroup) error guarantees for the deferral threshold.

### Cited Findings
- (Sep 2026) **Confidence-gated hybrid for emotion recognition**. System One is a stacked RF+XGBoost+LR over sentence embeddings with dialogue context, running in **<10 ms**. GPT-4o-mini handles escalations of the least-confident cases.
  - IEMOCAP weighted F1: **ensemble 0.595, LLM-only 0.460–0.536, hybrid 0.620**.
  - The hybrid also beats both pure systems on MELD (0.643) and CMU-MOSI (0.824), even though the LLM beats the ensemble on those two.
  - **Cost: hybrid $10–85 per million utterances vs LLM-only $99–170.**
  - The share of escalated queries is not stated in the abstract.
  - [arXiv 2609.17977](https://arxiv.org/abs/2609.17977)
- (Jun 2026) For intent detection, use a fine-tuned NLU model on the stable, high-QPS, latency-critical path, and an LLM for out-of-scope inputs (85.6% vs 58.1% recall), noisy ASR, and schemas that vary per tenant. This is a decision framework from the paper. — [arXiv 2608.20371](https://arxiv.org/abs/2608.20371)
- (Jul 2026, preprint) **Conformal Cascade** accepts a tier's answer when its calibrated conformal prediction set collapses to a single answer, and otherwise defers. The nonconformity score is answer frequency under repeated decoding.
  - Guarantee: coverage ≥ 1−Kα per tier, tightening to 1−α under a selection-preservation condition.
  - A critical review argues the headline accuracy depends on operating points (kappa>1, fallbacks) that the guarantee does not cover.
  - [arXiv 2607.25018](https://arxiv.org/abs/2607.25018); [Pith review](https://pith.science/paper/2607.25018)
- (2025; described secondhand) Jung et al. use **fixed-sequence testing** on a calibration set to choose escalation thresholds for an LLM-judge cascade, with error ≤ α with probability ≥ 1−δ. They report up to **~82.5% less use of the strongest model**. — described in [arXiv 2509.22984](https://arxiv.org/pdf/2509.22984)
- (Oct 2025) Edge–cloud cascades via **conformal alignment**: heuristic confidence gives no reliability guarantee, whereas calibrated thresholds give a marginal error guarantee. — [arXiv 2510.17543](https://arxiv.org/pdf/2510.17543)
- (Jan 2025) **Prune 'n Predict** defers to the LLM based on conformal set size. As the set-size cutoff rises, accuracy approaches the LLM's while deferrals fall, giving a tunable accuracy–deferral curve. — [arXiv 2501.00555](https://arxiv.org/pdf/2501.00555)
- (Sep 2026) **Caveat:** conformal guarantees that are marginal over questions can fail on targeted subgroups. This was shown for multi-agent debate under peer pressure. — [arXiv 2609.04445](https://arxiv.org/html/2609.04445v1)
- Measured router and cascade savings, with dates and sources, are in Q1:
  - FrugalGPT (May 2023): up to **98%**.
  - AutoMix (NeurIPS 2024): **>50%**.
  - RouteLLM (2024): **85% on MT-Bench at 95% of GPT-4 quality**.
  - Avengers-Pro (Aug 2025): **27% cheaper at parity** with GPT-5-medium.
  - [FrugalGPT](https://arxiv.org/abs/2305.05176); [AutoMix](https://arxiv.org/abs/2310.12963); [RouteLLM blog](https://lmsys.org/blog/2024-07-01-routellm/); [Avengers-Pro](https://arxiv.org/abs/2508.12631)
- (May 2025) **Shopify product classification** has a classic-ML lineage. It started in 2018 with TF-IDF + logistic regression, moved to a multimodal neural network in 2020, and now uses a **Qwen2-VL 7B VLM with FP8** (down from Llama 3.2 11B Vision). Details:
  - Two-stage prediction: category, then attributes, with KV-cache reuse. In-flight batching runs on NVIDIA Dynamo.
  - **>30M predictions per day** against 10k+ categories and 1k+ attributes.
  - **Hierarchical precision and recall about 2x** the earlier NN approach; **85% merchant acceptance**.
  - Training labels come from **several LLMs annotating independently, with judge models resolving disagreements and humans handling edge cases**.
  - [Shopify Engineering](https://shopify.engineering/evolution-product-classification)
- (2025) Toloka, on Shopify's catalogue work: VLM performance was not consistent enough at deep taxonomy levels, so those categories went to human annotators. This is an escalation tier above the model. — [Toloka blog](https://toloka.ai/blog/building-shopify-s-product-catalogue-at-ai-speed/)
- (2025) Stripe pairs foundation-model embeddings with a downstream classifier for fraud; see Q4. — [Stripe blog](https://stripe.com/en-gr/blog/the-ml-flywheel-how-we-continually-improve-our-models-to-reduce-card-testing)

**Speech and vision preprocessing (cheap specialist front-ends)**
- (Oct 2025; rev. 2026) **Open ASR Leaderboard paper** results:
  - **Parakeet-TDT-0.6B-v2: 6.05% average WER at RTFx ≈ 3,386**; v3: 6.32% at RTFx ≈ 3,333 across 25 languages.
  - **Canary-Qwen-2.5B: 5.63% WER at RTFx ≈ 418**; Canary-1B-Flash: 6.35% at RTFx ≈ 1,046.
  - Whisper-large-v3: ~7.4%, with the widest language coverage.
  - Conformer encoders with LLM decoders give the best WER; CTC and TDT decoders give much higher throughput.
  - [arXiv 2510.06961](https://arxiv.org/html/2510.06961v2)
- (May 2026 snapshot, secondary) **IBM Granite Speech 4.1 2B** tops the leaderboard at **5.33% WER**, followed by Cohere Transcribe at 5.42%. — [Northflank blog](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- (Feb 2025) **olmOCR** is a fine-tuned 7B VLM for converting PDFs to text at scale, at about **$190 per million pages** by the authors' estimate. It shows that a specialist document model replaces general frontier-VLM calls for bulk OCR. — [arXiv 2502.18443](https://arxiv.org/abs/2502.18443)

### Inferences
**Sub-task → cheapest model class that usually meets the accuracy bar** (synthesis of the findings above; check each choice on your own data):

| Agent sub-task | Default cheap model (System One) | Escalate to | Evidence |
|---|---|---|---|
| Intent / route / tool-choice classification (closed set) | Fine-tuned encoder (ModernBERT/Ettin/DeBERTa-v3, 100–400M) or linear probe on the agent LLM's hidden states; SetFit if labels are scarce | LLM when confidence is low, input is out-of-scope, or the schema is new | ATIS 95.9 vs 84.1; OOS 58 vs 86 ([2608.20371](https://arxiv.org/abs/2608.20371)); probes > verbal ([2610.08018](https://arxiv.org/abs/2610.08018)) |
| Model routing (which LLM) | Embedding + cluster or kNN scorer (Avengers-style) or a trained router (RouteLLM) | — | [2505.19797](https://arxiv.org/abs/2505.19797v3), [2508.12631](https://arxiv.org/abs/2508.12631), [2601.07206](https://arxiv.org/abs/2601.07206) |
| NER / span extraction | GLiNER / GLiNER2 on CPU (~130–200 ms) | SLM (≤8B, fine-tuned) for nested JSON; LLM for open schemas | [2507.18546](https://arxiv.org/pdf/2507.18546) |
| Structured tool-argument filling | Fine-tuned 4–8B SLM (xLAM-2-8B / Qwen3-4B class), checked with a JSON-schema validator | Frontier LLM when validation fails | [2506.02153](https://arxiv.org/abs/2506.02153), [2608.22472](https://arxiv.org/pdf/2608.22472) |
| Retrieval / ranking | Embedding model (Qwen3-Embedding 0.6–8B) → cross-encoder reranker (Qwen3-Reranker) | LLM for listwise reranking of the top few only | [2506.05176](https://arxiv.org/abs/2506.05176) |
| Tabular decisions | TabPFN-2.5/3.x (≤100k rows) or tuned LightGBM/CatBoost/XGBoost, with LR or temperature-scaled outputs | Human or LLM explanation layer only | TabArena snapshots ([2609.17895](https://arxiv.org/pdf/2609.17895)); serialization loses ([2605.06117](https://arxiv.org/html/2605.06117)) |
| Forecasting | Chronos-2 / TimesFM-2.5 zero-shot, then fine-tuned | — (do not use an LLM) | [2510.15821](https://arxiv.org/abs/2510.15821), [2406.16964](https://arxiv.org/abs/2406.16964) |
| Safety / PII guard | Encoder guard (GLiGuard / GLiNER Guard class) | Qwen3Guard / gpt-oss-safeguard for borderline cases | [2605.07982](https://arxiv.org/pdf/2605.07982), [2605.05277](https://arxiv.org/pdf/2605.05277) |
| Verification of outputs | Execution or schema checks; a reward model or PairRanker-style cross-encoder selecting among N samples | LLM judge | [2407.21787](https://arxiv.org/abs/2407.21787), [2603.20324](https://arxiv.org/abs/2603.20324) |
| Speech front-end | Parakeet-TDT (very high RTFx) or Canary/Granite Speech (best WER) | — | [2510.06961](https://arxiv.org/html/2510.06961v2) |
| Document / vision preprocessing | Specialist OCR VLM (olmOCR-class 7B) | Frontier VLM for hard pages | [2502.18443](https://arxiv.org/abs/2502.18443) |

**"Best of all worlds" ensemble design**, assembled from the evidence above:
1. **Front-end specialists:** ASR, OCR, embeddings and an encoder guard run first and turn raw input into compact features or text.
2. **System One decision layer:** encoder classifiers, probes on the agent LLM's hidden states, GBDT or TabPFN for tabular inputs, and time-series foundation models for forecasts. Each emits a *calibrated* probability, using temperature scaling or Platt scaling per model on a held-out set.
3. **Gate:** a conformal or fixed-sequence-tested threshold per decision type sets accept vs defer, with a target α. Confident cases are accepted; the rest are deferred. Expect marginal, not per-subgroup, guarantees, so monitor important slices separately.
4. **System Two:** a router (Avengers-style cluster scoring) sends deferred items to the cheapest capable SLM or LLM. Only for high-stakes or verifiable outputs does the system add best-of-N plus a *selector or verifier*, not a synthesizer.
5. **Flywheel:** log LLM outputs on deferred items, distill them back into System One (TRACER or Distilling step-by-step style), and retrain periodically. This is NVIDIA's step S6, and it is how the share of escalated traffic falls over time.

- **Calibration across heterogeneous models:** do not compare raw softmax, verbalized confidence and GBDT scores directly. Calibrate each model on its own held-out data first, then gate on calibrated risk or on conformal set size. LLM verbal confidence is overconfident; use sampling consistency or answer frequency instead (Conformal Cascade's score).

### Gaps
- No primary engineering posts from **Uber, Airbnb, Netflix or DoorDash** with measured cost savings from a classifier → LLM escalation were retrieved in this pass. The only well-sourced production cases found are Stripe and Shopify.
- The fraction of traffic escalated in the confidence-gated hybrid, and its end-to-end latency, were not reported in the abstract.
- No published head-to-head compares temperature scaling with conformal gating for heterogeneous agent components on one benchmark.
- The current (Oct 2026) Open ASR Leaderboard leader was not confirmed; the latest dated snapshot found is May 2026.

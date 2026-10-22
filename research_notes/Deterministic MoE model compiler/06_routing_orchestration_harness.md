# Task-to-Model Routing, Multi-Model Orchestration, and Deterministic Execution Harnesses (state as of Oct 2026)

Scope: (a) how to route a task deterministically and efficiently to the most suitable specialist model; (b) how to build the harness around it (orchestration, caching, record-replay, versioning, fallbacks) so a multi-model LLM system behaves like a deterministic compiler toolchain. Research date: 2026-10-21. Every bullet gives the date of the source in [brackets]. "Secondary" marks claims I saw only in aggregator or blog summaries, not in the primary source.

---

## Q1. LLM routing research and systems: accuracy, cost savings, and whether routing decisions are deterministic or stable

### Takeaway
Learned routers reliably cut cost (roughly 30–85% at near-parity quality on curated benchmarks). Independent 2026 benchmarks, though, find that most sophisticated routers barely beat the best single model or a static per-task-type lookup table, and that their gains often come from which tier they pick rather than from per-query targeting. Routing decisions are **not** inherently stable. Classifier and embedding routers flip under keyword injection (up to a ~98% flip rate) and under paraphrase (16.7% pick-flip in one production router, cut to 2.4% with a margin tie-breaker). Adversarial suffixes can also redirect them. A deterministic harness therefore has to treat the router as a versioned, pure function with explicit tie-breaking. It should also check the router against a static-table baseline.

### Cited Findings

**Foundational routers (2023–2025)**
- [Jul 2024] RouteLLM (LMSYS/Berkeley/Anyscale) trains routers on Chatbot Arena preference data plus augmentation to choose between a strong model (GPT-4 Turbo) and a weak one (Mixtral 8x7B). The blog reports cost reductions of over 85% on MT Bench, 45% on MMLU and 35% on GSM8K compared with GPT-4 only, while keeping 95% of GPT-4's performance. — [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/)
- [Jun 2024, rev. Feb 2025] The RouteLLM paper abstract claims costs fall "by over 2 times in certain cases" without hurting quality. It also reports that the routers keep their performance "even when the strong and weak models are changed at test time" (transfer). — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665)
- [2024] The RouteLLM repo recommends the matrix-factorization (MF) router, a "matrix factorization model trained on the preference data". Other router types are similarity-weighted ranking, a BERT classifier and a causal-LLM classifier. — [GitHub lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM)
- [Secondary, 2024] MF reportedly reaches 95% of GPT-4 performance with 26% GPT-4 calls (~48% cheaper than the random baseline). Not verified against the primary source. — [Medium summary](https://medium.com/accredian/llm-routing-optimizing-pathways-in-language-processing-c52c2adf7c4e)
- [May 2023] FrugalGPT (LLM cascade) can "match the performance of the best individual LLM (e.g. GPT-4) with up to 98% cost reduction or improve the accuracy over GPT-4 by 4% with the same cost". Its scorer needs labeled data. — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- [Apr 2024, ICLR 2024] Hybrid LLM (Microsoft) makes "up to 40% fewer calls to the large model, with no drop in response quality." — [arXiv 2404.14618](https://arxiv.org/abs/2404.14618)
- [Oct 2023, NeurIPS 2024] AutoMix uses few-shot self-verification and a POMDP router and reduces "computational cost by over 50% for comparable performance" across 5 LMs and 5 datasets. Its authors say it beats FrugalGPT and HybridLLM on performance per cost. — [arXiv 2310.12963](https://arxiv.org/abs/2310.12963); [PDF](https://arxiv.org/pdf/2310.12963)
- [Mar 2024, ICML 2024] RouterBench (Martian) contains 405k+ pre-computed inference outcomes (8 datasets, 11 models), so routers can be evaluated offline without running inference. It found that some routers fail to generalize to complex tasks and newer models, while simple routing does well in some areas. — [arXiv 2403.12031](https://arxiv.org/html/2403.12031v2)
- [Mar 2025, EMNLP Findings 2025] RouterEval has 200M+ performance records from 8,500+ LLMs on 12 evaluations. It reports "model-level scaling up": a strong router improves as the candidate pool grows and can beat the best single model. Existing methods "still have significant room for improvement." — [arXiv 2503.10657](https://arxiv.org/abs/2503.10657v2); [ACL Anthology](https://preview.aclanthology.org/dashboard/2025.findings-emnlp.208)
- [ICLR 2025] GraphRouter builds a heterogeneous task–query–LLM graph and predicts edge effect and cost. It handles new LLMs without retraining and reports a ≥12.3% improvement over prior routers and ≥9.5% on new-LLM settings. — [MLAnthology](https://mlanthology.org/iclr/2025/feng2025iclr-graphrouter); [ChatPaper summary](https://chatpaper.com/paper/112454)
- [ICLR 2025] EmbedLLM learns compact vector embeddings of models (an encoder-decoder) that support correctness forecasting and routing, and it "outperforms prior methods in model routing" in both accuracy and latency. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bf5e4b85d203481d6e37bd32d9600162-Abstract-Conference.html); [arXiv 2410.02223](https://arxiv.org/pdf/2410.02223)
- [NAACL 2025] MixLLM, a contextual-bandit router with lightweight quality and cost predictors plus a meta-decision maker, reaches 97.25% of GPT-4 quality at 24.18% of the cost under a time constraint. — [ACL Anthology 2025.naacl-long.545](https://aclanthology.org/2025.naacl-long.545); [arXiv 2502.18482](https://arxiv.org/abs/2502.18482v1)
- [Jun 2025, NeurIPS 2025] Router-R1 makes the router itself an LLM, trained with RL, that interleaves "think" and "route" actions over multiple rounds. It uses a rule-based reward (format + outcome + cost) and conditions on model descriptors such as price and latency, which lets it generalize to unseen models. It was evaluated on 7 QA benchmarks. — [arXiv 2506.09033](https://arxiv.org/html/2506.09033v3); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/119214)
- [May 2025] The Avengers, a training-free recipe (embed → cluster → score → vote) over 10 open ~7B models, beats GPT-4.1 on 9 of 15 datasets (+18.21% on math, +7.46% on code). It can match GPT-4.1 with 3 models chosen from a pool of 22. — [arXiv 2505.19797](https://arxiv.org/html/2505.19797v2)
- [Aug 2025] Avengers-Pro uses an 8-model pool (including GPT-5 and Claude) with a performance–efficiency α. It beats GPT-5-medium by +7% average accuracy. Secondary sources report ~27% cost reduction at comparable accuracy. — [arXiv 2508.12631](https://arxiv.org/html/2508.12631v2); [DAIR summary](https://academy.dair.ai/papers/avengers-pro-beyond-gpt-5)

**Preference / policy routers (deployable)**
- [Jun 2025] Arch-Router (Katanemo) is a 1.5B generative router fine-tuned from Qwen2.5-1.5B on ~43k synthetic samples. It maps (natural-language route policies + conversation) to a route name, and a separate mapping links each route to a model. New routes need no retraining because the policies are in-context. — [arXiv 2506.16655](https://arxiv.org/html/2506.16655v1)
- [Jun 2025] Arch-Router accuracy (turn / span / conversation / overall, %): Arch-Router 96.05 / 94.98 / 88.48 / **93.17**; Claude-Sonnet-3.7 96.24 / 94.70 / 87.45 / 92.79; GPT-4o 93.96 / 90.74 / 84.52 / 89.74; Gemini-2.0-flash 85.63 overall; base Qwen2.5-1.5B 20.69 overall. — [arXiv 2506.16655](https://arxiv.org/html/2506.16655v1)
- [Jun 2025] Arch-Router latency is **51 ± 12 ms** self-hosted on an L40S, against 510 ms (Gemini-2.0-flash-lite), 836 ms (GPT-4o) and 1450 ms (Claude-Sonnet-3.7) measured via OpenRouter. The paper's "28x faster than closest competitor" claim actually matches Sonnet-3.7; the gap to the closest competitor is ~10x. — [arXiv 2506.16655](https://arxiv.org/html/2506.16655v1)
- [Jun 2025] The Arch-Router paper's stated limitations are that overlapping policy descriptions "can lead to arbitrary routing", that errors cluster on the first turn, and that no decoding or determinism settings are reported. The model card shows output as JSON (`{"route": "..."}`, or `"other"`) and gives no temperature setting. — [arXiv 2506.16655](https://arxiv.org/html/2506.16655v1); [HF model card](https://huggingface.co/katanemo/Arch-Router-1.5B)
- [Docs, 2025–2026] semantic-router (Aurelio) is embedding-based with no LLM call. A route is selected only if its similarity score exceeds a per-route `score_threshold` (default example 0.5); otherwise it returns another route or none. `fit()` tunes thresholds from labeled data. The class was renamed `RouteLayer` → `SemanticRouter` in v0.1.x. — [Aurelio docs: thresholds](https://docs.aurelio.ai/semantic-router/user-guide/features/threshold-optimization.md); [overview](https://docs.aurelio.ai/semantic-router/user-guide/concepts/overview.md)
- [Jan 5, 2026] vLLM Semantic Router v0.1 "Iris" introduced a signal→decision plugin chain with six signal types (domain, keyword, embedding, factual, feedback, preference) and a configurable decision engine. — [vLLM blog](https://vllm.ai/blog/2026-01-05-vllm-sr-iris); [arXiv 2603.04444](https://arxiv.org/html/2603.04444v1)
- [Jun 5, 2026] vLLM-SR v0.3 "Themis" adds stateful, session-aware routing. Hard locks block model switches during active tool loops and provider-state continuations. `session_aware` selection weighs quality gap, switch margin, stay bias and prefix-cache locality (e.g. `prefix_cache_weight: 0.20`, `handoff_penalty_weight: 1.0`). The routing DSL gained `TEST`, `TIER` and `SIGNAL_GROUP` plus conflict detection. Policy versions have shadow/activate/revert states, replay records store projection traces, and the config loader warns on unknown YAML fields "to catch silent routing drift". It cites a RouterArena snapshot of weighted score 75.4 (#1), accuracy 76.0, $0.11 per 1K queries, robustness 73.1. — [vLLM blog v0.3](https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release)

**Commercial routers**
- [Aug 2025, secondary] ChatGPT's GPT-5 is a system: a fast model, a reasoning model, and a "real-time router" that picks per request based on conversation type, complexity, tool needs and explicit intent ("think hard"). It is trained on signals such as user model-switches, preference rates and measured correctness. One analysis says the API does not expose this router. — [FoneArena](https://www.fonearena.com/blog/461231/openai-gpt-5-features.html/amp); [Where's Your Ed At](https://www.wheresyoured.at/how-does-gpt-5-work/)
- [2026, conflicting, secondary] OpenRouter's `openrouter/auto` was historically powered by Not Diamond. Mid-2026 sources say it was deprecated and replaced by `openrouter/auto-beta`, which ranks models using OpenRouter's own task-type data and a rolling 7-day community spending signal, from ~Aug 10, 2026. Not Diamond's own blog still describes itself as powering auto-routing. — [Developers Digest](https://www.developersdigest.tech/blog/openrouter-review-setup-2026); [AIReiter](https://aireiter.com/blog/openrouter-auto-router-guide); [Not Diamond blog](https://www.notdiamond.ai/blog/the-top-10-ai-gateways-for-the-multi-model-future-2026)
- [Jan 2026] LLMRouterBench measured OpenRouter's auto router at **−24.7%** performance gain relative to the Best Single model. — [arXiv 2601.07206](https://arxiv.org/html/2601.07206)

**2025–2026 independent benchmarks: do routers beat simple baselines?**
- [Oct 2025, ICLR 2026] RouterArena is an open leaderboard with domain and difficulty stratification and many metrics. In the initial preview, MIRT-BERT scored 67.3 and Azure-Router 66.4. GPT-5's router had higher accuracy but did not rank first on accuracy–cost because its pool is OpenAI-only. — [arXiv 2510.00202](https://arxiv.org/html/2510.00202v3); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10011126)
- [Jan 2026] LLMRouterBench: 400K+ instances, 21 datasets, 33 models, ~$2.8k API spend plus ~1K A800 GPU-hours. Findings:
  - Leading routers (EmbedLLM, GraphRouter, MODEL-SAT, Avengers) perform similarly.
  - Several recent routers fail to reliably beat Best Single.
  - Top cost-aware routers gain up to 4% average accuracy or cut cost by up to 31.7% at matched accuracy.
  - Binary routers (HybridLLM, FrugalGPT) struggle to save cost without losing accuracy.
  - Swapping the embedding backbone (gte-Qwen2-7B vs MiniLM vs nli-bert) made **no significant difference**.
  - A large oracle gap remains. On the 11.9% of queries where ≤3 models are correct, Avengers reaches 24.6% and EmbedLLM 23.2%.

  — [arXiv 2601.07206](https://arxiv.org/html/2601.07206)
- [Jul 2026] Kumar & Saminathan evaluated four open-source routers (incl. vLLM Semantic Router and Aurelio) on RouterBench, BFCL v4, tau2-bench and WebArena, using 290 frozen tasks and a locked 2,610-outcome matrix.
  - **Three routers emitted constant or near-constant tier assignments.**
  - An "Always-Mid" policy matched Aurelio exactly on 3 benchmarks and within 0.003 on the 4th.
  - vLLM-SR showed no task-specific advantage over a share-matched, content-blind allocation.
  - Conclusion: "Fixed-tier baselines and selected-tier distributions are therefore necessary controls in router evaluation."

  — [arXiv 2608.14641](https://arxiv.org/abs/2608.14641)
- [Aug 2026] "Most of the LLM Routing Gap Is Task Type" (Lee) used 14 models, 294 questions, 7 task types and 3 languages.
  - Re-running the identical setup changed the score of **5.37% of 4,116 model–question cells**.
  - Under a strict two-run rule, only 29 questions are improvable by routing. A **static one-model-per-task-type table** captures 21 of them, and adding a language split brings it to 23.
  - The static table gets 262/294 correct at $3.33 per run, versus 245/294 at $7.69 for the best single model.
  - Caveat: no held-out set.

  — [arXiv 2608.23023](https://arxiv.org/abs/2608.23023)
- [Aug 2026] LLMRouter (UIUC, You lab) is an open-source framework with 16+ routers, formulated as context encoder + model encoder + scoring function + decision rule + learning signal. On its xRouteBench, learned routers beat the strongest fixed model by 14.6% relative, and lightweight routers become competitive under tight cost budgets. — [arXiv 2608.06867](https://arxiv.org/abs/2608.06867)
- [Mar 2026, TMLR 2026] The survey "Dynamic Model Routing and Cascading for Efficient LLM Inference" (Moslem & Kelleher) organizes the field by difficulty, preference, clustering, uncertainty, RL, multimodal and cascade paradigms. — [arXiv 2603.04445](https://arxiv.org/abs/2603.04445)

**Stability, paraphrase sensitivity, and adversarial robustness of routing**
- [Mar 2025] Kassem, Schölkopf & Jin found that preference-trained routers make "category-driven decisions". A BERT router sends **all** coding and math queries to the strongest LLM even when unnecessary, and sends jailbreaks to *weaker* models. They introduce the DSC benchmark. The PDF reports a "Flipping Rate": injecting math or coding keywords redirected queries to the strong model at an **average 98% flip rate**. — [arXiv 2504.07113](https://arxiv.org/abs/2504.07113); [PDF](https://arxiv.org/pdf/2504.07113)
- [Jan 2025] "Rerouting LLM Routers" (Shafran et al.) shows that short, query-independent "confounder gadgets" appended to any query force routing to the strong model, in white-box and black-box settings, against open and commercial routers. — [arXiv 2501.01818](https://arxiv.org/abs/2501.01818)
- [2026] Follow-up attacks and defenses:
  - "Route to Rome": adversarial suffixes optimized on a surrogate router. — [arXiv 2604.15022](https://arxiv.org/html/2604.15022v1)
  - RerouteGuard: an embedding-based detector with adaptive thresholds. — [alphaXiv 2601.21380](https://www.alphaxiv.org/abs/2601.21380)
  - "Forced Deferral": adversarial images lower weak-model confidence in multimodal cascades. — [arXiv 2606.15308](https://arxiv.org/html/2606.15308)
- [May 2026] OrcaRouter defines **pick-flip rate** as the fraction of paraphrased prompt pairs whose selected arm differs from the original's, and attributes flips to narrow top-2 score margins. A tie-breaker picks the arm with the higher mean training reward when the top two arms are within ε = 0.02. This cut pick-flip from **16.7% to 2.4%** at a cost of ~0.86 points of score. Its RouterArena robustness score was 22.62, which the authors call the main weakness. — [arXiv 2605.30736](https://arxiv.org/pdf/2605.30736)

### Inferences
- A router's decision is only as deterministic as three things: its feature extractor, its scoring arithmetic and its tie-breaking. Embedding and classifier routers are deterministic for a fixed input on a fixed kernel stack, but **unstable under paraphrase** because many decisions sit near a margin. A compiler-like design should:
  1. canonicalize the prompt (whitespace, Unicode NFC, template-filled fields) before hashing;
  2. route on structured, declared features where possible (task type, language, tool-use presence), as in vLLM-SR's signal → decision DSL;
  3. use a margin/hysteresis rule (OrcaRouter ε-tie-break, vLLM-SR stay bias and switch margin);
  4. emit the routing decision plus policy version into the trace/lockfile, so replay never re-routes.
- Generative routers (Arch-Router, Router-R1) add LLM-decoding nondeterminism on top. Run them greedily on batch-invariant kernels (see Q3), or memoize decisions by canonical-prompt hash.
- The strongest 2026 evidence (2608.23023, 2608.14641, 2601.07206) suggests starting with a **static task-type → model table** (fully deterministic and auditable). Add a learned per-query router only where held-out, multi-run evaluation shows a gain larger than run-to-run noise (~5% cell flips).
- Router evaluation needs fixed-tier and share-matched baselines, multi-seed runs, a paraphrase flip-rate metric and an adversarial-suffix test. These belong in the harness's CI.

### Gaps
- No standardized cross-router paraphrase flip-rate benchmark was found. RouterArena reports a "robustness" metric, but I did not verify its exact definition.
- No published latency or accuracy numbers were found for Martian or Not Diamond beyond vendor claims. OpenRouter auto-router status (Not Diamond vs in-house) conflicts across sources.
- The exact RouteLLM MF numbers (26% GPT-4 calls) were seen only in a secondary summary.
- Router-R1's benchmark numbers and EmbedLLM's routing accuracy tables were not extracted (abstracts only).

---

## Q2. Hierarchical routing: task-level routing to specialist models vs token-level MoE, mixture-of-agents, merging vs routing, and when routing beats a generalist

### Takeaway
System-level "mixtures of models" come in four forms: routing (one model per query), ensembling and mixture-of-agents (many models per query), per-instance expert recruitment (Symbolic-MoE), and parameter-level MoE built from specialists (BTX). Routing beats a single generalist when the pool is complementary. The evidence is that no single model dominates every domain, oracle gaps are large and small-model ensembles beat GPT-4.1 on math and code. But the realized gain over the strongest model is often small (≤4–7%), and mixing weaker models in MoA can *hurt*: Self-MoA, which samples one strong model repeatedly, beats mixed MoA. Token-level MoE (inside a model) and task-level routing (across models) are complementary layers, not substitutes.

### Cited Findings
- [Jan 2026] LLMRouterBench: "no single model achieves universal dominance" across 21 datasets and 33 models, so models are complementary. Adding models yields diminishing returns, and a well-chosen top-k subset beats a random same-size subset. — [arXiv 2601.07206](https://arxiv.org/html/2601.07206)
- [Mar 2025, EMNLP Findings 2025] RouterEval's "model-level scaling up": with a capable router, performance grows with candidate-pool size and can exceed the best single model. — [arXiv 2503.10657](https://arxiv.org/abs/2503.10657v2)
- [May 2025] Avengers: 10 open ~7B models collectively beat GPT-4.1 on 9 of 15 datasets (math +18.21%, code +7.46%). This is evidence that a pool of small specialists can beat a frontier generalist on specific domains. — [arXiv 2505.19797](https://arxiv.org/html/2505.19797v2)
- [Mar 2025, ICML 2026] Symbolic-MoE (Skill-MoE) infers per-instance skills (e.g. "algebra") and recruits experts per query, then an aggregator combines their answers. It gains +8.15% absolute average over the best multi-agent baseline on MMLU-Pro, GPQA, AIME and MedMCQA. **Batching queries by assigned expert lets 16 experts run on 1 GPU** with runtime comparable to baselines on 4 GPUs. Top-3 beats top-5, because adding weaker experts hurts. — [arXiv 2503.05641](https://arxiv.org/abs/2503.05641); [ICML 2026](https://icml.cc/virtual/2026/poster/64575); [PMLR](https://proceedings.mlr.press/v306/chen26fy.html)
- [Jun 2024] Mixture-of-Agents (Wang et al.) layers proposer and aggregator LLMs and reached a 65.8% AlpacaEval 2.0 win rate versus 57.5% for GPT-4o. — [arXiv 2406.04692](https://arxiv.org/html/2406.04692v1)
- [Feb 2025, TMLR 2026] "Rethinking Mixture-of-Agents": **Self-MoA**, which aggregates multiple samples of one top model, beats mixed MoA by 6.6% on AlpacaEval 2.0 and by 3.8% on average across MMLU, CRUX and MATH. Mixing different LLMs lowers average proposer quality. — [arXiv 2502.00674](https://arxiv.org/abs/2502.00674); [TMLR 2026](https://mlanthology.org/tmlr/2026/li2026tmlr-rethinking/)
- [Mar 2024] Branch-Train-MiX (BTX) branches a seed model, trains domain experts in parallel, then merges their FFNs as MoE experts (averaging other parameters), followed by MoE fine-tuning to learn **token-level** routing. Branch-Train-Merge (BTM) keeps independent LLMs, which "preclud[es] crucial post-training". BTX requires identical architectures. — [arXiv 2403.07816](https://arxiv.org/abs/2403.07816v1); [Mediator paper discussing limitation](https://arxiv.org/pdf/2502.04411)
- [Secondary, 2025–2026] Microsoft Research (Lotfi et al.) compares ensembling, merging and routing among parameter-efficient experts. A summarized follow-up reports a hierarchy of "routing > ensembling > merging" in task-agnostic settings. This applies to adapters, not full models. — [Microsoft Research](https://www.microsoft.com/en-us/research/?p=1163960)
- [May 2024] SambaNova Composition of Experts (Samba-CoE) uses 150 independently fine-tuned experts (~1T total parameters) plus a router model that "dynamically assigns each input prompt to the most relevant expert". It differs from MoE in that experts are "trained/fine-tuned independently". In the paper all experts and the router derive from Llama2-7B. SambaStudio allows up to 60 experts per CoE. — [arXiv 2405.07518](https://arxiv.org/pdf/2405.07518); [SambaNova docs](https://docs.sambanova.ai/sambastudio/latest/create-coe.html)
- [Jun 2025, NeurIPS 2025] Router-R1 is a hierarchical, multi-round router: an LLM decides at each step whether to reason internally or call a model, and aggregates the responses. — [arXiv 2506.09033](https://arxiv.org/html/2506.09033v3)
- [May 2025 / May 2026] Harvey (legal) routes across OpenAI, Anthropic and Google: "There's no longer a single 'best' model". Its multi-model system "will break down the request into sub-tasks, select a model to use, then synthesize the outputs". — [Harvey blog](https://www.harvey.ai/blog/expanding-harveys-model-offerings); [Harvey help center (updated May 4, 2026)](https://help.harvey.ai/articles/harvey-ai-overview)
- [Dec 2024] Anthropic's "Building effective agents" describes routing as a *workflow* pattern ("predefined code paths") that fits "complex tasks with distinct categories … where classification can be handled accurately", for example easy questions to Haiku and hard ones to Sonnet. Orchestrator-workers covers subtasks that are "not pre-defined". — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

### Inferences
- For a "deterministic MoE model compiler", the hierarchy is:
  - **L0, request → task type** (static table or rule DSL; deterministic);
  - **L1, task type → specialist model** (lockfile-pinned);
  - **L2, optional per-instance expert recruitment or ensembling** for hard buckets only;
  - **L3, token-level MoE inside each model** (learned gate; deterministic only with batch-invariant kernels and fixed expert-parallel layout).
- Prefer Self-MoA-style multiple samples from the best specialist over mixing weak models. Restrict ensembles to a top-k selected by held-out data (LLMRouterBench and Symbolic-MoE both show weaker members hurt).
- Symbolic-MoE's "batch by assigned expert" trick is directly relevant to deterministic scheduling. It fixes which model is resident, and it can make batch composition a deterministic function of the routing plan.
- I found no matched-compute comparison of system-level routing (CoE) against a single token-level MoE (e.g. BTX). This is a key open question.

### Gaps
- No head-to-head BTX-MoE vs router-over-independent-specialists result at matched compute was found.
- The Lotfi et al. ranking was seen only via a secondary summary.
- No quantitative Samba-CoE benchmark from a primary source was extracted.

---

## Q3. Deterministic harness and pipeline design: record-replay, content-addressed caching, golden tests, version pinning, durable agent loops, framework features, DSPy compilation, LLMOps tooling

### Takeaway
Bitwise determinism requires control of the serving stack. Temperature 0 is not enough, because batch-size-dependent reduction order (not "GPU concurrency") is the dominant cause of nondeterminism. Batch-invariant kernels now ship in vLLM (`VLLM_BATCH_INVARIANT=1`) and SGLang (`--enable-deterministic-inference`) at roughly 25–60% throughput cost. LLM-42 offers a selective, cheaper alternative. Hosted APIs (OpenAI `seed`, Anthropic) explicitly do not guarantee determinism, so they must be treated as recorded nondeterministic activities. The 2026 harness pattern has four parts:
- durable execution (Temporal, DBOS, LangGraph checkpointers) where the orchestrator is deterministic and LLM and tool calls are recorded activities;
- cassette and cut-point replay for tests;
- exact-match caches keyed on full request plus model build;
- golden-set or ground-truth regression gates in CI.

### Cited Findings

**Inference-level determinism**
- [Sep 10, 2025] Thinking Machines (Horace He et al.): the forward pass has essentially no atomic adds and is run-to-run deterministic. Nondeterminism comes from **lack of batch invariance**, because server load changes the batch size. Only RMSNorm, matmul and attention need batch-invariant kernels.
  - Experiment: Qwen3-235B, T=0, 1,000 runs of the same prompt produced **80 unique completions**, diverging at token 103. With batch-invariant kernels all 1,000 were identical.
  - Cost (Qwen3-8B, 1,000 sequences): default 26 s, naive deterministic 55 s, improved attention 42 s.
  - Bitwise-identical sampler and trainer give zero KL and "true on-policy RL".

  — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference)
- [Sep 22, 2025] SGLang deterministic mode (`--enable-deterministic-inference`, sglang ≥ 0.5.3) supports FlashInfer, FA3 and Triton backends, chunked prefill, CUDA graphs, and radix cache (FA3 only). It also supports temperature > 0 via a per-request seeded Gumbel sampler.
  - Overhead: **average 34.35%** slowdown (most cases 25–45%) versus TML's 61.5%.
  - Limitation (Sep 2025): **dense models only, not MoE**; deterministic at TP1/TP2 only.
  - With extra flags, two Qwen3-8B RL runs gave identical curves.

  — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [2026 docs] vLLM batch invariance (`VLLM_BATCH_INVARIANT=1`) makes output "independent of the batch size or the order of requests in a batch". Hardware: compute capability ≥ 9.0 in v0.14–v0.20 docs, ≥ 8.0 in v0.26/stable docs. Newer docs list **DeepSeek-V3/R1/V3.1 (MoE) as tested**. — [vLLM docs (stable)](https://docs.vllm.ai/en/stable/features/batch_invariance/); [v0.20.2](https://docs.vllm.ai/en/v0.20.2/features/batch_invariance/)
- [Secondary, 2026] vLLM's async scheduling (on by default) can still break batch invariance even with the flag set. — [Spheron blog](https://www.spheron.network/blog/vllm-batch-invariance-why-temperature-0-isnt-deterministic/)
- [Jan 2026] LLM-42 decodes on the normal fast path, then verifies candidate tokens under a fixed-shape reduction schedule and rolls back mismatches. This is analogous to speculative decoding, and overhead is "only in proportion to the traffic that requires determinism". According to the paper, SGLang's deterministic mode costs up to 56% (ACM version: 52%). — [arXiv 2601.17768](https://arxiv.org/html/2601.17768v1); [ACM](https://dl.acm.org/doi/pdf/10.1145/3830418.3843901)
- [Nov 2025] TP-invariant deterministic inference across tensor-parallel sizes reports ~18–99% overhead from the batch-invariant module relative to BF16 (search excerpt). — [arXiv 2511.17826](https://arxiv.org/pdf/2511.17826)
- [2026, titles seen only] Further work on cheaper determinism: "CoRun: Padding is Simple and Efficient for Deterministic LLM Inference" and "MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference". — [arXiv 2608.14376](https://arxiv.org/pdf/2608.14376); [arXiv 2605.30218](https://arxiv.org/pdf/2605.30218)
- [Nov 2023 onward] OpenAI's `seed` is best-effort: "Determinism is not guaranteed". `system_fingerprint` changes when OpenAI updates backend numerical configuration, so it should be monitored. Azure's docs: variability is "not uncommon" even when seed and fingerprint match, and the seed feature is in Preview there. — [OpenAI cookbook](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter); [Azure docs](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/how-to/reproducible-output)
- [Secondary, 2026] Anthropic's docs state that even at temperature 0.0 results are not fully deterministic, and (per a secondary source, May 2026) Anthropic ships no `seed` parameter. A framework bug silently dropped `temperature=0` for Claude through a truthiness check. — [Agno article](https://www.agno.com/articles/honor-temperature-0-on-claude-for-deterministic-output.md); [FutureAGI](https://futureagi.com/blog/llm-temperature/)

**Durable execution and deterministic agent loops**
- [Jul 30, 2025 → Mar 23, 2026] The Temporal + OpenAI Agents SDK integration (Python) reached **GA on March 23, 2026** after a July 2025 public preview. The agent loop, tool selection and handoffs run in a deterministic Workflow. Model calls are Activities, so they "retry durably and are not repeated during Workflow replay". The TypeScript version is pre-release. — [Temporal blog](https://temporal.io/blog/announcing-openai-agents-sdk-integration); [Temporal docs](https://docs.temporal.io/develop/python/integrations/openai-agents); [BusinessWire](https://www.businesswire.com/news/home/20250730783559/en/Temporal-and-OpenAI-Launch-Integration-for-Enterprises-Developing-Production-Agents)
- [Nov 6, 2025] Pydantic AI + Temporal: a single `TemporalAgent` wrapper turns model requests, tool calls and external APIs into Activities with retry policies, while coordination runs as a Workflow. On a crash the workflow replays and reuses recorded results for completed steps, and workflows can wait days for human input. — [Temporal blog](https://temporal.io/blog/build-durable-ai-agents-pydantic-ai-and-temporal)
- [2026, secondary] Pydantic AI moved from wrapper agents (`TemporalAgent`, `DBOSAgent`) to a composable Capabilities system (`TemporalDurability`). It supports Temporal, DBOS and Prefect. DBOS needs only Postgres or SQLite, but function tools must be marked `@DBOS.step` manually. Workflow code must use the engine-provided random, now and sleep to stay replay-deterministic. — [Cadenya](https://cadenya.com/other-content/pydantic-ai-agents-in-production); [DeepWiki pydantic-ai](https://deepwiki.com/pydantic/pydantic-ai/7-durable-execution-and-workflow-integration); [Developers Digest](https://www.developersdigest.tech/blog/durable-ai-agent-frameworks-compared-2026)
- [Docs, 2026] LangGraph: any graph with a checkpointer has durable execution. The docs advise making workflows "deterministic and idempotent" and wrapping side effects and non-deterministic operations in tasks. Durability modes are exit, async and sync.
  - Time-travel replay skips nodes *before* the checkpoint but **re-executes later nodes, including LLM calls**, which "may produce different results". It is a replay, not a cache.

  — [LangGraph durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution); [time travel](https://docs.langchain.com/oss/javascript/langgraph/use-time-travel.md)

**Record-and-replay testing**
- [Sep 17, 2026] Chronicle records an agent run at its nondeterministic boundaries (model calls, tool calls, routing decisions) as immutable envelopes. **Cut-point replay** serves a chosen subset of boundaries from the recording and runs the rest live against new code, which turns incidents into CI regression tests.
  - Recording overhead is ~23 µs per boundary (0.008% of a 300 ms call).
  - Full replay makes zero model calls and is **bit-stable across 20 repetitions**.
  - Cut-point tests caught every unsafe mutant; a stub-everything baseline caught none.

  — [arXiv 2609.20625](https://arxiv.org/pdf/2609.20625)
- [2025–2026] Open-source cassette tools:
  - agentloop: JSON cassettes. Replay **fails** if a request differs, if the request count changes, or if any interaction goes unplayed. — [GitHub](https://github.com/PDiTO/agentloop)
  - llm-cassette (Python): httpx-transport VCR for OpenAI and Anthropic SDKs, with a pytest plugin. — [GitHub](https://github.com/aozmekik/llm-cassette)
  - agentcassette: wraps the model-call function, so it is provider-agnostic. — [GitHub](https://github.com/aenealabs/agentcassette)
  - Sixty North recommends recording only model decisions while tools run live (langchain-replay). — [Sixty North](https://blog.sixty-north.com/deterministic-testing-for-langchain-agents.html)
  - A common practice is to match requests on a stable fingerprint that ignores timestamps, UUIDs and keys, fail loudly on unmatched requests, and scan cassettes for secrets. — [OracleDesk issue](https://github.com/OracleDesk/OracleDesk-Backend/issues/72)

**Caching and versioning**
- [2026] Exact-match cache: hash the full request (model, prompt, parameters), including system message, history, retrieved context and temperature, and include the model in the key. Semantic caches use embedding thresholds (Tyk default 0.95, partitioned by model and system prompt; MLflow cites 0.80–0.95). Model-version or template changes should invalidate entries. — [MLflow article](https://mlflow.org/articles/llm-cache-strategies/); [Tyk docs](https://tyk.io/docs/ai-management/ai-studio/semantic-caching.md)
- [2026] Audit and compliance guidance: a family name ("GPT-4") is insufficient, so pin exact dated builds and log the exact build, retrieval sources and tool calls so a decision can be replayed. — [Fiddler](https://www.fiddler.ai/blog/audit-ready-llm-logging); [Kognitos 2026 checklist](https://www.kognitos.com/blog/ai-audit-trail-requirements-2026-checklist/)
- [2026] Hosted-model drift is real. A deployed nutrition-estimation cascade ran 179 days across unannounced version changes, which made it non-comparable to its pinned benchmark. A scoping-review study notes that hosted proprietary workflows "cannot be guaranteed to reproduce original outputs". — [MAKE journal, doi 10.3390/make8100331](https://doi.org/10.3390/make8100331); [arXiv 2608.26885](https://arxiv.org/pdf/2608.26885)

**DSPy compilation and LLMOps tooling**
- [Docs, 2025–2026] DSPy saves optimizer output (instructions, demos, LM config) as JSON state, or the whole program via cloudpickle (dspy ≥ 2.6). The saved metadata records Python, DSPy and cloudpickle versions, and a version mismatch at load time logs a warning but proceeds. — [DSPy saving docs](https://dspy.ai/tutorials/saving/)
- [Secondary, 2026] In DSPy 3.3.0, GEPA-optimized `module_src` is part of serialized state. — [newreleases.io](https://newreleases.io/project/github/stanfordnlp/dspy/release/3.3.0)
- [2026] Langfuse: prompt versions with deployment labels (rollback is a label move), and `langfuse/experiment-action` runs experiments on a named dataset in CI, raising `RegressionError` and failing the job when scores miss a threshold. Promptfoo is a lighter declarative CI harness with assertions and red-teaming. W&B Weave traces, versions prompts and data, and evaluates. — [Langfuse prompt CI/CD](https://langfuse.com/resources/engineering/prompt-cicd); [Langfuse vs Promptfoo migration](https://langfuse.com/resources/engineering/migrate-from-promptfoo); [PromptLayer on Weave](https://www.promptlayer.com/glossary/wandb-weave)
- [2026] Golden-set practice (practitioner guides): 50–200 traces per agent, re-validated monthly, with production failures fed back; block deploys if the pass rate drops >3%. — [FutureAGI](https://futureagi.com/blog/llm-eval-golden-set-design-2026/); [Alok blog](https://aloknecessary.github.io/blogs/llm-evaluation-in-production/)
- [Jun 2026] vLLM-SR exposes router replay records, a topology dry-run, policy shadow/activate/revert, and unknown-YAML-field warnings. These are router-level harness features. — [vLLM blog v0.3](https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release)

### Inferences
- **Cache and lockfile key design.** The cited cache guidance says the key must include model, full prompt and parameters, and audit guidance says to pin exact builds. A compiler-grade key should go further and hash:
  - weights;
  - tokenizer;
  - chat template;
  - engine version and kernel flags (`VLLM_BATCH_INVARIANT`, attention backend, CUDA graphs, async scheduling off);
  - TP/EP layout and dtype;
  - sampling params and seed;
  - router policy version;
  - canonical prompt bytes;
  - tool-schema hash.

  Several of these components (TP layout, attention backend, async scheduling) are cited as determinism-breaking in the sources above.
- **Three determinism layers:**
  1. routing (pure function plus tie-break plus recorded decision);
  2. inference (batch-invariant kernels on self-hosted open weights; hosted APIs are cassette-recorded activities only);
  3. orchestration (durable-execution event history; tools idempotent with idempotency keys).
- **MoE caveat.** SGLang's deterministic mode excluded MoE models as of Sep 2025, while vLLM's 2026 docs list DeepSeek MoE models as tested. A deterministic MoE project should verify expert-parallel and all-to-all reduction order explicitly per engine version.
- **LangGraph replay re-executes post-checkpoint LLM calls.** For compiler-like semantics, pair it with a content-addressed cache or cassette layer, or use Temporal/DBOS, where completed activities are not re-run.
- **Test pyramid:**
  - unit: cassette replay with strict request-match and count checks;
  - integration: cut-point replay (Chronicle);
  - system: golden or ground-truth sets with CI thresholds;
  - nightly "determinism canary": N identical requests under concurrent load, asserting one unique output (the TML methodology).

### Gaps
- I found no documentation of determinism-specific features in the Claude Agent SDK (session resume/fork, hooks) or the OpenAI Agents SDK beyond the Temporal integration.
- Restate, Inngest and Burr determinism specifics were not researched in depth.
- MLflow's prompt-registry and CI features were not covered by the search results.
- How LiteLLM and DSPy construct cache keys internally was not found.
- vLLM batch-invariance throughput overhead was not found in official docs. LLM-42's full overhead tables were not extracted.

---

## Q4. Fallback and escalation strategies (confidence escalation, abstention, ensemble-agreement gates, human-in-the-loop) and keeping them deterministic

### Takeaway
Cascades (cheap model → verify/score → escalate) are the main fallback pattern. The 2026 literature focuses on **calibrating** the escalation signal: token-level calibrated confidence, shared confidence scales, and "expected benefit of escalation" rather than "is the cheap answer correct". It also adds deliberation or abstention stages that end in human experts. Escalation is deterministic only if:
- the confidence signal is computed on deterministic inference (or recorded);
- thresholds are versioned constants;
- ties and near-threshold cases follow a fixed rule.

Cascades are also attackable: an adversary can force deferral and drive up cost.

### Cited Findings
- [May 2023] FrugalGPT's cascade uses a learned response scorer to stop at the first acceptable answer (up to 98% cost reduction at GPT-4 parity), which requires labeled data. — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- [Oct 2023, NeurIPS 2024] AutoMix uses few-shot **self-verification** by the small model and a POMDP router to handle noisy verification. It reports >50% cost reduction at comparable performance. — [arXiv 2310.12963](https://arxiv.org/abs/2310.12963)
- [Jun 2025] A cascaded human-AI framework combines a **deferral** policy (regenerate with a large model) and an **abstention** policy (hand off to human experts when uncertainty stays high). — [arXiv 2506.11887](https://arxiv.org/html/2506.11887v2)
- [2026] Calibrated cascade work:
  - UCCI: calibrated token-level confidence for structured outputs, arguing that raw token confidences are miscalibrated. — [arXiv 2605.18796](https://arxiv.org/pdf/2605.18796)
  - CAUC: maps each model's confidence onto a shared scale for early accept, escalate and selective fusion, keeping the cheap answer as a fusion input. — [arXiv 2609.11446](https://arxiv.org/html/2609.11446v1)
  - Signed Rescue Routing: calibrates to the *benefit* of escalation, with two heads estimating P(escalation fixes an error) and P(escalation breaks a correct answer). — [arXiv 2609.07786](https://arxiv.org/pdf/2609.07786)
- [Aug 2026] Bayesian self-escalation decides **token by token during streaming**, so a doomed cheap generation can be aborted early instead of waiting for a full response. — [arXiv 2608.24087](https://arxiv.org/pdf/2608.24087)
- [Apr 2026] CascadeDebate inserts multi-agent deliberation at each escalation boundary, with human experts as the final fallback. It reports up to +26.75% over single-model cascades and standalone multi-agent systems. — [arXiv 2604.12262](https://arxiv.org/html/2604.12262v1)
- [Sep 2026] READY frames enterprise-agent accept-or-escalate as **selective prediction** using risk–coverage. — [arXiv 2609.02095](https://arxiv.org/pdf/2609.02095)
- [2026] Adversarial risk to cascades:
  - "Forced Deferral": adversarial images reduce weak-model confidence and force strong-model calls. — [arXiv 2606.15308](https://arxiv.org/html/2606.15308)
  - "When Efficiency Backfires: Cascading LLMs Trigger Cascade Failure under Adversarial Attack" (title only seen). — [arXiv 2605.17288](https://arxiv.org/pdf/2605.17288)
- [Jan 2026] LLMRouterBench: binary routers and cascades (HybridLLM, FrugalGPT) struggle to save cost without accuracy loss on flagship-model pools. — [arXiv 2601.07206](https://arxiv.org/html/2601.07206)
- [Jun 2026] vLLM-SR hard locks prevent model switches mid tool loop and route tool results back to the requesting model. This is a deterministic "no-escalation-mid-transaction" rule. — [vLLM blog v0.3](https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release)
- [Nov 2025] Temporal workflows can block for hours or days awaiting human input, then resume deterministically. This is the substrate for human-in-the-loop escalation. — [Temporal blog](https://temporal.io/blog/build-durable-ai-agents-pydantic-ai-and-temporal)
- [Dec 2024] Anthropic recommends programmatic "gates" between prompt-chain steps and the evaluator-optimizer loop when clear evaluation criteria exist. — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [Secondary, 2025] Shopify Sidekick's LLM judges were iterated from Cohen's κ 0.02 to 0.61 against a human-agreement ceiling of 0.69. Judge reliability bounds any judge-based gate. — [ZenML summary](https://www.zenml.io/llmops-database/building-production-ready-ai-assistant-with-agentic-architecture); [Shopify Engineering](https://shopify.engineering/building-production-ready-agentic-systems)

### Inferences
- **Making escalation deterministic:**
  - compute confidence (logprob margin, verifier score) on batch-invariant inference, or record it in the trace;
  - compare it against versioned thresholds stored in the lockfile;
  - add a dead band around the threshold that maps to a fixed tier (analogous to OrcaRouter's ε-tie-break), so tiny numeric drift cannot flip the decision;
  - make ensemble-agreement gates use exact-match or canonicalized-answer equality with a fixed member order and a fixed tie rule.
- Prefer verifiers that are themselves deterministic programs where possible (unit tests, schema validators, type checkers, symbolic math). LLM judges should be the last resort and calibrated against a human ceiling.
- Escalation budgets (max tiers, max retries) should be static constants to bound cost against Forced-Deferral-style attacks.

### Gaps
- No primary source quantifies how often cascade decisions flip across reruns on hosted APIs.
- No standard benchmark for the determinism of escalation policies was found.
- Most 2026 cascade papers were read at abstract level only, without per-dataset numbers.

---

## Q5. Industry case studies (2024–2026) of deterministic or high-reliability LLM pipelines, and "LLM as compiler" analogies

### Takeaway
The clearest production pattern is **hybrid deterministic-plus-agentic graphs**: Stripe's "blueprints" interleave fixed code nodes with agent loops, and Anthropic's workflow/agent distinction describes the same split. These are combined with production-derived ground-truth evaluation sets (Shopify) and multi-provider routing with reliability-based failover (Harvey). The "LLM as compiler" framing has moved into research systems that confine the LLM to a compile phase and then run deterministic artifacts:
- Compiled AI: 96% task completion with zero runtime tokens;
- PlanCompiler: typed node registry plus static validator;
- StructCore: deterministic compiler stage with bit-for-bit reproducibility.

I found no named finance or healthcare deployment with a published, quantified deterministic LLM pipeline.

### Cited Findings
- [Mar 2026] Stripe "Minions" are unattended coding agents producing "over 1,300 pull requests per week" (up from 1,000). All code is human-reviewed. They evolved from a fork of Block's Goose agent. Code-defined "blueprints" split work into subtasks handled by deterministic routines or by the agent, and reliability rests on CI, automated tests and static analysis before review. — [InfoQ](https://infoq.com/news/2026/03/stripe-autonomous-coding-agents/)
- [Secondary, 2026] Stripe blueprints are state machines interleaving deterministic nodes (lint, branch creation, push, test validation, with no LLM call) and agentic nodes (implement, fix CI, write tests). Guidance: "if a step should always happen the same way, the model shouldn't decide how to do it", which also cuts tokens. — [ByteByteGo](https://blog.bytebytego.com/p/how-stripes-minions-ship-1300-prs); [MindStudio](https://www.mindstudio.ai/blog/stripe-minions-blueprint-architecture-deterministic-agentic-nodes)
- [2025] Shopify Sidekick replaced static golden datasets with **Ground Truth Sets** sampled from real merchant conversations, labeled by product experts. It uses statistically validated LLM judges and an LLM merchant simulator to replay goals against candidate system changes before deploy. GRPO training showed reward hacking, which they countered by refining judges and datasets. — [Shopify Engineering](https://shopify.engineering/building-production-ready-agentic-systems); [RubyEvents talk](https://www.rubyevents.org/talks/llm-evaluations-reinforcement-learning-for-shopify-sidekick-on-rails)
- [May 2025 → May 2026] Harvey (legal) routes across Anthropic, Google and OpenAI by default ("Auto" mode), and admins can enable a Model Selector. It evaluates each model on BigLaw Bench before inclusion. Job postings (indicative only) describe a "Unified Model Controller" that detects model degradations and routes by reliability, latency, quality, compliance and cost, with cross-provider failover. — [Harvey blog](https://www.harvey.ai/blog/expanding-harveys-model-offerings); [Harvey help center](https://help.harvey.ai/articles/harvey-ai-overview); [job listing](https://jobsbyculture.com/jobs/harvey/staff-software-engineer-model-infrastructure)
- [Dec 2024] Anthropic recommends starting with the simplest solution, distinguishing workflows ("predefined code paths") from agents ("LLMs dynamically direct their own processes"), and using routing to a small vs large model as a canonical workflow. — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [Apr 2026] "Compiled AI": the LLM generates code in a compile phase and workflows then run with no model calls. It reports 96% task completion on function-calling with **zero execution tokens** and break-even versus runtime inference at ~17 transactions. — [arXiv 2604.05150](https://arxiv.org/abs/2604.05150)
- [Apr 2026] PlanCompiler: the LLM only selects and parameterizes nodes from a fixed, typed registry, and a static validator plus deterministic compiler emit code. It reports much higher first-pass reliability than direct prompting on a 300-task benchmark. — [arXiv 2604.13092](https://arxiv.org/pdf/2604.13092)
- [Apr 2026] LLM StructCore: the LLM-based extraction variant showed vocabulary drift and run-to-run inconsistency, while the deterministic compiler stage gives "perfect vocabulary compliance" and "bit-for-bit reproducibility". — [arXiv 2604.20560](https://arxiv.org/pdf/2604.20560)
- [Nov 2025] An end-to-end "LLM as a compiler" study found models show basic compiler abilities but "low compilation success rates". — [arXiv 2511.04132](https://arxiv.org/abs/2511.04132)
- [May 2026] Fully Open Meditron: an auditable, open pipeline for clinical LLMs, positioned as a model for reproducible validation (abstract level). — [arXiv 2605.16215](https://arxiv.org/abs/2605.16215)
- [2026, title only] A contrarian position paper argues that deterministic inference harms model cognition: "Stochastic CHAOS: Why Deterministic Inference Kills…". — [arXiv 2601.07239](https://arxiv.org/pdf/2601.07239)
- [2026, title only] "Reproducibility is the New Copyleft: Defining AGI-oriented Reproducible Builds" applies the reproducible-builds concept to AI systems. — [arXiv 2606.03019](https://arxiv.org/pdf/2606.03019)

### Inferences
- The practical "compiler" analogy maps as follows:
  - source = task spec plus inputs;
  - front-end = canonicalizer and task classifier (deterministic);
  - optimizer = DSPy/GEPA-compiled prompts and route tables (built offline and versioned);
  - code generation = per-specialist LLM calls (deterministic only on batch-invariant self-hosted kernels; otherwise recorded);
  - linker = deterministic glue nodes (Stripe-style);
  - test suite = cassette, cut-point and golden sets;
  - build reproducibility = lockfile of model, tokenizer, engine and kernel hashes.
- The strongest determinism gains in industry come from *removing* LLM decisions from steps that should be fixed (Stripe, Compiled AI, PlanCompiler), not from making LLM sampling deterministic.

### Gaps
- Stripe's own blog posts on Minions and blueprints were not fetched; the details come via InfoQ and secondary write-ups.
- No primary Uber, Databricks, Ramp, bank or hospital write-up describing a quantified deterministic multi-model pipeline was found in this pass.
- No company publicly reports a paraphrase-stability SLO for routing.

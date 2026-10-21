# MoE Interpretability, Circuit Discovery and Interventions for Predictability (state as of Oct 21, 2026)

Scope: methods and tools to map every layer, expert, attention head and router of a Mixture-of-Experts (MoE) LLM; to find circuits behind task behaviors (decisions, tool calls, judgments, generation, multi-step reasoning); and to intervene (steering, ablation, expert pinning or deactivation, router editing, clamping) so outputs vary less and are more predictable. Every finding is dated. "[FOUNDATIONAL, pre-2025]" marks older work that later work may supersede. arXiv IDs 26xx.xxxxx are 2026 preprints, and most of them are not yet peer-reviewed unless a venue is named.

Research-process note: the shared web-search budget ran out near the end of this session. Transluce, Apollo (parameter decomposition) and some tool support tables were therefore not researched; see the Gaps sections.

---

## Q1. Circuit discovery methods and how far they scale (patching family, attribution graphs/CLTs, SAEs/transcoders/crosscoders, Scope releases, lenses)

### Takeaway
As of Oct 2026 the standard pipeline is: (a) cheap gradient-based attribution patching (EAP/EAP-IG) to localize components, (b) a learned sparse dictionary (SAE, transcoder or cross-layer transcoder) to name the features, and (c) attribution graphs (circuit-tracer or Neuronpedia) checked by causal interventions. Attribution graphs still give "satisfying insight" on only about a quarter of prompts (Anthropic, Mar 2025). Open dictionaries now exist for a few MoE models: Qwen-Scope residual SAEs for Qwen3-30B-A3B and Qwen3.5-35B-A3B (May 2026), a CLT and BatchTopK SAEs for gpt-oss-20b, and single-layer SAEs for DeepSeek-R1. I found no complete per-layer transcoder/CLT replacement model for any Qwen or DeepSeek MoE.

### Cited Findings

**Patching / attribution family (component-level causal localization)**
- [FOUNDATIONAL, 2022] Causal tracing / activation patching (ROME) locates where factual associations are mediated by corrupting inputs and restoring hidden states one at a time. Cost is one forward pass per (component × position), so it scales poorly as a full sweep. — [Meng et al., ROME](https://arxiv.org/abs/2202.05262)
- [FOUNDATIONAL, 2023] Path patching restricts patching to specific paths (sender to receiver components) to test hypothesized circuits. — [Goldowsky-Dill et al.](https://arxiv.org/abs/2304.05969)
- [FOUNDATIONAL, 2023] ACDC automates circuit discovery by iteratively pruning edges using activation patching. It is expensive (many forward passes per edge) and largely superseded for large models by gradient approximations. — [Conmy et al., ACDC](https://arxiv.org/abs/2304.14997)
- [FOUNDATIONAL, 2023] Edge Attribution Patching (EAP) estimates the importance of every edge from two forward passes and one backward pass, and outperforms ACDC at circuit recovery. — [Syed, Rager, Conmy, "Attribution Patching Outperforms Automated Circuit Discovery"](https://arxiv.org/abs/2310.10348); the forward/backward-pass cost is restated in [arXiv 2502.11812](https://arxiv.org/pdf/2502.11812)
- [2024] EAP-IG integrates gradients along the clean→corrupted path to fix EAP's linear-approximation errors, and argues circuits should be judged by faithfulness, not by overlap with a reference circuit. — [Hanna et al., "Have Faith in Faithfulness"](https://arxiv.org/abs/2403.17806)
- [2024] AtP* (Google DeepMind) is an efficient, scalable attribution-patching variant for localizing behavior to components in large LLMs, with corrections for attention saturation and cancellation. — [Kramár et al., AtP*](https://arxiv.org/abs/2403.00745)
- [2025, ICML] The MIB benchmark (4 tasks, 5 models) found that attribution methods and mask-optimization methods perform best on circuit localization. Its baselines include activation patching, attribution patching, AP-IG, information-flow routes and uniform gradient sampling. — [MIB, PMLR v267](https://proceedings.mlr.press/v267/mueller25a.html); [arXiv 2504.13151](https://arxiv.org/abs/2504.13151)
- [2024] Sparse Feature Circuits: circuits whose nodes are SAE features rather than neurons or heads, discovered with attribution patching. — [Marks et al.](https://arxiv.org/abs/2403.19647)

**Lenses (layer-wise readouts)**
- [FOUNDATIONAL, 2020] The logit lens decodes intermediate residual streams through the final unembedding to see what each layer "predicts". — [nostalgebraist, LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- [FOUNDATIONAL, 2023] The tuned lens learns a per-layer affine translator before unembedding, which makes it more reliable than the logit lens on early layers. — [Belrose et al., Tuned Lens](https://arxiv.org/abs/2303.08112)
- [2026] MoE Lens applies logit-lens-style early decoding to each expert's contribution to the residual stream at each layer. It reports that a single expert can be enough to reach the output representation for next-token prediction. — [arXiv 2603.05806](https://arxiv.org/html/2603.05806v1)

**Sparse dictionaries: SAEs, transcoders, crosscoders, CLTs**
- [FOUNDATIONAL, 2024] Scaling evidence: Anthropic trained SAEs of up to 34M features on Claude 3 Sonnet ([Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)), and OpenAI trained a 16M-latent TopK SAE on GPT-4 ([Gao et al.](https://arxiv.org/abs/2406.04093)).
- [2024] Transcoders replace an MLP with a sparse, wide MLP that is trained to imitate it. Because they approximate a computation rather than just reconstruct an activation, they give input-invariant feature circuits. — [Dunefsky et al., Transcoders](https://arxiv.org/abs/2406.11944)
- [2024] Crosscoders read from or write to multiple layers or models. They are used for cross-layer features and for model diffing (e.g., base vs. chat). — [Anthropic, Crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html); follow-up [Insights on Crosscoder Model Diffing, Feb 2025](https://transformer-circuits.pub/)
- [Mar 2025] Anthropic "Circuit Tracing" (methods paper):
  - A cross-layer transcoder (CLT) replaces all MLPs, giving a "replacement model". CLT sizes ran from 300K to 10M features on an 18-layer model and from 300K to 30M on Claude 3.5 Haiku.
  - The largest 18L CLT matches the model's next-token completion on about 50% of prompts. Normalized reconstruction error is about 11.5% at L0≈88 for the 18L model and 21.7% at L0≈235 for Haiku at 30M features.
  - Attention patterns are frozen, so QK circuits are not explained. Error nodes absorb the residual reconstruction error.
  - Pruning cuts nodes by about 10× for about 20% loss of explained behavior (completeness 0.70 with 55 nodes at threshold 0.8; 0.87 with 236 nodes at 0.95).
  - CLT vs. per-layer transcoders at 10M features: completeness 0.80 vs. 0.78, replacement score 0.61 vs. 0.37, average path length 2.3 vs. 3.7.
  - Intervention faithfulness: about 0.8 cosine one layer after a perturbation, with errors compounding across layers. Feature-to-feature influence predicts ablation effects with Spearman ρ = 0.72. The up-front CLT training cost is described as "significant". — [Ameisen et al., Circuit Tracing methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [Mar 27, 2025] "On the Biology of a Large Language Model" (Claude 3.5 Haiku, 30M-feature CLT). Case studies: multi-hop reasoning (Dallas→Texas→Austin), planning in poems, multilingual circuits, addition, medical diagnosis, entity recognition and hallucination, refusals, a jailbreak, CoT faithfulness, hidden goals.
  - Key limitation, verbatim: "our attribution graphs provide us with satisfying insight for about a quarter of the prompts we've tried."
  - It found a default "can't answer" circuit that "known entity/known answer" features inhibit. Hallucinations arise when this inhibition misfires. — [Lindsey et al., Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Jul 2025] Anthropic extended attribution graphs to explain attention with "QK attributions", which score head attention as a bilinear function of query-side and key-side features. This addresses the frozen-attention limitation. — [Kamath et al., Tracing Attention Computation](https://transformer-circuits.pub/2025/attention-qk/index.html)
- [Jul 2025] Anthropic also published "Sparse mixtures of linear transforms" (MOLT), an alternative transcoder architecture, and "A Toy Model of Mechanistic (Un)Faithfulness", which shows ways transcoders can mislead. — [Transformer Circuits index](https://transformer-circuits.pub/)
- [May 2026] Anthropic used TWERA-weighted virtual weights to rank a feature's downstream targets. This "modestly" improved an LLM's ability to predict which feature will steer a behavior, but the problem is "far from solved". — [Circuits Updates May 2026](https://transformer-circuits.pub/2026/may-update/index.html)
- [Jun 2026] Anthropic introduced turn-averaged SAEs for long multi-turn transcripts, where per-token SAEs produce "thousands to millions" of feature activations. It includes a 14-turn case study and a nested-vs-per-token SAE comparison for attribution graphs. — [Circuits Updates June 2026](https://transformer-circuits.pub/2026/june-update/index.html)
- [May 2026] Natural Language Autoencoders train Claude to verbalize its activations. One case applies attribution graphs to a model ignoring a tool result: an "ends in 92" feature from the tool output loses to an "ends in 91" feature from an answer the model had already computed. This is a direct example of mechanistic diagnosis of a tool-use failure. — [Transformer Circuits, NLA](https://transformer-circuits.pub/2026/nla/)
- [Jul 2026] "Verbalizable Representations Form a Global Workspace" (Claude) separates a small set of reportable, controllable representations from much larger automatic processing. — [Gurnee et al.](https://transformer-circuits.pub/2026/workspace/)

**Open-source attribution graphs**
- [May 29, 2025] Anthropic open-sourced `circuit-tracer` with a Neuronpedia frontend. It was led by Anthropic Fellows (Hanna, Piotrowski) with Decode Research, and launched with Gemma-2-2B and Llama-3.2-1B examples. — [Anthropic announcement](https://www.anthropic.com/research/open-source-circuit-tracing)
- [Oct 2026 README] `circuit-tracer` supported transcoders:
  - Gemma-2-2B: per-layer transcoders (PLTs) from GemmaScope, plus CLTs with 426K and 2.5M features.
  - Llama-3.2-1B: PLTs and a 524K CLT.
  - Qwen3 0.6B/1.7B/4B/8B/14B: PLTs.
  - **GPT-OSS-20B (an MoE): a 131K CLT.**
  - Gemma-3 270M–27B PT/IT: PLTs; these need the nnsight backend.
  - Llama-3.1-8B-Instruct: TopK PLTs.
  - Backends: TransformerLens (default) and nnsight ("still experimental", slower, less memory-efficient).
  - Defaults: node_threshold 0.8, edge_threshold 0.98, max 7,500 feature nodes, batch size 256. Gemma-2-2B runs on a 15GB Colab GPU. Offloading is available.
  - The README does not discuss MoE handling. — [safety-research/circuit-tracer](https://github.com/safety-research/circuit-tracer)
- [Jun 13, 2025] Neuronpedia launched on-demand graph generation (Gemma-2-2B only at launch). Graphs take "a few seconds"; about 5,000 graphs had been generated at the time of the post. — [Neuronpedia blog](https://www.neuronpedia.org/blog/circuit-tracer)
- [Nov 12, 2025] Neuronpedia added:
  - Graph steering of nodes and supernodes for Qwen3-4B.
  - gpt-oss-20b dashboards and inference using Andy Arditi's BatchTopK SAEs.
  - Anthropic Fellows CLTs for Gemma-2-2B.
  - Llama 3.3 70B with Goodfire's layer-50 SAE. — [Neuronpedia Fall Update](https://www.neuronpedia.org/blog/fall-update)
- [2025–2026] OpenMOSS (Fudan) "Complete Replacement Models" (Llama Scope 2) add LORSA (low-rank sparse attention) nodes next to transcoder nodes, giving QK-traceable attention in graphs. Neuronpedia generates CRM graphs for Qwen3-1.7B (e.g., a 16k-feature, k=64 LORSA). — [Neuronpedia Llama Scope 2](https://www.neuronpedia.org/llama-scope-2)
- [Mar 2026, rev. Jul 2026] CLT-Forge is an open-source library for end-to-end CLT training: distributed training with model sharding, compressed activation caching, autointerp, attribution graphs via circuit-tracer, and visualization. — [Draye et al., arXiv 2603.21014](https://arxiv.org/abs/2603.21014); code at github.com/LLM-Interp/CLT-Forge (per abstract)
- [Apr 2026] ADAG automatically describes attribution graphs. It uses "attribution profiles" (input/output gradient effects), a new clustering algorithm and an explainer-simulator. It recovers known circuits and finds steerable feature clusters behind a harmful-advice jailbreak in Llama 3.1 8B Instruct. — [Arora, Wu, Steinhardt, Schwettmann, arXiv 2604.07615](https://arxiv.org/abs/2604.07615)

**"Scope" dictionary releases**
- [Aug 2024] Gemma Scope (Gemma 2): JumpReLU SAEs on all layers and sub-layers of Gemma 2 2B/9B and select layers of 27B. More than 400 SAEs (over 2,000 counting sparsity variants) and "hundreds of billions" of SAE parameters.
  - Compute: "over 20% of the training compute of GPT-3" and about 20 PiB of activations saved to disk.
  - Hardware: mostly TPUv3 4x2, with the widest SAEs on TPUv5p. — [Lieberum et al., Gemma Scope](https://arxiv.org/html/2408.05147)
  - Superseded by Gemma Scope 2 for current Gemma models.
- [Dec 19, 2025; InfoQ coverage Jan 2026] Gemma Scope 2 (Gemma 3, 270M–27B, PT and IT):
  - SAEs on three sites plus transcoders for every layer, skip-transcoders, Matryoshka training.
  - Partial residual-stream crosscoders for every base model.
  - CLTs only for 270M and 1B.
  - Reportedly about 110 PB stored and over 1T total parameters trained. This figure comes from secondary coverage and is not verified against the primary source. — [InfoQ](https://infoq.com/news/2026/01/google-gemma-scope-2/); [HF google/gemma-scope-2-27b-it](https://huggingface.co/google/gemma-scope-2-27b-it); [OpenSourceForU, Dec 2025](https://www.opensourceforu.com/2025/12/google-deepmind-open-sources-gemma-scope-2/); [weel.co.jp (date)](https://weel.co.jp/media/tech/gemma-scope-2/)
- [Oct 2024] Llama Scope (OpenMOSS): SAEs across Llama-3.1-8B layers and sub-layers. — [arXiv 2410.20526](https://arxiv.org/abs/2410.20526). Its successor, Llama Scope 2, is listed above (LORSA/CRM on Qwen3).
- [May 12, 2026] **Qwen-Scope** (Qwen team) — the most relevant release for a Qwen MoE:
  - 14 SAE groups across 7 backbones, all residual-stream SAEs, one per layer, TopK with k = 50 or 100.
  - MoE coverage: **Qwen3-30B-A3B-Base, layers 1–48**, and **Qwen3.5-35B-A3B-Base, layers 1–40**. Each has a W32K (L0 = 50, 16× expansion) SAE and a wide W128K (L0 = 100, 64× expansion) SAE.
  - Dense coverage: Qwen3-1.7B and 8B; Qwen3.5-2B, 9B and 27B.
  - Training: in-house pretraining data (≈0.5B tokens per the ModelScope announcement), auxiliary dead-latent loss weight 1/32, and filtering of extreme-norm (often first-token) activations.
  - Not reported: token counts per SAE, GPU-hours, FVE, measured L0.
  - Released on HF (huggingface.co/collections/Qwen/qwen-scope) and ModelScope. — [Qwen-Scope paper, arXiv 2605.11887](https://arxiv.org/html/2605.11887v1); [Qwen blog](https://qwen.ai/blog?id=qwen-scope); [ModelScope announcement](https://x.com/ModelScope2022/status/2052009415000076568)
- [Apr 15, 2025] Goodfire released SAEs on **DeepSeek-R1 (671B MoE)**: two SAEs, one on a custom reasoning dataset (r1-collect) and one on OpenR1-Math. The HF repo name suggests layer 37. They needed a custom inference engine for "half-trillion-parameter scale". Code: goodfire-ai/r1-interpretability. — [Goodfire, Under the Hood of a Reasoning Model](https://www.goodfire.com/research/under-the-hood-of-a-reasoning-model)
- [Sep 2025] gpt-oss-20b BatchTopK SAEs by Andy Arditi: residual-stream (post) site, 131k features, hosted on Neuronpedia. — [Neuronpedia gpt-oss-20b](https://www.neuronpedia.org/gpt-oss-20b)

**Evidence that tempers SAE enthusiasm**
- [Mar 26, 2025] Google DeepMind mech-interp team:
  - Dense linear probes "perform nearly perfectly, including out of distribution" on harmful-intent detection, while 1-sparse SAE probes are "much worse".
  - Chat-specialized SAEs closed about half the gap.
  - The team is "deprioritising fundamental SAE research". — [GDM progress update #2, Alignment Forum](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)
- [Jan 2025] AxBench (Gemma-2-2B/9B) concept-steering scores: prompting 0.894, DiffMean 0.239, SAEs 0.165. A different paper version lists DiffMean 0.409 and SAE 0.315. — [AxBench, arXiv 2501.17148](https://arxiv.org/abs/2501.17148)

### Inferences
- For a 30B-class MoE, the realistic circuit stack is: EAP-IG/AtP* over coarse nodes (attention heads, whole experts, shared expert, router logits) for localization; Qwen-Scope residual SAEs (if the target is Qwen3-30B-A3B or Qwen3.5-35B-A3B) for feature labels; then a CLT trained in-house (CLT-Forge) if full attribution graphs are needed. No public CLT or transcoder exists for Qwen MoE models, but the gpt-oss-20b CLT in circuit-tracer shows MoE blocks can be replaced by a CLT.
- A CLT that replaces a whole MoE block absorbs the router's top-k nonlinearity into the transcoder. Feature-level graphs therefore lose expert identity unless experts are also tracked as separate nodes (e.g., via routing logs or per-expert transcoders). This is a design choice the program must make explicitly.
- Since graphs give satisfying insight on only about 25% of prompts (Mar 2025) and reconstruction error on a production model was about 22%, graphs should generate hypotheses. Interventions (patching, ablation, steering) are the acceptance test.

### Gaps
- No published full-model per-layer transcoder or CLT, and no attribution-graph study, for Qwen3-30B-A3B, Qwen3.5-35B-A3B or any DeepSeek-V3-family model was found.
- Qwen-Scope does not report training compute, FVE or dead-latent counts. The license was not visible in the parts read.
- Gemma Scope 2's compute figures (110 PB, over 1T parameters) were only seen in secondary coverage.
- Neither the cost of computing one attribution graph at 30B scale nor CLT training cost relative to model pretraining was found. Anthropic's appendix cost figures were in the unread part of the page.
- EAP-IG has not been benchmarked on MoE models with experts as nodes (MIB used dense models).

---

## Q2. MoE-specific interpretability: expert specialization, super experts, expert attribution, router probing and lenses, SteerMoE, expert ablation, monosemanticity of experts

### Takeaway
The 2025–2026 MoE literature agrees on three points:
1. A handful of early-layer **"super experts"** create massive activations and attention sinks and must never be ablated. Pruning 3 of 6,144 experts collapses Qwen3-30B-A3B.
2. Small sets of experts are **causally linked to behaviors**: safety, faithfulness, reasoning effort. Switching them on or off via router-logit edits steers behavior (SteerMoE, SAFEx, RICE).
3. **Expert "modularity" is weaker than routing statistics suggest.** Routing mostly mirrors hidden-state geometry, experts hold superposed fine-grained features, and a pre-registered causal test on a 218B MoE found only 1 of 6 expert families cleanly selective.

The emerging consensus is that the useful units are "fine-grained task experts" and **routing trajectories (paths)**, not domain-level experts.

### Cited Findings

**Classic routing analyses (older; partly superseded)**
- [FOUNDATIONAL, Jan 2024] Mixtral 8x7B's routing analysis found no obvious topic-based expert assignment. Routing showed syntactic or positional regularities instead. — [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [FOUNDATIONAL, Jan 2024] DeepSeekMoE argues that fine-grained experts plus always-on shared experts improve specialization and reduce redundancy. Shared experts are a non-routed component that any program must analyze separately. — [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [FOUNDATIONAL, Sep 2024] The OLMoE paper analyzes router saturation, expert co-activation, and domain and vocabulary specialization in a fully open MoE (1B active / 7B total; 8 of 64 experts). It is the standard open testbed with released checkpoints and data. — [OLMoE](https://arxiv.org/abs/2409.02060)

**Super experts / massive activations / attention sinks**
- [FOUNDATIONAL, 2023–2024] Background mechanisms:
  - Attention sinks: initial tokens absorb attention mass. — [StreamingLLM](https://arxiv.org/abs/2309.17453)
  - Massive activations: a few huge residual-stream values act as biases tied to sinks. — [Sun et al.](https://arxiv.org/abs/2402.17762)
  - gpt-oss builds learned attention sinks into each head. — [gpt-oss model card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)
- [Jul 2025; ICLR 2026] "Unveiling Super Experts":
  - Identification: an expert is a super expert (SE) if its max down_proj output magnitude exceeds the 99.5th percentile of all experts, is at least 1/10 of the model maximum, and sits in a layer where massive activations form. Locations are stable across calibration sets (C4, WikiText-2, C-Eval, GSM8K, HumanEval) and unchanged by post-training.
  - Qwen3-30B-A3B: **3 of 6,144** (L1 E68, L2 E92, L3 E82). DeepSeek-R1: 10 of 15,677. DeepSeek-V2-Lite: 2 of 1,782. Mixtral-8x7B: 1 of 256 (L1 E3).
  - Effect of pruning SEs:
    - Qwen3-30B-A3B (thinking): Math-500 88.0→1.4, AIME→0, WikiText-2 perplexity 8.70→59.86.
    - DeepSeek-R1: Math-500 97.6→4.0.
    - Non-reasoning models: average accuracy −22–27%, GSM8K −53–74%.
    - Output becomes repetitive text up to the length limit. Pruning the same number of random experts has negligible effect.
  - Mechanism: the sink token is routed to SEs, and attention sinks decay about 90% when SEs are pruned. A profiling tool is released. — [Su et al., arXiv 2507.23279](https://arxiv.org/html/2507.23279v2); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010189)

**Behavior-linked experts and (de)activation steering**
- [Sep 2025; ICLR 2026] **SteerMoE** (UCLA/Adobe/LMU):
  - Detection: per-expert activation-rate risk difference Δᵢ = p₁ − p₂ on paired contrastive prompts (e.g., faithful vs. unfaithful), ranked by |Δᵢ|.
  - Steering: a soft router-logit edit. Log-softmax scores are set to max+ε for experts to activate and min−ε for experts to deactivate, then re-softmaxed before normal top-k. No weight changes.
  - Models: GPT-OSS-120B/20B, Qwen3-30B-A3B, OLMoE-1B-7B-Instruct, Phi-3.5-MoE, Mixtral-8x7B-Instruct.
  - Scale of edits: e.g., Qwen3-30B-A3B safe steering forces 15 experts on and 500 off; GPT-OSS-120B faithfulness uses 5 on and 100 off.
  - Results across 11 benchmarks: up to +20% safety and +27% faithfulness. "Unsafe" steering cuts safety 41%, and −100% when combined with an AIM jailbreak (GPT-OSS-120B 100%→0% safe on 50 AdvBench prompts).
  - Capability checks: only fluency (LLM-judged) and MCTest; MMLU not reported. No seeds, CIs or variance reported. Logging routing costs "virtually nothing". — [SteerMoE, arXiv 2509.09660](https://arxiv.org/html/2509.09660v1); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d61c5f5ef91e7e8a091b7b8f72b853c-Abstract-Conference.html); code adobe-research/SteerMoE ([Adobe Research](https://research.adobe.com/publication/steering-moe-llms-via-expert-deactivation))
- [Jun 2025; NeurIPS 2025] **SAFEx** on Qwen3-30B-A3B (48 MoE layers × 128 experts, top-8):
  - Stability-based selection finds safety-critical experts and splits them into a harmful-content-detection group and a harmful-response-control group.
  - Disabling 12 experts cuts refusal rate by 22%.
  - LoRA plus "negative merging" on the identified experts raised adversarial refusal from 47.7% to 81.5% with negligible AIME impact. — [SAFEx, arXiv 2506.17368](https://arxiv.org/abs/2506.17368); [code](https://github.com/Bearisbug/SAFEx)
- [2026] RASET (arXiv 2605.29708) reports "localized safety enforcement failures" via router-agnostic safety-critical expert tuning. Only the title and abstract snippet were seen. — [RASET](https://arxiv.org/pdf/2605.29708)
- [May 2025] **RICE** ("Two Experts Are All You Need for Steering Thinking"):
  - Uses normalized PMI between experts and reasoning-marker tokens (e.g., `<think>`) to find "cognitive experts" in DeepSeek-R1 and Qwen3-235B.
  - Reinforcing them at inference, with no training, gives "noticeable and consistent" gains in reasoning accuracy and efficiency, beats prompt-design and decoding-constraint baselines, and preserves instruction following.
  - The abstract gives no numbers. — [arXiv 2505.14681](https://arxiv.org/abs/2505.14681)
- [Dec 2025] GateBreaker (gate-guided attacks on MoE LLMs) and [Apr 2025] BadMoE (backdoors via routing triggers and dormant experts) show the router is a control surface that attackers can also use. Titles only. — [GateBreaker](https://arxiv.org/pdf/2512.21008); [BadMoE](https://arxiv.org/pdf/2504.18598)

**Is expert specialization real? (2026 debate)**
- [Apr 2026] "The Myth of Expert Specialization in MoEs" (5 pre-trained MoEs):
  - Routers are linear, so hidden-state similarity is necessary and sufficient to explain similar expert usage.
  - Load balancing suppresses shared directions.
  - Expert overlap between models answering the same question is about 60%, no higher than for different questions.
  - Prompt-level routing does not predict rollout-level routing.
  - Deeper layers show near-identical expert activation across unrelated inputs, especially in reasoning models. — [Wang, Hayou, Nalisnick, arXiv 2604.09780](https://arxiv.org/abs/2604.09780)
- [Jun 23, 2026] Pre-registered causal test on **Command A+** (218B total / 25B active, 128 experts, top-8 plus 1 shared):
  - Setup: built a routing-mass atlas, pre-registered 6 family-to-capability hypotheses, ablated each family against size-matched random-expert nulls, and required worst off-target effect ≤ 1/3 of on-target, across 4 metrics, a held-out corpus and bootstrap CIs.
  - Result: only **1 of 6** families (Arabic) was cleanly selective under the conservative bar; 3 of 6 under a permissive bar, which is threshold-sensitive.
  - A positive control on Qwen3-30B-A3B recovered its known disjoint structure. Results held in BF16, so they are not a 4-bit artifact. — [Salomone, Gandhi, Asaria, arXiv 2606.25092](https://arxiv.org/abs/2606.25092)
- [Apr 2026; ICML 2026] "The Expert Strikes Back":
  - k-sparse probing shows expert neurons are consistently less polysemantic than dense FFN neurons, and the gap widens as routing gets sparser.
  - Hundreds of experts were auto-interpreted. Experts are neither domain specialists nor token-level processors; they are "fine-grained task experts" (e.g., closing brackets in LaTeX).
  - A secondary summary says 12 models were studied, including the OLMoE and Mixtral families. — [Herbst, Wermter, Lee, arXiv 2604.02178](https://arxiv.org/abs/2604.02178); [paper note (secondary)](https://en.papernotes.org/ICML2026/interpretability/the_expert_strikes_back_interpreting_mixture-of-experts_language_models_at_exper/); code [jerryy33/MoE_analysis](https://github.com/jerryy33/MoE_analysis)
- [Apr 2026] "Polysemantic Experts, Monosemantic Paths" (Ye, Yuan, Sharkey; 6 MoE architectures):
  - Parameter-free split of each MoE layer's hidden state into a **control signal** that causally drives routing and an orthogonal **content channel** the router can't see (language, token identity, position).
  - Individual experts are polysemantic, but **expert paths across layers are monosemantic**. For example, ":" takes different paths as a type annotation, an introductory colon, or a time separator.
  - Conclusion: the unit of interpretability is "not the expert but the trajectory." — [arXiv 2604.17837](https://arxiv.org/abs/2604.17837). A secondary snippet reports small-subspace probes predicting OLMoE's top-1 routing with 30–80% accuracy. — [search snippet via arXiv PDF](https://arxiv.org/pdf/2604.17837)
- [Oct 8, 2026; ICML 2026] **RouterInterp**: the "Superposed Specialisation Hypothesis" says experts specialize in a disjoint union of fine-grained features. The method finds the SAE features most predictive of routing and writes natural-language explanations of why tokens go to an expert. It reports about 65% higher detection accuracy on gpt-oss-20b than token-statistics methods. No code link found. — [Lasy, Cai, Ayonrinde, arXiv 2610.11775](https://arxiv.org/abs/2610.11775)
- [Jan 2026] SPAR report "Interpreting MoE Routing through Sparse Autoencoders" studies which SAE features drive routing and challenges the "one expert = one domain" view. — [SPAR Research Library](https://library.sparai.org/reports/interpreting-mixture-of-experts-routing-through-sparse-autoencoders-z8khhk/)
- [Oct 2025, rev. Dec 2025] "Sparsity and Superposition in MoE": higher network sparsity (active/total experts) means more monosemanticity. Specialization is better defined by monosemantic feature representation than by load. — [Chaudhari et al., arXiv 2510.23671](https://arxiv.org/abs/2510.23671)
- [Sep 2026] Routing states at different layers share a common geometry hidden by layer-specific coordinates, and routing across depth evolves by a roughly linear transition. — [arXiv 2609.02404](https://arxiv.org/html/2609.02404v1)
- [May 2026] "Routers Learn the Geometry of Their Experts" (title-level evidence of router–expert weight coupling). — [arXiv 2605.12476](https://arxiv.org/pdf/2605.12476)
- [Jul 2026] "Beyond Routing Weights" (MoE reward models) argues routing weights alone are not a faithful explanation and proposes contribution contrast. — [arXiv 2608.06400](https://arxiv.org/abs/2608.06400)

**Expert-level causal control, robustness and ablation**
- [Apr 2026] Geometric (cosine) routing in a low-dimensional metric space. Models are not named in the abstract and appear to be custom-trained.
  - About 15% of experts are monosemantic specialists in 10 categories.
  - Early layers route by word frequency, deeper layers by syntactic class (p < 0.001 with Zipf controls).
  - Interventions: steering toward a temporal expert centroid gives +321% P(temporal) (median, 44 prompts); suppressing a geographic expert gives −23%; rewriting an expert's output vector roughly halves the target-category probability. Effects add across layers.
  - Linear routers support comparable steering but are less transparent. — [Ternovtsii & Bilak, arXiv 2604.14434](https://arxiv.org/abs/2604.14434)
- [Aug 2026] "Output Dilution" (OLMoE-1B-7B vs. a matched dense model):
  - Moral valence is linearly decodable from nearly every expert-layer combination (above 90% peak accuracy).
  - It is **4.2× more fragile** to activation noise, because averaging over active experts shrinks the FFN signal reaching later layers by about two orders of magnitude.
  - Routing itself stays stable under noise. — [arXiv 2608.25231](https://arxiv.org/abs/2608.25231)
- [Jun 2026] Depth-aware magnitude-based expert masking on a model reported as "Qwen3.6-35B-A3B" (40 MoE layers × 256 experts, top-8). The model name is unverified by me.
  - Early layers (0–9) and middle layers (10–29) are fragile; late layers (30–39, especially 35–39) tolerate aggressive masking.
  - Masking 30% of experts in every layer kept only 150/300 "Good+Similar" outputs. Masking 50% of experts in layers 35–39 (640 of 10,240) kept 419/500.
  - Cutting top-8 to top-6 gave a speedup with no loss on a 100-prompt probe. — [arXiv 2608.13565](https://arxiv.org/abs/2608.13565)

**Intrinsically interpretable MoE architectures** (relevant only if retraining)
- [ICML 2025] MoE-X reports interpretability surpassing SAE-based approaches. — [mlanthology](https://mlanthology.org/icml/2025/yang2025icml-mixture/)
- [ICLR 2025] MONET ("Mixture of monosemantic experts for transformers"), seen only as a citation in [search results](https://arxiv.org/pdf/2604.02178).
- [May 2026] EMO pretrains MoEs for emergent modularity. — [arXiv 2605.06663](https://arxiv.org/pdf/2605.06663)

### Inferences
- **Protect super experts first.** Any expert-deactivation, pruning or "pinning" program must start by profiling SEs (in Qwen3-30B-A3B: L1 E68, L2 E92, L3 E82) and excluding them from intervention sets. Removing them reproduces the repetitive, degenerate failure mode a determinism program is trying to remove.
- **Expect off-target effects from expert-family interventions.** Given the Command A+ result and the "myth" paper, every expert-level control (SteerMoE-style) needs a pre-registered selectivity test against random-expert nulls on held-out corpora and several metrics before deployment.
- **Paths beat single experts as signatures.** Routing paths and router control-subspace clusters are likely more stable "behavior signatures" than individual experts. Logging full top-k paths per token is cheap and should be the core artifact.
- **Late layers are the low-risk intervention zone.** Late-layer experts tolerate masking (Qwen3.6-35B-A3B), while early layers hold super experts and are fragile.
- **Shared experts need separate analysis.** In DeepSeek and Qwen3.5-style models with shared experts, router edits cannot touch a shared expert, so shared-expert contributions need their own attribution.

### Gaps
- No public, peer-reviewed expert-level attribution-patching (EAP over experts) or full circuit study on a DeepSeek-V3-class or Qwen3-30B-A3B model was found.
- RICE's quantitative numbers and the number of experts reinforced were not visible in the abstract.
- The "Qwen3.6-35B-A3B" model name in arXiv 2608.13565 is unverified.
- No study was found of how SteerMoE-style edits affect run-to-run variance or self-consistency.

---

## Q3. Interventions for predictability: steering/RepE, single-direction controls, entropy/confidence neurons, attention sinks, tool-call and format features — and evidence that they reduce variance without hurting capability

### Takeaway
There is good evidence that single-direction and mean-difference steering can **change which discrete decision** a model makes:
- tool selection: 83–100% success on 4B+ dense models (2026);
- refusal: one direction (2024);
- safety and faithfulness: via expert (de)activation in MoEs (2025–26).

There is little direct evidence that these interventions **reduce sample-to-sample variance**:
- steering is "highly variable across inputs" (2024);
- it often loses to prompting (AxBench, 2025);
- models partly resist it mid-generation (ESR, 2026);
- it reduces output diversity, which is related to but not the same as self-consistency.

For MoE predictability, the most concrete levers are deterministic routing (routing replay, batch-invariant kernels) and protecting the mechanisms that, when broken, cause degenerate outputs (super experts, attention sinks, confidence-regulation neurons).

### Cited Findings

**Steering / representation engineering basics** [FOUNDATIONAL, 2023–2024]
- ActAdd adds activation differences from prompt pairs. — [Turner et al.](https://arxiv.org/abs/2308.10248)
- Representation Engineering reads and controls concepts from population-level directions. — [Zou et al., RepE](https://arxiv.org/abs/2310.01405)
- Contrastive Activation Addition (CAA) uses mean differences over many A/B behavior pairs, added at all positions. — [Panickssery/Rimsky et al., CAA](https://arxiv.org/abs/2312.06681)
- Refusal is mediated by a single residual-stream direction across many chat models. Ablating it removes refusal; adding it induces refusal. — [Arditi et al.](https://arxiv.org/abs/2406.11717)

**Reliability, variance and limits of steering**
- [Jul 2024, rev. May 2025] Steerability is "highly variable across different inputs" even in-distribution. Spurious biases drive per-input success. Vectors are brittle to reasonable prompt changes for several concepts. — [Tan et al.](https://arxiv.org/abs/2407.12404)
- [Jan 2025] AxBench: prompting beats DiffMean and SAE steering (0.894 vs. 0.239 vs. 0.165 in one version). — [AxBench](https://arxiv.org/abs/2501.17148)
- [May 2026] PrOSV: "it remains an open question whether SVs can reach the same level of variance as prompting." It reports standard deviations of steering scores on AxBench. — [Towards Steering without Sacrifice, arXiv 2605.05983](https://arxiv.org/html/2605.05983)
- [Feb 2026, rev. Jul 2026] **Endogenous Steering Resistance (ESR)**: models recover mid-generation from off-task steering ("wait, that's not right").
  - Llama-3.3-70B shows explicit ESR in 3.8% of cases; smaller Llama-3 and Gemma-2 models show it less.
  - 26 SAE latents were found; zero-ablating them cuts the multi-attempt rate by 25%.
  - Implication: steering-based control may be less dependable than it looks. — [McKenzie et al., arXiv 2602.06941](https://arxiv.org/abs/2602.06941)
- [Oct 2025; ICLR 2026] PID Steering treats standard steering as a P-controller and adds integral (persistent cross-layer correction) and derivative (overshoot damping) terms. It claims "more robust and reliable" control, with no numbers in the abstract. — [Nguyen et al., arXiv 2510.04309](https://arxiv.org/abs/2510.04309)
- [May 2026] Stronger steering "consistently reduces LLM's response diversity". That is direct evidence of output narrowing, though measured for synthetic-data generation rather than as a self-consistency target. — [arXiv 2605.28664](https://arxiv.org/html/2605.28664v1)
- [Jan 2025] LF-Steering targets semantic consistency across paraphrases, measured as the std of accuracy across paraphrased inputs, by steering latent features. — [arXiv 2501.11036](https://arxiv.org/pdf/2501.11036)
- [ICLR 2025] Instruction-following / format adherence: steering vectors computed from activation differences with and without an instruction improve adherence to format, length and word-inclusion constraints, and can be composed. — [Stolfo et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c3262a4c965ba9888f120d4f9e13478-Paper-Conference.pdf)
- [Apr 2025] Goodfire on DeepSeek-R1:
  - SAE steering only worked after the model started its response with "Okay, so the user has asked…" (more than 95% of R1's English traces begin with "Okay,").
  - Attention sinks sat near the end of this prefix.
  - Oversteering made R1 **revert to its original behavior** before outputs became incoherent.
  - An "Answer Quickly" feature (15072) shortened thinking in one example. — [Goodfire](https://www.goodfire.com/research/under-the-hood-of-a-reasoning-model)

**Tool-call and decision circuits**
- [May 2026] "Tool Calling is Linearly Readable and Steerable" (12 IT + 6 base dense models; Gemma 3 270M–27B, Qwen3/2.5 0.6B–14B, Llama 3.1 8B):
  - Method: mean-difference per-tool vectors added at the second-to-last layer (Gemma 3 4B L33, Qwen3 4B L35). Causal importance peaks at **89–98% depth**.
  - Tool-selection steering:
    - 100% (60/60) with 5 tools; random matched-norm vectors give 0%.
    - 15 tools: Qwen3 4B 93%, Gemma 3 4B 80%.
    - ≥90% up to 150 tools; τ-bench airline 77–94%.
  - The tool-name switch rewrites the JSON schema to match, but **argument-value steering failed (0/30)**.
  - On Gemma 3 27B, uncertain queries (small top-1/top-2 cosine gap) fail about **21× more often**, a usable internal uncertainty flag.
  - Multi-turn τ-bench deltas ranged from **−30 to +10 points** with no consistent direction.
  - The circuit is absent at 270M and emerges at about 1B. **No MoE models were tested.** — [Wu et al., arXiv 2605.07990](https://arxiv.org/html/2605.07990v2)
- [May 2026] "To Call or Not to Call" diagnoses intrinsic over-calling bias. Residual-stream SAE features that track call/no-call are validated with linear probes and tested causally with a counter-bias intervention. Abstract-level only. — [arXiv 2605.18882](https://arxiv.org/pdf/2605.18882)
- [May 2026] Tool-call dependency structure is linearly decodable from agent residual streams (title-level). — [arXiv 2605.25310](https://arxiv.org/pdf/2605.25310)

**Confidence, entropy and degenerate-output mechanisms**
- [NeurIPS 2024] **Entropy neurons** have large weight norms and write into an unembedding null space. They change the final-LayerNorm scale and so modulate logit entropy without directly changing logit rankings. They were found in models up to 7B and manage confidence during induction. **Token-frequency neurons** shift outputs toward or away from the unigram distribution. — [Stolfo et al., Confidence Regulation Neurons](https://arxiv.org/abs/2406.16254)
- [Mar 2025] Claude 3.5 Haiku has a default "can't answer" pathway inhibited by "known entity" features. Hallucinations come from misfires of that inhibition. The authors also note signs of "primitive metacognitive circuits" that track the extent of the model's own knowledge. — [Biology paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [May 2026] Qwen-Scope used SAE features operationally:
  - **Repetition suppression in RL**: repetition features were steered to create a negative rollout in each DAPO group. Repetition dropped faster and lower than vanilla RL. Qwen3-30B-A3B: MGSM 76.56→82.40, MMLU +0.48, IFEval −0.73.
  - **Code-switching**: ablating language features reduces it. SASFT cut code-switching by 31–100% across settings, with mixed small capability changes (e.g., Qwen3-8B MMLU −2.06, HumanEval +2.40).
  - The authors caution that repetition features also fire on benign repetition. — [Qwen-Scope](https://arxiv.org/html/2605.11887v1)
- [Jul 2025; ICLR 2026] Super experts drive attention sinks. Pruning them causes repetitive output until the length limit (see Q2). — [arXiv 2507.23279](https://arxiv.org/html/2507.23279v2)

**MoE routing as a source of nondeterminism (directly relevant to predictability)**
- [TransformerLens docs] Small numerical errors are "amplified greatly" when a different expert is chosen under top-k gating. Mixtral in half precision showed a logit absolute-error std of about 2e-3. The docs recommend `from_pretrained_no_processing` or higher precision. — [TransformerLens Special Cases](https://transformerlensorg.github.io/TransformerLens/content/special_cases.html)
- [Oct 2025; ICML 2026] R3 (Rollout Routing Replay):
  - Identical inputs can produce **different expert selections across repeated forward passes** (training vs. inference engines).
  - Recording rollout routing and replaying it in training cuts tokens with large train/infer probability discrepancies by about 10× and prevents RL collapse.
  - ms-swift implements R2 and R3 variants. — [Ma et al., arXiv 2510.11370](https://arxiv.org/abs/2510.11370); [ms-swift Router Replay docs](https://swift.readthedocs.io/en/v4.4/Instruction/GRPO/AdvancedResearch/router_replay.html)
- [Sep 2025] Inference nondeterminism mainly comes from kernels that are not batch-invariant, so results depend on server load and batch composition. Batch-invariant kernels restore bitwise reproducibility. — [Thinking Machines, Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

### Inferences
- **Discrete decisions vs. variance.** Interpretability interventions are best supported for moving discrete decisions: tool choice, refuse/comply, safe/unsafe, language. A "deterministic" program should use them to *fix* specific decision points, not expect them to lower general sampling variance. Variance reduction should come from decoding (greedy or constrained), batch-invariant kernels and routing replay or logging. Interventions should be checked by measuring agreement across k samples and across paraphrases.
- **Router margins as an uncertainty signal.** Two cheap internal signals come straight from the cited work: the top-1/top-2 cosine gap of tool-choice readouts (21× failure enrichment) and router-logit margins at the top-k boundary (where TL and R3 show flips happen). Using them to trigger fallback or abstention is an inference that has not been tested.
- **Where to intervene.** Tool-call decisions in dense models are steerable late (89–98% depth), and late-layer MoE experts tolerate masking. A late-layer residual or expert intervention is the most plausible low-collateral place to pin tool-call behavior in an MoE. This needs replication on an MoE.
- **Clamps to avoid.** Clamping or ablating mechanisms that regulate confidence (entropy neurons, super experts, sink-forming paths) is likely to *increase* degenerate variance such as repetition and collapse. These should be preserved or monitored, not edited.
- **Baselines first.** Given AxBench and ESR, every steering or expert edit should be benchmarked against prompt-only and constrained-decoding baselines before being adopted.

### Gaps
- I found no paper that directly measures sample-level self-consistency, pass^k or answer-agreement under steering vs. no steering, on dense or MoE models.
- No published circuit for format adherence or JSON validity in an MoE model was found.
- Tool-calling steering has been shown only on dense models; MoE replication is open.
- Entropy and confidence-regulation neurons have not been characterized in MoE experts (are they in shared experts? in super experts?).
- The capability cost of SteerMoE edits on standard benchmarks (MMLU etc.) is not reported.

---

## Q4. Tooling: TransformerLens, nnsight/NDIF, SAELens, circuit-tracer, Neuronpedia, pyvene, Inseq, Captum, Garcon, MoE-aware tools — support for Qwen MoE / DeepSeek MoE and compute cost

### Takeaway
By Oct 2026 the practical stack for a Qwen or DeepSeek MoE is **nnsight** (any HF model; NDIF remote execution up to 405B+ and DeepSeek-R1) or **TransformerLens 3.x TransformerBridge** (wraps native HF code and has `qwen3_moe`, `qwen2_moe` and `olmoe` adapters). Add **SAELens or sparsify** for dictionaries, **circuit-tracer or CLT-Forge** for graphs, and **Neuronpedia** for hosting.

MoE-specific functions are mostly in paper repos, not general libraries:
- super-expert profiler
- SteerMoE router-logit editing
- SAFEx expert identification
- MoE_analysis expert auto-interp
- RouterInterp (no code found)

Neuronpedia hosts gpt-oss-20b SAEs but I found no Qwen3-30B-A3B hosting.

### Cited Findings
- **TransformerLens**:
  - v3.0.0 released Apr 17, 2026; 3.3.0 on Jun 9, 2026. Docs also include a "Migrating to TransformerLens 4.0" page.
  - TransformerBridge is the recommended path. It keeps the HF implementation behind architecture adapters. PyPI cites 9,000+ models across 50+ families; the README cites 15,000+ across 140+ families.
  - The docs list adapter modules `olmoe`, `qwen3`, `qwen3_moe`, `qwen2_moe` and a generalized `moe` component. That support covers each specific model is my inference from module names. — [PyPI](https://pypi.org/project/transformer-lens/); [Migrating to v3](https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v3.html); [GitHub](https://github.com/TransformerLensOrg/TransformerLens); [Releases](https://github.com/TransformerLensOrg/TransformerLens/releases)
  - MoE numerical caveat (expert flips under top-k amplify errors). — [Special Cases](https://transformerlensorg.github.io/TransformerLens/content/special_cases.html)
- **nnsight / NDIF**:
  - The same intervention code runs locally or remotely (`remote=True`) on models up to 405B+.
  - NDIF hosts Llama 3.1 sizes and DeepSeek-R1 models; 405B access requires an application. NDIF is piloting model "hotswapping".
  - Remote code is restricted to approved packages (torch, numpy, einops, sympy, nnterp). — [nnsight Remote Execution](https://nnsight.net/features/13_remote_execution/); [Remote Access tutorial](https://nnsight.net/tutorials/tutorials/get_started/start_remote_access/); [NNsight & NDIF paper, 2024](https://arxiv.org/abs/2407.14561)
- **SAELens**:
  - Trains SAEs (v6 refactor). Inference works with any PyTorch model, with deep TransformerLens integration via `HookedSAETransformer`; HF/nnsight work via encode/decode.
  - Has a pretrained-SAE registry, a Neuronpedia tutorial and SAE-Vis dashboards. No MoE-specific features are mentioned. — [SAELens GitHub](https://github.com/jbloomAus/SAELens)
- **EleutherAI sparsify** trains TopK SAEs and transcoders on HF model activations at custom hookpoints. It computes activations on the fly, so there is no disk caching. — [sparsify overview](https://www.sourcepulse.org/projects/1832629); community wrapper [SAE-Transcoder](https://github.com/behroozazarkhalili/SAE-Transcoder)
- **circuit-tracer** (model list and backends as in Q1): the nnsight backend allows most HF models but is experimental; there is a GPT-OSS-20B CLT. — [GitHub](https://github.com/safety-research/circuit-tracer)
- **CLT-Forge** (Mar 2026): sharded distributed CLT training, compressed activation cache, circuit-tracer integration. — [arXiv 2603.21014](https://arxiv.org/abs/2603.21014)
- **Neuronpedia**:
  - Hosts dictionaries and graphs: Gemma-2-2B CLTs, Qwen3-4B graph steering, Qwen3-1.7B CRM/LORSA, gpt-oss-20b SAEs, Llama 3.3 70B Goodfire SAE.
  - Its homepage lists Qwen3-32B (dense), Qwen3.5 and Gemma 4 models.
  - No Qwen3-30B-A3B page was found. — [Fall update](https://www.neuronpedia.org/blog/fall-update); [gpt-oss-20b](https://www.neuronpedia.org/gpt-oss-20b); [Llama Scope 2](https://www.neuronpedia.org/llama-scope-2); [Neuronpedia home](https://neuronpedia.org)
- **Goodfire**:
  - Ember was launched (Dec 2024) as a hosted interpretability API (e.g., Llama 3.3 70B). — [Goodfire Ember](https://www.goodfire.ai/blog/announcing-goodfire-ember)
  - Open SAEs for Llama 3.1 8B and 3.3 70B. — [Goodfire SAE release](https://www.goodfire.ai/blog/sae-open-source-announcement)
  - DeepSeek-R1 SAEs. — [Goodfire R1](https://www.goodfire.com/research/under-the-hood-of-a-reasoning-model)
  - A secondary wiki says the public API was deprecated in Feb 2026 in favor of a partner platform (unverified). — [longtermwiki](https://www.longtermwiki.com/resources/6f8084939203873f)
- **pyvene** (2024): declarative intervention library (interchange interventions, trainable interventions, DAS). — [Wu et al., pyvene](https://arxiv.org/abs/2403.07809)
- **Inseq** (2023): feature-attribution toolkit for sequence generation. — [Inseq](https://arxiv.org/abs/2302.13942)
- **Captum**: PyTorch attribution library with integrated gradients and similar methods. — [captum.ai](https://captum.ai/)
- **Garcon** [FOUNDATIONAL, 2021] is Anthropic's internal infrastructure for probing large models, described publicly but not released. — [Garcon](https://transformer-circuits.pub/2021/garcon/index.html)
- **HeadVis** (May 2026) is Anthropic's interactive attention-head visualization tool (index listing). — [Transformer Circuits index](https://transformer-circuits.pub/)
- **MoE-specific repos**:
  - SteerMoE: router-logit (de)activation. — [Adobe Research page](https://research.adobe.com/publication/steering-moe-llms-via-expert-deactivation)
  - Super Experts profiler (released per paper). — [arXiv 2507.23279](https://arxiv.org/html/2507.23279v2)
  - SAFEx. — [GitHub](https://github.com/Bearisbug/SAFEx)
  - MoE_analysis (expert auto-interp). — [GitHub](https://github.com/jerryy33/MoE_analysis)
  - ms-swift Router Replay. — [docs](https://swift.readthedocs.io/en/v4.4/Instruction/GRPO/AdvancedResearch/router_replay.html)
- **Compute reference points**:
  - Gemma Scope 1: over 20% of GPT-3 training compute, about 20 PiB of activations, TPUv3/v5p. — [arXiv 2408.05147](https://arxiv.org/html/2408.05147)
  - Gemma Scope 2: about 110 PB and over 1T parameters (secondary). — [OpenSourceForU](https://www.opensourceforu.com/2025/12/google-deepmind-open-sources-gemma-scope-2/)
  - circuit-tracer on Gemma-2-2B: a 15GB Colab GPU. — [circuit-tracer](https://github.com/safety-research/circuit-tracer)
  - Goodfire needed custom infrastructure for 671B. — [Goodfire](https://www.goodfire.com/research/under-the-hood-of-a-reasoning-model)
  - Super-expert pruning experiments used Qwen3-30B-A3B and DeepSeek-R1; the masking study used three H100 servers for a 35B-A3B model. — [arXiv 2608.13565](https://arxiv.org/abs/2608.13565)

### Inferences
- **Recommended stack for a Qwen3-30B-A3B-class model:**
  - nnsight or TransformerBridge for hooks: residual stream, router logits, per-expert outputs, shared expert.
  - Qwen-Scope SAEs (SAELens-style encode/decode) for features.
  - sparsify or CLT-Forge to train transcoders or CLTs.
  - circuit-tracer (nnsight backend) for graphs.
  - Paper repos for MoE-specific steps: SE profiling, SteerMoE edits, SAFEx.
  - Budget engineering time for gaps, e.g., expert-as-node attribution, which no library offers.
- **Back-of-envelope sizes** (my arithmetic, not sourced):
  - Qwen3-30B-A3B in BF16 is about 61 GB of weights (≈30.5B × 2 bytes). Forward passes with hooks fit on one 80 GB GPU; gradient-based attribution (EAP-IG) over long contexts will likely need 2+ GPUs or offloading.
  - Residual width is 2048, implied by Qwen-Scope's W32K at 16× expansion. Caching all 48 residual layers costs about 196 KB per token in BF16, roughly 20 TB per 100M tokens. That argues for on-the-fly training (sparsify) or per-layer streaming.
  - Routing logs (48 layers × top-8 indices and weights) are about 1–2 KB per token, so cheap to keep for every evaluation run.

### Gaps
- No authoritative support matrix was found for TransformerBridge or circuit-tracer on Qwen3-30B-A3B or DeepSeek-V3. Support is inferred from adapter names and from nnsight's generic HF support.
- No published GPU-hour figures were found for training residual SAEs or CLTs on a 30B MoE. Qwen-Scope did not report compute.
- Tools from Transluce, Apollo Research and EleutherAI's newer releases were not researched because the search budget ran out.
- Whether Neuronpedia hosts Qwen-Scope MoE SAEs is unconfirmed.

---

## Q5. A practical, staged "discovery program" for a ~30B-class MoE (compute, order of operations, artifacts, success metrics) based on how labs have done it

### Takeaway
How labs have actually done this: Anthropic built a CLT replacement model, then attribution graphs plus interventions on targeted prompts. GDM and Qwen trained per-layer dictionaries for all layers, then ran downstream use cases. Goodfire trained SAEs at one or two layers on a 671B MoE, then steered. MoE-specific groups profiled experts and routing first, then did contrastive expert detection and router-logit steering with ablation controls.

A defensible program for a Qwen3-30B-A3B-class model is five stages:
1. Determinism and instrumentation baseline.
2. Routing and expert atlas, with super-expert profiling.
3. Feature dictionaries (reuse Qwen-Scope; train transcoders/CLTs where graphs are needed).
4. Behavior-specific circuit discovery on contrastive datasets.
5. Pre-registered interventions with selectivity and variance metrics.

Every claim about an expert, feature or circuit is accepted only on causal evidence measured against random nulls.

### Cited Findings (precedents that anchor each stage)
- **Stage 0 precedent: routing nondeterminism must be controlled before interpretation.**
  - Top-k flips amplify tiny numeric differences. — [TL Special Cases](https://transformerlensorg.github.io/TransformerLens/content/special_cases.html)
  - Identical inputs can route differently across engines; routing replay cuts large discrepancies about 10×. — [R3](https://arxiv.org/abs/2510.11370)
  - Batch-invariant kernels give bitwise reproducibility. — [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **Stage 1 precedents: routing atlas and super-expert profiling.**
  - Down_proj max-magnitude profiling finds SEs; Qwen3-30B-A3B has 3. — [Super Experts](https://arxiv.org/html/2507.23279v2)
  - The Command A+ study's first artifact was a routing-mass atlas. — [arXiv 2606.25092](https://arxiv.org/abs/2606.25092)
  - Per-expert early decoding. — [MoE Lens](https://arxiv.org/html/2603.05806v1)
  - Router control-subspace decomposition. — [arXiv 2604.17837](https://arxiv.org/abs/2604.17837)
  - Layer-depth sensitivity map. — [arXiv 2608.13565](https://arxiv.org/abs/2608.13565)
- **Stage 2 precedents: dictionaries.**
  - Qwen-Scope SAEs for all 48 layers of Qwen3-30B-A3B (W32K/k50 and W128K/k100). — [Qwen-Scope](https://arxiv.org/html/2605.11887v1)
  - Full-coverage dictionary suites. — [Gemma Scope](https://arxiv.org/html/2408.05147); [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-27b-it)
  - CLT quality metrics to target: replacement score, completeness, reconstruction error, L0. — [Anthropic methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
  - SAE features to explain routing. — [RouterInterp](https://arxiv.org/abs/2610.11775)
  - Chat-specialized dictionaries matter for chat behaviors. — [GDM negative results](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)
- **Stage 3 precedents: behavior circuits.**
  - Contrastive paired datasets with expert risk-difference detection. — [SteerMoE](https://arxiv.org/html/2509.09660v1)
  - Stability-based expert selection. — [SAFEx](https://arxiv.org/abs/2506.17368)
  - nPMI between experts and marker tokens. — [RICE](https://arxiv.org/abs/2505.14681)
  - Activation/attribution patching plus transcoders for tool selection. — [Tool calling](https://arxiv.org/html/2605.07990v2)
  - Attribution graphs with automatic description. — [circuit-tracer](https://github.com/safety-research/circuit-tracer); [ADAG](https://arxiv.org/abs/2604.07615)
  - Attention/QK attribution. — [Anthropic QK](https://transformer-circuits.pub/2025/attention-qk/index.html); [OpenMOSS CRM/LORSA](https://www.neuronpedia.org/llama-scope-2)
  - Turn-averaged SAEs for multi-turn agent transcripts. — [June 2026 update](https://transformer-circuits.pub/2026/june-update/index.html)
- **Stage 4 precedents: interventions and validation.**
  - Pre-registered selectivity: off-target ≤ 1/3 of on-target, size-matched random-expert nulls, several metrics and corpora, bootstrap CIs. — [arXiv 2606.25092](https://arxiv.org/abs/2606.25092)
  - Router-logit (de)activation. — [SteerMoE](https://arxiv.org/html/2509.09660v1)
  - Expert LoRA or negative merging to make edits permanent. — [SAFEx](https://arxiv.org/abs/2506.17368)
  - Late-layer mean-difference steering for tool selection, with random-vector controls. — [Tool calling](https://arxiv.org/html/2605.07990v2)
  - Feature-based RL shaping (repetition) as a training-time route to stable behavior. — [Qwen-Scope](https://arxiv.org/html/2605.11887v1)
  - Watch for steering resistance. — [ESR](https://arxiv.org/abs/2602.06941)
  - Compare against prompting. — [AxBench](https://arxiv.org/abs/2501.17148)

### Inferences — proposed program (synthesis; target model Qwen3-30B-A3B: 48 layers × 128 experts, top-8, 6,144 routed experts)

**Stage 0 — Reproducibility and instrumentation (about 1–2 weeks; 1–2 × 80GB GPUs)**
- Pin the inference stack with batch-invariant kernels and BF16/FP32 router computation.
- Build hooks for: residual stream at each layer, attention outputs per head, router logits (pre-top-k), selected indices and weights, per-expert outputs, and down_proj max magnitudes.
- Artifacts: a determinism harness (bitwise equality across batch sizes); a per-token routing log format; router-margin statistics (top-8 vs. 9th logit gap).
- Metrics: identical outputs across repeated runs and batch compositions; fraction of tokens whose top-k set changes under small numeric perturbations (the "routing flip rate").

**Stage 1 — Routing and expert atlas (about 2–3 weeks; a few GPU-days of forward passes over roughly 10–100M tokens)**
- Per-layer, per-expert routing frequency by domain, task and language, plus co-activation and path statistics.
- Super-expert profiling.
- MoE Lens / tuned-lens decoding per expert and per layer.
- Router control-subspace probes.
- Layer-depth masking sensitivity (expect early and middle layers fragile, late layers tolerant).
- Artifacts: expert cards (top tokens and contexts, autointerp labels, as in MoE_analysis); a list of SEs and other "never-touch" components; routing-path clusters.
- Metrics: auto-interp explanation scores; random-null comparisons; label stability across corpora.

**Stage 2 — Feature dictionaries (reuse first)**
- Load Qwen-Scope SAEs for all 48 layers.
- Where attribution graphs are needed, train transcoders or a CLT with CLT-Forge or sparsify on chat and agentic data. GDM found chat-specialized SAEs close about half the probe gap.
- Use RouterInterp-style analysis to map SAE features to routing decisions.
- Metrics: FVE/normalized MSE, L0, dead latents, downstream CE-loss delta when spliced in, CLT replacement and completeness scores (Anthropic reference: 0.61 / 0.80 on its 18L model).

**Stage 3 — Behavior circuits (per behavior: tool-call vs. no-call, tool choice, JSON/format validity, refusal/judgment, CoT step transitions, creative vs. factual mode)**
- Build contrastive datasets with matched pairs.
- Run EAP-IG over nodes {heads, experts, shared expert, router logits, SAE features}.
- Run expert risk-difference and nPMI detection.
- Build attribution graphs on representative prompts.
- Confirm with activation patching and path patching.
- Artifacts: per-behavior circuit cards listing components, layers and features; patching effect sizes; failure-mode notes (e.g., default-refusal inhibition, a tool-result-vs-prior conflict like the NLA "92 vs. 91" case).

**Stage 4 — Interventions for predictability (pre-registered)**
- Candidate levers:
  - router-logit pinning or deactivation of behavior experts (never SEs);
  - late-layer mean-difference vectors for discrete decisions;
  - SAE feature clamps for format and language features;
  - router-bias edits to widen margins at decision tokens (untested in the literature).
- Acceptance criteria:
  - on-target effect with CI;
  - off-target ≤ 1/3 of on-target;
  - beats or matches a prompt-only baseline;
  - no capability regression beyond a set budget (MMLU, GSM8K/MGSM, IFEval, BFCL/τ-bench);
  - reduced variance, measured as k-sample answer agreement, pass^k, paraphrase-consistency std, and multi-turn stability (the tool-calling paper showed multi-turn deltas from −30 to +10).
- Make accepted edits permanent through weight edits (expert LoRA or negative merging) or router-bias constants, so runtime hooks are not needed.

**Stage 5 — Regression and monitoring**
- Keep routing logs and the internal uncertainty signals (router margins, readout gaps) as production telemetry.
- Re-run the atlas and circuit cards after every fine-tune.
- Prior evidence says SE locations survive post-training, but behavior experts may not.

### Gaps
- No lab has published an end-to-end "full discovery" of a 30B-class MoE (every layer, expert, head and router, with circuits). The program above is a synthesis of partial precedents, not a replicated recipe.
- No published GPU-hour budgets exist for SAE, transcoder or CLT training or for attribution graphs on a 30B MoE. The stage durations and GPU counts above are my estimates.
- Router-bias "margin widening" and expert pinning **as variance-reduction tools** are untested in the literature found. So is whether interventions that fix decisions also reduce sampling variance.
- Reliability of expert-level selectivity varies by model: clean on Qwen3-30B-A3B per the Command A+ positive control, mostly not on Command A+. Each target model needs its own validation.

# MoE Architecture, Router Mechanics, Routing Stability/Determinism, and the Open-Weight MoE Landscape (as of 21 Oct 2026)

Notation used below:
- **[V]**: fetched and checked in this session (21 Oct 2026).
- **[K]**: a canonical paper or model card cited from prior knowledge. Its URL is the primary source, but it was not re-fetched in this session. The report writer may want a spot-check before quoting exact numbers.
- **[S]**: secondary or aggregator source only. Lower confidence.

The web-search budget ran out near the end of the research, so some landscape items (Mistral Large 3, NVIDIA Nemotron 3, Xiaomi MiMo-V2.x) could not be verified. They are listed under Gaps.

---

## Q1. Router designs in current MoE LLMs: what they are, and which add batch-dependence or nondeterminism at inference

### Takeaway
In every major open MoE LLM shipped from 2025 to 2026, the inference-time router is a **per-token, token-choice top-k**. In principle it is a pure function of that token's hidden state:
- Qwen3 and Qwen3.5–3.8, OLMoE, gpt-oss, Kimi and Mixtral use softmax top-k.
- DeepSeek-V3 used sigmoid scores, and DeepSeek-V4 uses sqrt(softplus) scores, both with aux-loss-free bias.
- DeepSeek-V4 uses hash routing in its first layers.

The designs that are **batch-dependent by construction** are:
- expert-choice routing
- capacity-factor token dropping
- BASE-style balanced assignment
- noisy gating

Modern decoder LLMs avoid all of these at inference. Load-balancing machinery (aux loss, global-batch loss, bias updates) acts only during training. In practice, inference nondeterminism comes from **numerics feeding a discrete argmax** (Q2) and from **kernels whose reduction order depends on batch composition** (Q5), not from the router's definition.

### Cited Findings
**Classic router designs**
- **Noisy top-k gating** (Shazeer 2017) adds tunable Gaussian noise to gate logits before top-k during training, plus importance and load losses. This is stochastic routing in training. [K] — [arXiv 1701.06538](https://arxiv.org/abs/1701.06538)
- **GShard** uses top-2 gating, a per-expert **capacity**, and local group dispatch. Tokens over capacity are dropped (passed through the residual). The second expert is dispatched stochastically in proportion to its gate weight. Both make one token's computation depend on the other tokens in its group. [K] — [arXiv 2006.16668](https://arxiv.org/abs/2006.16668)
- **Switch Transformer** uses top-1 routing with a capacity factor. Overflow tokens skip the layer via the residual.
  - It introduced "selective precision": the router computation is cast to float32 locally while the rest runs in bf16, for stability.
  - Aux load-balance loss coefficient α = 1e-2.
  - [K] — [arXiv 2101.03961](https://arxiv.org/abs/2101.03961)
- **ST-MoE** introduced **router z-loss**, which penalizes large logits entering the gating softmax (coefficient ~1e-3). It improves training stability without hurting quality. The paper attributes instability partly to round-off in exponentiated large logits. [K] — [arXiv 2202.08906](https://arxiv.org/abs/2202.08906)
- **BASE layers** solve a balanced linear-assignment problem across the batch during training, which is inherently batch-coupled. At test time each token goes greedily to its highest-scoring expert. [K] — [arXiv 2103.16716](https://arxiv.org/abs/2103.16716)
- **Hash layers** route by a fixed hash of the token ID. There are no learned router parameters, so routing is fully deterministic and context-free, and was reported as competitive with Switch and BASE. [K] — [arXiv 2106.04426](https://arxiv.org/abs/2106.04426)
- **Expert-choice routing**: each expert picks its top-k tokens from the batch. This gives perfect load balance and a variable number of experts per token. A token's routing depends on which other tokens are in the batch, and the scheme is not causal for autoregressive decoding. [K] — [arXiv 2202.09368](https://arxiv.org/abs/2202.09368)
- **OLMoE ablation of expert choice vs. token choice** [V]:
  - Token choice outperformed expert choice at equal token budget.
  - Expert choice was ~20% faster (29.4k vs. 24.4k tokens/s per device) but can drop tokens.
  - OLMoE chose **dropless token choice**.
  - [arXiv 2409.02060v2](https://arxiv.org/html/2409.02060v2)

**Fine-grained and shared experts**
- **DeepSeekMoE** combines fine-grained expert segmentation (split N experts into mN smaller ones and activate mK) with shared-expert isolation (Ks always-on experts). [K] — [arXiv 2401.06066](https://arxiv.org/abs/2401.06066)
- **OLMoE's shared-expert ablation** [V]:
  - 1 shared + 3 routed experts performed slightly worse than 4 routed.
  - Converting a routed expert into a shared one cuts possible expert combinations by ~90% (4,495 vs. 35,960 per layer), so OLMoE uses no shared experts.
  - [arXiv 2409.02060v2](https://arxiv.org/html/2409.02060v2)
- **Qwen3 MoE** [V]:
  - "Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts."
  - Qwen3-30B-A3B has 48 layers, 128 experts and 8 active.
  - Qwen3-235B-A22B has 94 layers, 128 experts and 8 active.
  - Qwen3 "adopt[s] the global-batch load balancing loss" to "encourage expert specialization."
  - [Qwen3 Technical Report, arXiv 2505.09388](https://arxiv.org/html/2505.09388)
- Qwen re-introduced **one shared expert** in the Qwen3.5/3.6/3.8 generation:
  - Qwen3.5-397B-A17B: 512 experts, 10 routed + 1 shared. [V] — [HF card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
  - Qwen3.6-35B-A3B: 256 experts, 8 routed + 1 shared. [V] — [HF card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
  - Qwen3.8-2.4T-A95B: 512 experts, 10 routed + 1 shared. [V] — [HF card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)

**Load-balancing schemes (all training-only)**
- **Global-batch vs. micro-batch balancing** ("Demons in the Detail", Jan 2025) [V]:
  - A micro-batch load-balancing loss is "almost at the sequence level". It forces every sequence, including domain-specific ones such as code, to spread across all experts, which hinders specialization.
  - Computing the loss over the global batch (one extra sync of expert-selection frequencies) improved perplexity and downstream results at up to 42.8B parameters and 400B tokens.
  - It also "greatly improves domain specialization."
  - [arXiv 2501.11873](https://arxiv.org/abs/2501.11873)
- **Auxiliary-loss-free balancing**: a per-expert bias is added to affinity scores **only for top-k selection**, not for the gate weights, and is updated by the sign of each expert's load violation. [K] — [arXiv 2408.15664](https://arxiv.org/abs/2408.15664)
- **DeepSeek-V3 routing** [K] — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437):
  - 671B total / 37B active, 1 shared + 256 routed experts, top-8.
  - Sigmoid affinities normalized over the selected experts.
  - Bias update speed γ = 0.001 and sequence-wise balance loss α = 0.0001.
  - Node-limited routing (≤ 4 nodes per token).
  - "No token-dropping" in training or inference.
- **DeepSeek-V4** (paper submitted 26 Apr 2026) [V] — [arXiv 2606.19348 HTML](https://arxiv.org/html/2606.19348v1):
  - Affinity changes from Sigmoid to **Sqrt(Softplus(·))**.
  - Aux-loss-free balancing with bias update speed 0.001 plus a "slight" sequence-wise balance loss of weight 0.0001.
  - **Node-limited routing removed.**
  - **The first 3 MoE layers use hash routing**: target experts come from a predefined hash of the input token ID.
- **DeepSeek-V4 expert configuration** (§4.2.1) [V]:
  - V4-Flash: 43 layers, 1 shared + 256 routed experts (intermediate size 2048), top-6, 284B total / 13B active.
  - V4-Pro: 61 layers, 1 shared + 384 routed experts (intermediate size 3072), top-6, 1.6T total / 49B active.
  - [arXiv 2606.19348](https://arxiv.org/html/2606.19348v1)
- **DeepSeek-V4.1-Flash config** (from a FlashInfer issue) [S/V]:
  - `scoring_func = sqrtsoftplus`, `topk_method = noaux_tc`, `n_routed_experts = 384`, `num_experts_per_tok = 6`, `routed_scaling_factor = 1.5`.
  - FlashInfer's fixed fused-MoE routing enums cannot express sqrt-softplus scoring, so V4.1 cannot use that fused path.
  - [flashinfer #5190](https://github.com/flashinfer-ai/flashinfer/issues/5190)
  - The HF Transformers DeepSeek-V4 docs describe "hash_moe" layers that take expert indices from a frozen table keyed by token ID, while the learned gate still produces per-expert weights. — [HF docs](https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4) (per search summary)
- **Kimi K3** uses a "Stable LatentMoE framework" (latent MoE dimension 3584):
  - 896 routed experts, 16 per token, plus 2 shared.
  - The model card does not say whether gating is sigmoid or aux-loss-free.
  - [V] — [HF card](https://huggingface.co/moonshotai/Kimi-K3)

**Adaptive-compute routing**
- **Dynamic-k / top-p routing** activates experts until cumulative router probability ≥ p, so harder tokens get more experts. [K] — [arXiv 2403.07652](https://arxiv.org/abs/2403.07652); overview: [HF blog "Your MoE Model Does Not Have to Select Fixed Number of Experts"](https://huggingface.co/blog/Spico/dynamic-routing)
- **Null / zero-computation experts**:
  - MoE++ adds zero, copy and constant experts. [K] — [arXiv 2410.07348](https://arxiv.org/abs/2410.07348)
  - AdaMoE uses null experts to vary the token-wise number of experts. [K] — [arXiv 2406.13233](https://arxiv.org/abs/2406.13233)
  - Both are per-token deterministic functions, but compute per token varies.

### Inferences
- **Batch-dependent by definition at inference:** expert-choice routing, capacity-factor token dropping (GShard/Switch-style with finite capacity), BASE balanced assignment (training only), noisy or stochastic second-expert routing (training only).
- **Per-token deterministic by definition:**
  - token-choice top-k (softmax or sigmoid)
  - aux-loss-free bias, which is a frozen constant at inference
  - hash routing
  - shared experts
  - dynamic-k / top-p and null experts, which are deterministic per token but make kernel shapes data-dependent
  - global-batch or micro-batch balancing losses, which are training-only
- **Hash routing** (V4's first 3 layers) is the only router in a frontier model that is immune to numerical flips. For a "deterministic MoE", extending hash or frozen-table routing deeper, or using a task-ID → expert-set table, is the purest form of determinism, at a quality cost that is unquantified at scale.
- **Shared experts** add an always-on path that does not depend on routing. This plausibly dampens the downstream impact of a routed-expert flip, but I found no measurement of this.
- **Removing node-limited routing in V4** removes one source of routing that depends on deployment topology, which is good for determinism. DeepSeek-V3's group-limited top-k is still a per-token function, though.

### Gaps
- I found no published MoE LLM that uses expert-choice or capacity-dropping at inference in 2026. I am inferring that dropless token choice is universal from the models checked: Qwen3/3.5/3.6/3.8, DeepSeek-V3/V4, OLMoE and Kimi K3.
- GLM-5.x and MiniMax-M3 model cards do not state gating function, expert count or top-k (see Q4).
- DeepSeek-V4's exact hash function and the content of "Stable LatentMoE" (Kimi K3) were not retrieved.

---

## Q2. Routing fragility: flip rates under perturbation, router margins, and train–inference routing mismatch in RL (plus fixes)

### Takeaway
Top-k routing is a discrete argmax over continuous logits, and it **does flip under tiny numerical changes**. The best numbers measure engine-vs-engine flips in Qwen3-30B-A3B (SGLang rollout vs. Megatron training):
- ~10% of router decisions differ
- 94% of tokens differ in at least one layer
- ~6 routers per token differ on average

RL weight updates flip ~10% of activated experts per gradient step. This drives MoE-specific RL collapse.

Fixes:
- **routing replay** (R3 / GSPO-era Routing Replay)
- **keeping the router out of low precision** (BF16/FP32 router under FP8 rollout)
- **anticipatory routing** (DeepSeek-V4, for pretraining spikes)
- **bitwise-aligned training and inference kernels**

There are very few measurements of routing consistency under paraphrase, and I found no explicit "router margin" distributions for production LLMs.

### Cited Findings
**Engine mismatch and run-to-run noise (R3 paper, PKU + Xiaomi, arXiv 2510.11370, ICML 2026)** [V] — [arXiv 2510.11370v2](https://arxiv.org/html/2510.11370v2)
- **Setup:** 2,048 math prompts, generated with SGLang and replayed through Megatron on Qwen3-30B-A3B.
  - "Approximately 10% of the routers select different experts during training compared to inference."
  - **94% of tokens** select a different expert in at least one layer.
  - The mean difference is ~**6 routers per token**.
- **Policy KL (train vs. inference):**
  - MoE: 1.535×10⁻³
  - dense Qwen3-8B: 6.4×10⁻⁴
  - MoE + R3: 7.5×10⁻⁴
- **Run-to-run noise in one engine:** two Megatron forward passes on identical sequences gave **KL = 8.4×10⁻⁴**. The paper notes that even under identical conditions, repeated forward passes can produce divergent expert selections.
- **Extreme tokens:** with τ > 2, the MoE model has ~10× more extreme-ratio tokens than the dense model, and R3 cuts this by ~10×. In GRPO with a single mini-step, F(τ=2) exceeded 0.1 after 60 steps without R3 and stayed < 10⁻⁴ with R3.
- **How R3 works:**
  - Record top-k masks in the inference engine during rollout.
  - Replay them in Megatron for both the old-policy recompute and the policy update.
  - Recompute gate weights as a softmax over training logits restricted to the replayed experts, so the router still gets gradients.
  - Cache masks alongside the KV prefix cache, which supports multi-turn agents.
  - Rollout latency overhead is < 3%.
- **Results:**
  - SFT model, single mini-step: GRPO 62.23 (crashed at step 60) vs. GRPO+R3 71.83.
  - Multi mini-step: GSPO 66.76 vs. GRPO+R3 68.05 vs. GSPO+R3 69.00.
  - SWE-bench Verified multi-turn RL: GRPO 31.80 (collapsed ~step 90) vs. GRPO+R3 38.60.
  - Adding TIS on top of R3 gave no clear gain.
- **Limitations:** only one MoE model was studied, and no auxiliary balancing loss was used.
- **Follow-up ("Stale but Stable", arXiv 2607.18722):** R3 does not reproduce the whole rollout forward pass (hidden states, logits and expert weights can still differ), but it removes one important mismatch source. — [arXiv 2607.18722](https://arxiv.org/pdf/2607.18722) (per search summary)

**Expert volatility across RL updates (GSPO, Qwen team, arXiv 2507.18071)** [V] — [arXiv 2507.18071](https://arxiv.org/html/2507.18071)
- In the 48-layer Qwen3-30B-A3B-Base, "after each RL gradient update, about 10% of the experts activated for the same rollout sample under the new policy differ" from the old policy. The effect is stronger in deeper MoEs.
- **Routing Replay** caches old-policy expert choices and reuses them when computing the new-policy likelihood. It costs memory and communication and "can limit the MoE model's effective capacity."
- GSPO's sequence-level importance ratio is insensitive to individual token likelihoods, so it trains stably **without** Routing Replay.

**Precision effects (NVIDIA FP8-RL, arXiv 2601.18150, Qwen3-30B-A3B-Base)** [V] — [arXiv 2601.18150](https://arxiv.org/pdf/2601.18150)
- "MoE routing decisions are particularly sensitive to numerical precision, as small perturbations in router logits can alter expert selection and cascade into divergent model behavior."
- An FP8-quantized router gave the highest rollout/train mismatch KL (~0.004 early). A BF16 router or an FP32 router (upcast on both vLLM and Megatron sides) were similar and lower. "Excluding the router from FP8 quantization is sufficient… upgrading further from BF16 to FP32 provides marginal additional benefit."
- In MoE, mismatch KL **grows over training** even in BF16, unlike the dense model. Routing inconsistencies accumulate.
- An FP8-rollout-only configuration (BF16 training) with only token-level TIS **crashed**. The authors say stronger fixes (R3 or Masked Importance Sampling) may be needed.
- Their production recipe keeps the MoE router in BF16 on both sides.
- RL framework note: **slime** (SGLang-native) "specifically optimiz[es] for routing consistency in MoE models."

**Post-training quantization flips (EAQuant, arXiv 2506.13329)** [V, partial] — [arXiv 2506.13329](https://arxiv.org/abs/2506.13329)
- Quantization noise in router outputs significantly changes top-k assignments.
- The method adds "routing consistency alignment to preserve expert selection post-quantization", giving +1.15 to +13.81% average accuracy over baselines at W4A4/W3A4/W3A3/W2A4 across three MoE architectures.
- The abstract gives no flip-rate percentage.

**Counterfactual routing, i.e. how much the chosen route matters (arXiv 2605.07260, 8 May 2026)** [V] — [arXiv 2605.07260](https://arxiv.org/html/2605.07260)
- Tokens are binned by mean probability of the realized next token across 32 sampled alternative routes.
- **"Fragile" tokens (p̄ ≤ 0.5) at the final MoE layer:**

  | Model / set | Fragile tokens |
  |---|---|
  | Qwen3-30B-A3B, MATH-L5 | 6.9% |
  | Qwen3-30B-A3B, AIME24+25 | 10.5% |
  | Qwen3-30B-A3B, HMMT25 | 9.2% |
  | Qwen3-30B-A3B, GPQA-D | 5.5% |
  | GPT-OSS-20B | 12.8% |
  | DeepSeek-V2-Lite | 11.9% |
  | OLMoE-1B-7B | 14.8% |

- On fragile tokens the standard route is the best of 33 candidates only 0.8% of the time (Qwen3 L47; 51.9% on confident tokens). The best alternative route raises next-token probability by **~20 pp** (40.2% → 60.6%).
- The gap grows with depth: 11.4 pp (L0) → 12.8 (L24) → 20.4 (L47).
- A router-only DPO-style fix (EPO, ~0.001% of parameters) changed little. The authors call it an existence check.
- The paper reports **no router-margin distributions**.

**Training-time routing fluctuation**
- StableMoE (ACL 2022) defines "routing fluctuation": the target expert of the same input changes during training.
- Its fix:
  1. learn a balanced, cohesive routing and distill it into a lightweight router decoupled from the backbone;
  2. freeze that router.
- It reports faster convergence and better LM/MT results.
- [V] — [arXiv 2204.08396](https://arxiv.org/abs/2204.08396)
- OLMoE "router saturation" [V] — [arXiv 2409.02060v2](https://arxiv.org/html/2409.02060v2):
  - After 1% of pretraining (~20B tokens), up to ~60% of top-8 routing already matches the final checkpoint, rising to ~80% at 40%.
  - Top-1 saturates more slowly. Later layers saturate earlier, and layer 0 is an outlier.
  - Random baseline: 12.5% (k=8) and 1.6% (k=1).
- **DeepSeek-V4 "Anticipatory Routing"** (§4.2.3) [V] — [arXiv 2606.19348](https://arxiv.org/html/2606.19348v1):
  - At step t, features use current θ_t, but routing indices are computed with **historical θ_{t−Δt}**, precomputed and cached.
  - It is auto-triggered after a loss spike plus rollback, then reverts. Overhead is ≤ ~20% wall-clock.
  - The authors tie spikes to MoE outliers and say "routing appears to worsen them".
  - V4 also clamps SwiGLU (linear part to [−10, 10], gate capped at 10).
- **Kimi K3** advertises a "Stable LatentMoE framework" without details. [V] — [HF card](https://huggingface.co/moonshotai/Kimi-K3)

**Routing consistency across context**
- MCF-MoE (arXiv 2607.16427, v2 27 Sep 2026) argues that routers conditioned on "shallow or isolated token representations" give routing that is "unstable and semantically inconsistent across layers". Adding multi-level context fusion "consistently improves routing consistency". The abstract gives no numbers. [V] — [arXiv 2607.16427](https://arxiv.org/abs/2607.16427)
- A vision-MoE paper found baseline routing still differs from final assignments for ~40% of inputs at mid-training. This concerns training dynamics only. — [arXiv 2604.21330](https://arxiv.org/pdf/2604.21330) (per search summary) [S]

### Inferences
- **Order of magnitude to plan for:** with ~48 layers × top-8 and BF16 kernels that differ between two stacks, nearly every token (94%) will have at least one routing difference somewhere. Dense-model logprob noise is about 2.4× smaller (1.5e-3 vs. 6.4e-4 KL). For a compiler-like system, cross-stack bitwise reproducibility of an MoE is unattainable unless the router input is bitwise identical, or routing decisions are logged and replayed.
- **Practical determinism levers, from cheapest to strongest:**
  1. Run router logits in FP32 (or at least BF16, never FP8/INT4).
  2. Use a deterministic tie-break (lowest expert ID) in top-k.
  3. Use batch-invariant kernels end to end.
  4. Log top-k masks per token and replay them, as R3 does. This turns routing into a recorded "compilation artifact".
  5. Freeze or distill the router (StableMoE) or hash-route early layers (DeepSeek-V4).
- **Fragile tokens are exactly where routing matters:**
  - Routing flips concentrate on low-margin tokens.
  - The counterfactual paper shows 5–15% of tokens are route-fragile, with ~20 pp next-token probability swings.
  - In these places a determinism failure becomes a behavior change, not just noise.
- **Reasoning-heavy evaluations amplify the effect.** AIME and HMMT show more fragile tokens (9–10.5%) than GPQA (5.5%), so long chain-of-thought and deep-reasoning use cases are most exposed.

### Gaps
- I found **no published per-layer router-margin (top-k vs. (k+1) logit gap) distribution** for any 2025–2026 production MoE. I also found no direct "flip rate under bf16-vs-fp32 at fixed weights" table; the FP8-RL paper reports KL only, not flip counts.
- I found **no study measuring routing consistency across paraphrased prompts** for a production LLM. OLMoE and ST-MoE report token- and vocabulary-level specialization, not paraphrase invariance.
- The R3 figures cover one model and one engine pair (SGLang vs. Megatron). I found no comparable numbers for DeepSeek-V4, Kimi K3 or Qwen3.5+.

---

## Q3. Techniques to specialize or control routing per task or domain

### Takeaway
There are four families of task control, plus an analysis base:
1. **Build-time domain experts:** BTM/BTX, sparse upcycling.
2. **Post-hoc task-specific expert selection:** ESFT, pruning with REAP or router-sensitivity, expert skipping.
3. **Router-only or frozen-router tuning:** StableMoE freeze, router-only LoRA, EPO.
4. **Adapter-level MoE:** MoLE, X-LoRA, LoraHub. These keep a frozen base and add per-task LoRAs that can be selected deterministically by task ID.

Analysis finding: experts in fine-grained, natively trained MoEs (OLMoE, DeepSeek-style, Qwen3 with global-batch balancing) are domain- and vocabulary-specialized. Upcycled models (Mixtral) are much less so. Fine-grained, from-scratch MoEs are therefore better substrates for task-routed determinism.

### Cited Findings
**Specialization evidence**
- OLMoE [V] — [arXiv 2409.02060v2](https://arxiv.org/html/2409.02060v2):
  - Some experts are highly domain-specialized. A layer-0 expert is ~100% arXiv, and GitHub and arXiv co-activate in layer 7. Generic C4 text is routed much more evenly.
  - Experts show vocabulary/theme specialization (weekdays and sports events, country/nationality names, religious terms).
  - Experts rarely co-activate within a layer, which suggests low redundancy.
  - **Mixtral-8x7B shows little domain specialization** (near uniform). The authors attribute this to Mixtral being **upcycled** from a dense model.
- Global-batch load balancing (adopted by Qwen3) "greatly improves domain specialization" vs. micro-batch LBL. [V] — [arXiv 2501.11873](https://arxiv.org/abs/2501.11873); [Qwen3 TR](https://arxiv.org/html/2505.09388)
- **ESFT** (DeepSeek authors, Jul 2024) [V] — [arXiv 2407.01906](https://arxiv.org/abs/2407.01906):
  - "The routing distribution for a specific task tends to be highly concentrated", while activated experts "vary significantly across tasks".
  - Finer-grained experts make the task-relevant subset easier to isolate.
  - ESFT tunes only the task-relevant experts and freezes everything else. It "matches or even surpasses full-parameter fine-tuning" with better efficiency.
  - Code is at github.com/deepseek-ai.

**Build-time domain routing**
- **Branch-Train-Merge (BTM)** trains independent domain expert LMs (ELMs) branched from a seed and ensembles or averages them. [K] — [arXiv 2208.03306](https://arxiv.org/abs/2208.03306)
- **Branch-Train-MiX (BTX)** [K] — [arXiv 2403.07816](https://arxiv.org/abs/2403.07816):
  - Branch a seed model (Llama-2-7B in the paper) and train domain experts (math, code, Wikipedia) in parallel.
  - Merge their FFNs as experts in MoE layers and average the remaining parameters.
  - Then do an MoE fine-tuning stage that learns token-level routing.
- **Sparse upcycling** initializes an MoE from a dense checkpoint by copying FFNs into experts. [K] — [arXiv 2212.05055](https://arxiv.org/abs/2212.05055)
  - Per OLMoE's analysis above, upcycled MoEs (Mixtral) end up with weak domain specialization. [V] — [arXiv 2409.02060v2](https://arxiv.org/html/2409.02060v2)

**Expert pruning, merging and skipping per domain**
- **REAP** (v1 Oct 2025, v3 May 2026) [V] — [arXiv 2510.13999](https://arxiv.org/abs/2510.13999):
  - Router-weighted Expert Activation Pruning scores experts by gate value × activation norm.
  - **Pruning beats merging on generative tasks.** Merging "loses fine-grained routing control" and introduces irreducible error.
  - 50% expert pruning of Qwen3-Coder-480B and Kimi-K2 is near-lossless on code generation.
- **Router sensitivity under lightweight fine-tuning** (arXiv 2608.07890, v3 10 Oct 2026, NeurIPS 2026 workshop) [V] — [arXiv 2608.07890](https://arxiv.org/abs/2608.07890):
  - Fine-tune with a small adapter and prune the experts whose router weights change least.
  - Router-only LoRA (~0.002% of parameters) on Mixtral-8x7B with half the experts removed keeps 27.54% MMLU-Pro vs. ~16% for magnitude or random pruning.
  - Qwen1.5-MoE with half the experts removed keeps 49.7% mean accuracy on 11 math benchmarks.
  - Memory −49%, per-token latency −37%.
- **Task-specific expert pruning and dynamic skipping on Mixtral** ("Not All Experts are Equal"). [K] — [arXiv 2402.14800](https://arxiv.org/abs/2402.14800)

**Router-only control and freezing**
- StableMoE freezes a distilled router for stable routing. [V] — [arXiv 2204.08396](https://arxiv.org/abs/2204.08396)
- EPO updates only the final-layer router (~0.001% of parameters) toward better counterfactual routes, with small effects. [V] — [arXiv 2605.07260](https://arxiv.org/html/2605.07260)

**Mixture-of-LoRA (per-task adapters on a frozen base)**
- **MoLE** (ICLR 2024) learns a gating function that composes multiple trained LoRAs per layer. [K] — [arXiv 2404.13628](https://arxiv.org/abs/2404.13628)
- **X-LoRA** uses a scaling head on hidden states to mix LoRA adapters dynamically per token and per layer. [K] — [arXiv 2402.07148](https://arxiv.org/abs/2402.07148)
- **LoraHub** does gradient-free composition of LoRA modules, with weights fitted from a few examples of an unseen task. The result is fixed per task once fitted. [K] — [arXiv 2307.13269](https://arxiv.org/abs/2307.13269)

### Inferences
- **Most practical "deterministic MoE" recipe** suggested by this evidence:
  1. Start from a fine-grained, natively trained MoE with strong expert specialization: Qwen3.x with global-batch balancing, or DeepSeek-style.
  2. Profile per-task expert activation (ESFT-style) for each use case: decision-making, function calling, cross-evaluation, creative writing, deep reasoning.
  3. Derive a **task-conditioned expert allow-list or mask per layer**. This is like REAP or router-sensitivity pruning, but applied as a mask rather than a deletion.
  4. Optionally freeze the router (StableMoE) and fine-tune only the selected experts (ESFT).
  5. Add per-task LoRA adapters chosen by **explicit task ID**, LoraHub-style fixed weights, rather than a learned soft mixer like X-LoRA. Learned soft mixers reintroduce data-dependent gating.
- **Task-ID-keyed routing** (a hash or lookup table, as in DeepSeek-V4's hash layers) is the logical end state for "compiler-like" determinism. No published 2025–2026 LLM does task-keyed hard routing at scale, so quality risk is unknown.
- **Avoid merging experts.** REAP's result that merging destroys routing control suggests pruning or masking is preferable when shrinking for a domain.

### Gaps
- I found no 2026 head-to-head of BTX vs. ESFT vs. mixture-of-LoRA on determinism or consistency metrics. All comparisons are on task accuracy.
- The OLMoE vocabulary-specialization numbers were truncated in the fetch; only qualitative examples were captured.
- I found no study of whether **task-masked routing** (restricting top-k to an allow-list) degrades quality on a 2026 model. This would be the key experiment for the project.

---

## Q4. Best open-weight MoE base candidates as of Oct 2026, including verification of "Qwen 3.8 27B" and small CPU-scale MoEs

### Takeaway
**"Qwen3.8-27B" is real** (HF weights from August 2026, Apache-2.0), but it is a **dense** 64-layer hybrid (Gated DeltaNet + Gated Attention) vision-language model, **not an MoE**. [V]

The closest real Qwen MoEs:
- **Qwen3.6-35B-A3B** (Apr 2026, Apache-2.0, 256 experts, 8+1 active). This is the best permissive small MoE.
- **Qwen3.5-122B-A10B and Qwen3.5-397B-A17B** (Feb 2026, Apache-2.0).
- **Qwen3.8-Flash-Next** (Aug 2026, 125B-A6B + 51B n-gram embedding, qwen-community-1.0 license).
- **Qwen3.8-2.4T-A95B** (Aug 2026, custom "qwen3.8-max" license).

At the frontier, open-weight MoEs in Oct 2026:
- DeepSeek-V4-Pro/Flash (MIT)
- Kimi K3 (2.8T-A104B, custom license)
- GLM-5.3 (753B, custom license)
- MiniMax-M3 (428B-A23B, community license)

For a determinism research program, the most practical bases are:
- **Qwen3-30B-A3B**: the most-studied model for routing mismatch and batch invariance.
- **Qwen3.6-35B-A3B**: newest Apache small MoE.
- **gpt-oss-20b**: Apache, batch-invariance tested in vLLM.
- **DeepSeek-V4-Flash**: MIT, ships with batch-invariant/deterministic kernel work and hash-routed layers.
- **OLMoE-1B-7B, granite-3.1-1b-a400m, DeepSeek-V2-Lite**: for CPU-scale experiments.

### Cited Findings
**"Qwen 3.8 27B" verification**
- **Qwen3.8-27B** [V] — [HF card](https://huggingface.co/Qwen/Qwen3.8-27B):
  - Dense: "a compact, deployment-friendly dense model".
  - 27B (HF lists 28B), 64 layers, hidden size 5120, layout 16 × (3 × Gated DeltaNet→FFN + 1 × Gated Attention→FFN), FFN 17,408.
  - 262,144 native context, ~1M with YaRN. Apache-2.0. Citation dated Aug 2026.
  - Benchmarks: Terminal-Bench 2.1 73.0, SWE-bench Pro 61.7, GPQA-D 89.2, HLE 30.8, LiveCodeBench v6 90.3.
  - It compares itself against Qwen3.6-27B and Qwen3.7-Plus, which implies a dense Qwen3.6-27B exists and that Qwen3.7-Plus is a hosted model.
  - Corroborated by [LM Studio](https://lmstudio.ai/models/qwen3.8) (dense, thinking on by default, effort levels), [vLLM recipes](https://recipes.vllm.ai/Qwen/Qwen3.8-27B), [OpenRouter](https://openrouter.ai/qwen/qwen3.8-27b) and [Cloudflare](https://developers.cloudflare.com/workers-ai/models/qwen3.8-27b/).
  - **Release-date conflict:** secondary sources say 5 Aug or 14 Aug 2026. — [codersera](https://codersera.com/blog/qwen-3-8-model-lineup-2026/) [S]
- **Naming conflict:** a codersera guide claims "No Qwen 3.8 model uses an A-suffix name", yet its own table lists Qwen3.8-2.4T-A95B, which exists on HF. That claim is unreliable. [S] — [codersera](https://codersera.com/blog/qwen-3-8-model-lineup-2026/); [HF](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)

**Qwen MoE line (all verified on HF)**

| Model | Date | Total / active | Experts (routed+shared active) | Layout / context | License | Key scores |
|---|---|---|---|---|---|---|
| [Qwen3-30B-A3B](https://arxiv.org/html/2505.09388) | 2025 | 30B / 3B | 128, top-8, no shared | 48 layers | Apache-2.0 [K for license] | — |
| [Qwen3-235B-A22B](https://arxiv.org/html/2505.09388) | 2025 | 235B / 22B | 128, top-8 | 94 layers | Apache-2.0 [K for license] | — |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | Feb 2026 | 397B / 17B | 512, 10+1; expert FFN 1024 | 60 layers, 15×(3×GDN + 1×GatedAttn); 262K native (~1M) | Apache-2.0 | MMLU-Pro 87.8, GPQA 88.4, SWE-V 76.4, LCB v6 83.6 |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Apr 2026 | 35B / 3B | 256, 8+1; expert FFN 512 | 40 layers, 10×(3×GDN→MoE + 1×GatedAttn→MoE); vision encoder + MTP; 262K (~1.01M) | Apache-2.0 | SWE-V 73.4, AIME26 92.7, GPQA 86.0, LCB v6 80.4, MMLU-Pro 85.2 |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Aug 2026 | 125B / 6B, +51B n-gram embedding, +4B MTP (180B listed) | 512, 10+1 | 48 layers, 12×(3×GDN→MoE + 1×"Qwen Sparse Attention"→MoE); 262K (~1M) | qwen-community-1.0 | SWE-Pro 62.5, GPQA-D 91.7, LCB v6 91.9 |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | Aug 2026 | 2.4T / 95B | 512, 10+1; expert FFN 2048 | 92 layers, 23×(3×GDN + 1×GatedAttn); text-only | custom "qwen3.8-max" | as Qwen3.8-Max: TB2.1 86.6, SWE-Pro 67.7, GPQA-D 92.6, HLE 43.6 |

Notes on the Qwen line:
- Qwen3.5-122B-A10B (Feb 2026, Apache-2.0) is listed only by a secondary source. [S] — [codersera](https://codersera.com/blog/qwen-3-8-model-lineup-2026/)
- Qwen3.6-35B-A3B was released 17 Apr 2026. Qwen3.6-Plus is API-only (2 Apr 2026). [S/V] — [qwen.ai blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b); [noze.it](https://www.noze.it/en/insights/qwen-3-6/)
- Qwen3.6-35B-A3B's recommended sampling is temperature 0.6–1.0 with top_k 20. Thinking is on by default, and a `preserve_thinking` option exists.

**Frontier open-weight MoEs (Oct 2026)**

| Model | Total / active | Experts | Attention / context | License | Key scores | Source |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro | 1.6T / 49B | 1 shared + 384 routed, top-6; first 3 layers hash-routed | CSA+HCA hybrid; 1M | MIT | SWE-V 80.6, GPQA-D 90.1, LCB 93.5 | [V] [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash); [arXiv 2606.19348](https://arxiv.org/html/2606.19348v1) |
| DeepSeek-V4-Flash | 284B / 13B | 1 shared + 256 routed, top-6; first 3 layers hash-routed | CSA+HCA hybrid; 1M | MIT | SWE-V 79.0, GPQA-D 88.1, LCB 91.6 | same as above |
| Kimi K3 | 2.8T / 104B | 896 routed, 16 per token + 2 shared; latent MoE dim 3584 | 93 layers (1 dense), 69 KDA + 24 Gated MLA; 1M; MoonViT-V2 vision | "Kimi K3 License" | GPQA-D 93.5, HLE 43.5/56.0 (no tools/tools), Terminal-Bench 2.1 88.3, BrowseComp 91.2 | [V] [HF](https://huggingface.co/moonshotai/Kimi-K3) |
| GLM-5.3 | 753B / not stated | not stated (tag `glm_moe_dsa`) | DSA | custom "glm-5.3" | Terminal Bench 2.1 88.2, DeepSWE 66.9, HLE w/ tools 62.5 | [V] [HF](https://huggingface.co/zai-org/GLM-5.3) |
| MiniMax-M3 | ~428B / ~23B | not on card | MiniMax Sparse Attention; 1M | `minimax-community` | SWE-V 80.5, SWE-Pro 59 | [V] [HF](https://huggingface.co/MiniMaxAI/MiniMax-M3) |

Additional details and conflicts for the frontier table:
- **DeepSeek-V4:** FP4 routed experts and FP8 elsewhere, Muon optimizer, mHC, >32T pretraining tokens. Paper submitted 26 Apr 2026.
- **DeepSeek-V4.1-Flash:** config shows 384 routed experts, top-6, sqrtsoftplus. [V] — [flashinfer #5190](https://github.com/flashinfer-ai/flashinfer/issues/5190). One aggregator lists V4.1-Flash at 552B total. [S] — [morphllm search summary](https://www.morphllm.com/best-open-source-coding-model-2026)
- **Kimi K3 release date conflict:** 13 Jun 2026 per one aggregator vs. 26/27 Jul 2026 per others. [S] — [kingy.ai](https://kingy.ai/news/best-open-weight-ai-models-in-2026-glm-5-2-vs-deepseek-v4-vs-kimi-k2-6-vs-qwen-vs-mistral/); [letsdatascience](https://letsdatascience.com/news/moonshot-ai-releases-kimi-k3-open-weights-099c256f). Early trackers quoted 32B–104B active parameters; the card says 104B. [S]
- **GLM-5.x family on HF (Oct 2026)** [V] — [HF zai-org](https://huggingface.co/zai-org):
  - GLM-5 (754B, updated Aug 11) and GLM-5.2 (753B, Sep 1).
  - GLM-5.3 (753B) and GLM-5.3-Flash (321B, image-text-to-text), both updated ~mid-Oct 2026.
  - The GLM-5 paper is arXiv 2602.15763 (Feb 2026). [V] — [arXiv](https://arxiv.org/abs/2602.15763)
  - Secondary sources: GLM-5 has 744B/40B active, MIT license, 256 routed experts with 8 active, and uses DeepSeek Sparse Attention. [S] — [NVIDIA NeMo docs](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/glm-5-mo-e-dsa.md)
  - GLM-5.3-Flash is reported as ~320B/18B active and MIT. [S] — [morphllm](https://www.morphllm.com/best-open-source-coding-model-2026)
- **MiniMax-M3 secondary claims** [S] — [howaiworks](https://howaiworks.ai/models/minimax-m3); [developersdigest](https://www.developersdigest.tech/local-models/minimax-m3):
  - 128 routed experts, top-4, plus 1 shared, per config.json.
  - License includes a non-commercial base grant, attribution, and written authorization above $20M revenue.
  - Paper "published Jun 11" (2026).
- **Xiaomi MiMo-V2.6-Pro** is named by one aggregator as #1 open model on the Artificial Analysis index, ahead of GLM-5.3. Not verified. [S] — [morphllm search summary](https://www.morphllm.com/best-open-source-llm)

**Earlier but still-relevant MoEs** [K unless noted]
- **gpt-oss-120b / gpt-oss-20b** (5 Aug 2025, Apache-2.0) — [OpenAI](https://openai.com/index/introducing-gpt-oss/):
  - 120b: 117B total / 5.1B active, 128 experts, top-4, 36 layers.
  - 20b: 21B / 3.6B active, 32 experts, top-4, 24 layers.
  - MXFP4 MoE weights.
  - Both are in vLLM's batch-invariance tested list. [V] — [vLLM docs](https://docs.vllm.ai/en/v0.20.2/features/batch_invariance/)
- **Llama 4** (Apr 2025, Llama 4 Community License) — [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/):
  - Scout: 17B active, 16 experts, 109B total.
  - Maverick: 17B active, 128 routed experts + shared, 400B total, alternating dense and MoE layers.
- **Mixtral 8x7B** (Dec 2023, Apache-2.0): 8 experts, top-2, ~47B total / ~13B active. — [arXiv 2401.04088](https://arxiv.org/abs/2401.04088)
- **Kimi K2** (Jul 2025, Modified MIT): 1T / 32B active, 384 experts, 8 per token + 1 shared. — [HF](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- **GLM-4.5 / GLM-4.5-Air** (Jul 2025, MIT): 355B/32B and 106B/12B. — [HF](https://huggingface.co/zai-org/GLM-4.5)
- **MiniMax-M2** (Oct 2025, MIT): 230B / 10B active. — [HF](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- **ERNIE-4.5** (Jun 2025, Apache-2.0): 300B-A47B and 21B-A3B. — [HF](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT)
- **Hunyuan-A13B** (Jun 2025): 80B / 13B active, Tencent community license. — [HF](https://huggingface.co/tencent/Hunyuan-A13B-Instruct)
- **Ling-mini-2.0** (2025, MIT): 16B / ~1.4B active. — [HF](https://huggingface.co/inclusionAI/Ling-mini-2.0)
- **Qwen3-Next-80B-A3B** (Sep 2025): hybrid Gated DeltaNet, 512 experts, 10+1 active. In vLLM's batch-invariance tested list. [V for the vLLM listing] — [vLLM docs](https://docs.vllm.ai/en/v0.20.2/features/batch_invariance/); [HF](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

**Small MoEs for CPU-scale experiments**
- **OLMoE-1B-7B** [V] — [arXiv 2409.02060v2](https://arxiv.org/html/2409.02060v2):
  - 6.9B total / 1.3B active, 64 experts (FFN 1024), top-8, 16 layers.
  - Dropless; load-balance loss 0.01, z-loss 0.001; ~5.1T tokens.
  - Fully open (weights, data, code, logs). The paper is the richest routing-analysis reference.
- **granite-3.1-1b-a400m**: ~1.3B total / 400M active, 32 experts, top-8, 128K context. [S] — [featherless](https://featherless.ai/models/ibm-granite/granite-3.1-1b-a400m-base); [NVIDIA NeMo](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/granite-mo-e)
  - It was used in vLLM issue #57016 as the minimal reproducer of MoE batch-variance (see Q5). [V] — [vLLM #57016](https://github.com/vllm-project/vllm/issues/57016)
- **granite-4.0-h-tiny**: 7B total / ~1B active hybrid Mamba2 MoE, reportedly 64 experts with 6 active. [S] — [IBM](https://www.ibm.com/new/announcements/ibm-granite-4-0-tiny-preview-sneak-peek); [llmreference](https://www.llmreference.com/model/granite-4.0-h-tiny)
- **DeepSeek-V2-Lite**: 15.7B / 2.4B active, 64 routed + 2 shared, top-6. [K] — [HF](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
  - It is a subject of the counterfactual-routing study [V] — [arXiv 2605.07260](https://arxiv.org/html/2605.07260)
  - It appears in vLLM's batch-variance bug report [V] — [vLLM #57016](https://github.com/vllm-project/vllm/issues/57016)
- **Qwen1.5-MoE-A2.7B** is also in vLLM #57016 and in the router-sensitivity pruning paper. [V] — [arXiv 2608.07890](https://arxiv.org/abs/2608.07890)
- **Qwen3-30B-A3B and gpt-oss-20b** are the de facto "mid-size lab rats":
  - Qwen3-30B-A3B is used in R3, GSPO, FP8-RL, counterfactual routing, and the SGLang/vLLM determinism docs.
  - gpt-oss-20b is used in counterfactual routing and the vLLM tested list.

### Inferences
- **Best base candidates for a "deterministic MoE", ranked for this project:**
  1. **Qwen3.6-35B-A3B** (Apache-2.0, 256 fine-grained experts + 1 shared, strong agentic/coding scores). Caveat: the Gated DeltaNet recurrent kernels add batch-invariance surface. The architecturally similar Qwen3-Next-80B-A3B is on vLLM's tested list.
  2. **Qwen3-30B-A3B** (Apache-2.0, plain attention). It has the most published routing-mismatch and determinism data, which makes it the best calibration baseline.
  3. **DeepSeek-V4-Flash** (MIT, 284B-A13B). It already has hash-routed early layers, aux-loss-free routing, a vendor commitment to bitwise-aligned training and inference kernels, and an in-progress vLLM 0-diff-RL PR. It is too big for CPU.
  4. **gpt-oss-20b** (Apache-2.0, 32 experts, top-4). Small and batch-invariance-tested in vLLM. Coarse experts mean fewer specialization levers.
  5. **CPU-scale:** OLMoE-1B-7B (fully open data and routing analysis), granite-3.1-1b-a400m (tiny and has a known batch-variance repro), DeepSeek-V2-Lite.
- **Licenses:** for anything frontier-scale with permissive terms, DeepSeek-V4 (MIT) is the clear choice. Kimi K3, GLM-5.3, MiniMax-M3, Qwen3.8-2.4T and Qwen3.8-Flash-Next all use custom or community licenses.
- **The user's "Qwen 3.8 27B" is a dense model,** so it is not a candidate for the MoE routing work. It could serve as a dense control or teacher, comparable to Qwen3-8B as the dense control in R3.

### Gaps
- Unverified in this session because the search budget ran out:
  - Mistral Large 3 (reportedly a 675B / 41B-active MoE from Dec 2025, Apache-2.0)
  - NVIDIA Nemotron 3 Nano / Super (hybrid Mamba MoE)
  - Xiaomi MiMo-V2.x
  - Ling-2.x / Ring-1T updates
  - Kimi K2.5/K2.6
  - DeepSeek-V4.1 full specs
  - ERNIE 5 and Hunyuan updates
- Model cards for GLM-5.3 and MiniMax-M3 do not list expert counts or top-k; only secondary or config-derived figures exist.
- Artificial Analysis index rankings were not fetched directly (the morphllm page returned 429), so benchmark standing beyond vendor-reported numbers is unverified.
- I did not verify that an `hf-internal-testing` tiny random Mixtral/Qwen3-MoE checkpoint exists. Building one from `Qwen3MoeConfig` / `MixtralConfig` with random weights is the safe route.

---

## Q5. MoE-specific serving determinism: expert parallelism, all-to-all ordering, grouped GEMM, capacity, and vLLM/SGLang status

### Takeaway
The dominant cause of inference nondeterminism is **lack of batch invariance**: kernel reduction order changes with batch size or composition. It is not atomics.

MoE adds extra batch-variant surfaces:
1. Fused-MoE / grouped-GEMM tile configs chosen by **tokens-per-expert counts**, which depend on co-tenant requests.
2. **Dynamic activation quantization scales computed over the whole batch** (online FP8).
3. Variable expert capacity buffers (`expert_num_tokens.max()`).
4. Expert-parallel dispatch/combine and all-reduce ordering.
5. Top-k tie-breaking.

As of Oct 2026:
- vLLM's `VLLM_BATCH_INVARIANT=1` lists Qwen3-30B-A3B, Qwen3-Next, DeepSeek-V3/R1 and gpt-oss as tested, yet open bugs show fused-MoE paths are **not** fully invariant. Examples are online FP8, Triton bf16 experts with prefix caching, and EP with DP>1 and TP>1 before a fix.
- SGLang's deterministic mode documents a Qwen3-30B-A3B example. Its Sep 2025 blog said only dense models were supported then.
- DeepSeek-V4 reports deterministic MoE backward via per-rank token-order preprocessing and buffer isolation.

### Cited Findings
**Root cause and the reference experiment (Thinking Machines, 10 Sep 2025)** [V] — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- Forward passes are run-to-run deterministic. The cause of user-visible nondeterminism is **batch invariance**: server load changes batch size, which changes reduction strategies.
- **Experiment:** Qwen3-235B-A22B-Instruct-2507 (an MoE), 1,000 completions at temperature 0.
  - Default vLLM gave **80 unique completions**; the most common occurred 78 times.
  - Completions diverged at token 103 ("Queens, New York" ×992 vs. "New York City" ×8).
  - With batch-invariant kernels, all 1,000 were identical.
- **Ops needing invariance:**
  - RMSNorm: no split reduction.
  - Matmul: no split-K or stream-K; use one fixed config. This costs ~20% vs. cuBLAS.
  - Attention: fixed split *size*, not a fixed number of splits, in split-KV decoding.
- **Cost on Qwen3-8B:** 26 s default → 55 s deterministic → 42 s with an improved attention kernel.
- The blog **does not discuss MoE routing kernels**.
- **RL:** bitwise-identical sampler and trainer gives KL = 0 ("true on-policy RL").

**vLLM** [V] — [vLLM docs v0.20.2](https://docs.vllm.ai/en/v0.20.2/features/batch_invariance/)
- `VLLM_BATCH_INVARIANT=1` requires NVIDIA compute capability ≥ 9.0 (H100/H200/B100/B200) and is in beta.
- It disables some optimizations, such as custom all-reduce under TP.
- Tested models include DeepSeek-V3/V3-0324/R1/V3.1, Qwen3-30B-A3B, Qwen3-Next-80B-A3B and gpt-oss-20b/120b.
- The docs do not discuss EP or MoE kernels.
- **Issue #57016** (opened 15 Sep 2026, vLLM 0.27.1) [V] — [vLLM #57016](https://github.com/vllm-project/vllm/issues/57016)
  - "`VLLM_BATCH_INVARIANT=1` is not batch-invariant for fused-MoE experts with no warning."
  - Greedy output depends on co-batched requests when experts run through Triton fused-MoE with **online FP8**. Cause: one dynamic activation scale over the whole batch, because `per_act_token_quant` is not passed.
  - **bf16 experts on Hopper**: the Triton MoE config is chosen per token count, so a warm prefix-cache request ≠ the cold one.
  - Dense layers were invariant in every test. Block-scaled FP8 (Qwen3-30B-A3B-FP8) was invariant on H100.
  - **Measured first-token flips:**

    | Model / setting | Hardware | First flip at token | Logprob gap |
    |---|---|---|---|
    | granite-1b-a400m fp8 | H100 | 1 | 0.07 |
    | granite-1b-a400m fp8 | RTX 4090 | 4 | — |
    | Qwen1.5-MoE fp8 | H100 | 4 | 0.53 |
    | DeepSeek-V2-Lite fp8 | RTX 4090 | 12 | — |

  - Fix PR **#57092** switches to per-token activation scales under the flag and adds a warning for unsupported backends. It was open at capture. — [vLLM #57092](https://github.com/vllm-project/vllm/pull/57092)
- **PR #61118** enables inductor batch-invariant mode. It states that the **MoE + expert-parallel deployment with DP>1 and TP>1 was not batch-invariant** before the change. — [vLLM #61118](https://github.com/vllm-project/vllm/pull/61118) (per search summary) [S/V]
- **Issue #51290**: batch invariance broke under TP>1 because of the fused all-reduce+RMSNorm (`fuse_allreduce_rms`). Disabling it restored determinism at TP=4. — [vLLM #51290](https://github.com/vllm-project/vllm/issues/51290) (per search summary)
- **PR #54955** "[DeepSeek-V4] Batch-invariant kernels for 0-diff RL" (draft, Sep 2026, DO NOT MERGE) [V] — [vLLM #54955](https://github.com/vllm-project/vllm/pull/54955)
  - Requires a batch-invariant MoE backend with **fixed expert capacity instead of `expert_num_tokens.max()`**.
  - Router gate logits use a common **FP32 `torch.mm`**.
  - Deterministic top-k with **stable tie-break** (`ds4_bi.top_k_per_row_prefill`).
  - Fixed-geometry fused SiLU×mul + group FP8 quant.
  - Includes DeepEP low-latency dispatch changes.
  - Reported "chunked/full bitwise parity on GB200" after a 64-bit offset fix. Formal test results are still TODO.
- **ROCm/aiter PR #6583** adds a deterministic fused_moe mode (bf16×MXFP4) with no split-K, a single block size, and a reduce epilogue. The motivation: the down-projection's **atomic adds occur in workgroup-completion order**. — [aiter #6583](https://github.com/ROCm/aiter/pull/6583) (per search summary)

**SGLang**
- `--enable-deterministic-inference` builds on Thinking Machines' batch-invariant ops. [V] — [SGLang docs](https://docs.sglang.io/advanced_features/deterministic_inference.html)
  - Supported attention backends: FlashInfer (no radix cache), FA3 and Triton.
  - The docs show a **Qwen3-30B-A3B (MoE)** example.
  - Per-request `sampling_seed` gives reproducible temperature>0 sampling.
  - Tests cover `single`, `prefix` and `radix_cache` modes, expecting "Unique samples: 1".
  - Roadmap: [sglang #10278](https://github.com/sgl-project/sglang/issues/10278).
- **LMSYS blog** (22 Sep 2025) [V] — [LMSYS blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
  - Average deterministic-mode overhead of **34.35%** (FlashInfer/FA3) vs. 61.5% reported by Thinking Machines; Triton was +44–55%.
  - "Currently we only support deterministic inference for dense models like QWen3-8B or LLaMa-3.1-8B." MoE (Qwen3-30B-A3B, DeepSeek-V3) was on the roadmap.
  - TP1 and TP2 were deterministic; larger TP needs modified reduce kernels.
  - Reproducible RL with slime required FA2 (deterministic backward), Megatron `--deterministic-mode`, `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`.

**DeepSeek-V4 kernel determinism** (§3.3 "High-Performance Batch-Invariant and Deterministic Kernel Libraries") [V] — [arXiv 2606.19348](https://arxiv.org/html/2606.19348v1)
- MoE backward determinism comes from per-rank token-order preprocessing and buffer isolation across ranks. This makes EP send results and MoE backward accumulation order deterministic.
- The kernels target **bitwise alignment across pre-training, post-training and inference**.
- §5.2.3 notes that a batch-invariant, deterministic inference stack would allow regeneration with a fixed sampler seed.

**Capacity and token dropping**
- Capacity-factor dispatch (GShard/Switch) drops overflow tokens, which makes outputs depend on co-batched tokens. [K] — [arXiv 2101.03961](https://arxiv.org/abs/2101.03961); [arXiv 2006.16668](https://arxiv.org/abs/2006.16668)
- DeepSeek-V3 does "no token-dropping" in inference. [K] — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- OLMoE is dropless. [V] — [arXiv 2409.02060v2](https://arxiv.org/html/2409.02060v2)

### Inferences
- **A "deterministic MoE" serving checklist**, combining all sources above:
  1. Router logits in FP32 using a single fixed GEMM path.
  2. Deterministic top-k with a stable lowest-index tie-break.
  3. Dropless routing, or a fixed expert capacity that can never overflow.
  4. Per-token (not per-batch) activation quantization scales, or no online quantization.
  5. Fused-MoE/grouped-GEMM with a single tile config independent of tokens-per-expert, and no split-K or atomics in the down-projection or combine.
  6. Deterministic EP dispatch/combine order (DeepSeek-V4-style per-rank token ordering) and deterministic all-reduce. This means no fused all-reduce+RMSNorm, and NCCL Ring.
  7. Batch-invariant attention, including any linear-attention/DeltaNet recurrent kernels for Qwen3.5+.
  8. Prefix-cache warm/cold parity tests.
  9. Fixed TP/EP topology. Determinism across different TP or EP sizes is **not** claimed by any engine.
- **Testing:** the vLLM #57016 reproducer pattern (co-batch flips and warm-vs-cold prefix flips on granite-1b-a400m) is a cheap, ready-made regression test for a determinism harness. So is SGLang's `test_deterministic` (single/prefix/radix_cache).
- **Expected cost:** ~20–60% throughput, per the Thinking Machines and SGLang numbers. No MoE-specific overhead figure was found.

### Gaps
- I found no published throughput overhead for batch-invariant **MoE** serving (all overhead numbers are for Qwen3-8B dense).
- I found no documentation of determinism **across** different EP/TP degrees or across GPU generations (H100 vs. B200). DeepSeek-V4 claims bitwise alignment across training stages but gives no cross-hardware claim in the retrieved text.
- I could not confirm whether SGLang's MoE deterministic mode (beyond the Qwen3-30B-A3B doc example) covers DeepEP/EP > 1 as of Oct 2026. The docs do not say.
- I did not check CPU inference stacks (llama.cpp, ONNX Runtime) for MoE batch invariance. CPU-scale experiments would need this.

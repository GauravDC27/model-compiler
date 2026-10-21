# Extending a pretrained fine-grained MoE: expert-selective tuning, new experts/layers, router training, and goal-conditioned routing (state as of 21 Oct 2026)

Target: Qwen3.6-35B-A3B. Baseline recipe already in use: rank-16 LoRA on "hot" routed experts (~97/256 per layer), rank-32 LoRA on attention, linear attention and the shared expert, router frozen, top-20 distillation from a dense 27B teacher, then decision-token DPO. LoRA on all 256 experts added nothing.

Conventions: each finding gives the paper or page date in [brackets]. "Inferences" are my reasoning from the cited facts. They are not sourced claims.

---

## 1. Expert-selective fine-tuning and MoE-specific PEFT (ESFT, REAP-style importance, pruning/merging, hot-expert adapters, MoLE/LoRAMoE/MixLoRA/HydraLoRA/MoSLoRA/HMoRA, Qwen3.x results)

### Takeaway
On fine-grained MoEs, tuning only the routed experts that a task actually uses is the best-supported way to trade specialization against forgetting. In ESFT, training about 5–15% of experts reached about 98% of full fine-tuning's task gain and kept far more general ability. The Oct-2026 PEFT study on Qwen3-30B-A3B found expert-only adaptation best, router tuning near-neutral, and any load-balancing loss harmful. The newer frontier swaps capacity inside a fixed expert budget: UMoE prunes low-saliency experts (REAP score) and regrows them as copies of task-salient experts. On Qwen3.5-35B-A3B this beat direct SFT at identical serving cost. Gated "MoE-of-LoRA" add-ons (LoRAMoE, MixLoRA, HydraLoRA, HMoRA) were designed for dense bases. On native MoEs, plain weight-level LoRA on the experts matched or beat expert-level MoE-LoRA bypass variants.

### Cited Findings
**ESFT (DeepSeek) [2 Jul 2024]**
- Method: per layer, score each expert by relevance on about 32 task samples of 4,096 tokens. Two scores are used: average gate score (ESFT-Gate) and token-selection ratio (ESFT-Token). Experts are picked greedily until cumulative relevance reaches threshold p, and only those are trained. All other experts, the shared experts, attention and the router stay frozen — [ESFT HTML](https://arxiv.org/html/2407.01906v2)
- Thresholds: main runs used p=0.1 (Gate) and p=0.2 (Token). Trained experts ranged from 2 to 15 of 66 per layer, mostly 5–15%, which is 75–95% fewer trainable parameters than FFT. Gains saturate at p≈0.2 (Token) and p≈0.1 (Gate) — [ESFT HTML](https://arxiv.org/html/2407.01906v2)
- Results on DeepSeek-V2-Lite, specialized-task average: vanilla 33.6, FFT 51.0, LoRA 44.9, ESFT-Token 49.4, ESFT-Gate 50.2. General-ability average (MMLU, TriviaQA, IFEval, ARC, etc.): vanilla 62.4, FFT 58.8, LoRA 59.1, ESFT-Token 61.5, ESFT-Gate 60.6 — [ESFT HTML](https://arxiv.org/html/2407.01906v2)
- Ablation: training all parameters (15.7B) gave specialized 51.0 / general 58.8. Training only the task-relevant non-shared experts (~1.4B) gave 49.4 / 61.5. The authors attribute general-ability loss mainly to training the shared parameters (shared experts and non-expert modules) — [ESFT HTML](https://arxiv.org/html/2407.01906v2)
- Swapping the selected experts for random experts hurts every task significantly. Simulated coarse-grained grouping degrades ESFT faster than FFT, so finer experts favor ESFT. Mixing in alignment data (1:1) helps FFT and LoRA but not ESFT — [ESFT HTML](https://arxiv.org/html/2407.01906v2)
- Cost: training time was 19.8–20.9 min (ESFT), 28.5 min (FFT) and 16.5 min (LoRA). Stored parameters were 2.57–3.20 GB for ESFT vs 28.6 GB for FFT. ESFT was tested only on DeepSeek-V2-Lite — [ESFT HTML](https://arxiv.org/html/2407.01906v2)

**REAP: router-weighted expert importance [v1 15 Oct 2025, v3 13 May 2026]**
- Saliency formula: S_j = mean over tokens routed to j of g_j(x)·‖f_j(x)‖₂, averaged only over tokens where the expert is active, so it is not dominated by usage frequency — [REAP HTML](https://arxiv.org/html/2510.13999)
- Qwen3-30B-A3B (128 experts, top-8), baseline → 25% / 50% pruned:
  - code 0.558 → 0.551 / 0.541
  - math 0.887 → 0.888 / 0.857
  - WildBench 0.811 → 0.804 / 0.718
  - multiple choice 0.721 → 0.665 / 0.503
  
  Generative and code ability survive pruning far better than MC knowledge does — [REAP HTML](https://arxiv.org/html/2510.13999)
- Qwen3-Coder-480B: code drops only 0.16% at 25% pruning and 1.2% at 50%. Kimi-K2 BFCL goes 0.666 → 0.635 (25%) → 0.564 (50%), so tool-calling degrades faster than code — [REAP HTML](https://arxiv.org/html/2510.13999)
- Merging loses to pruning on generative tasks. At 50% on Qwen3-30B, HC-SMoE scored 0.364 on code and 0.008 on WildBench, and M-SMoE scored 0.397 on code. The authors call the cause "functional subspace collapse": merged experts contract toward the manifold centre, with up to 100× less spread in late layers of many-expert models. Frequency-based pruning scored 0.452 on code at 50% — [REAP HTML](https://arxiv.org/html/2510.13999)
- Calibration data matters. C4 calibration produced incoherent code models (0% accuracy in some cases). Domain calibration (e.g., evol-codealpaca) or combined data was best. Code is at github.com/CerebrasResearch/reap, and checkpoints are in the HF collection cerebras/cerebras-REAP — [REAP HTML](https://arxiv.org/html/2510.13999)

**Prune-and-regrow inside a fixed expert budget: UMoE [13 Jul 2026]**
- Pipeline:
  1. Compute domain REAP saliency on 1,024 domain SFT samples of 2,048 tokens.
  2. Prune 50% of routed experts per layer.
  3. Regrow back to the original count by copying each retained expert's MLP and router row into a freed slot.
  4. Perturb both parent and copy independently with Gaussian noise σ=0.05×std(W). Noise goes on both the MLP weights and the router rows.
  5. Run standard SFT of all parameters (built on slime; 1 epoch, LR 1e-4 → 1e-5).
  
  Expert count, parameter count and inference cost are unchanged — [UMoE HTML](https://arxiv.org/html/2607.11444)
- Qwen3.5-35B-A3B math (avg@32), direct SFT → UMoE:
  - AIME24 78.65 → 80.31
  - AIME25 70.99 → 72.03
  - HMMT Nov-25 54.69 → 58.75
  - Average 61.75 → 63.42 (+1.67)
  
  Qwen3-30B-A3B gained +3.4 on the math average and +6.0 on SWE-bench Verified — [UMoE HTML](https://arxiv.org/html/2607.11444); [UMoE abstract](https://arxiv.org/abs/2607.11444)
- Failure modes:
  - Perturbing only the regrown copy ("one-sided") underperformed direct SFT (33.61 vs 37.85), because routing to the regrown half declines during SFT.
  - Training loss starts higher than direct SFT.
  - Post-hoc pruning of the UMoE model back to 64 experts costs about 5 points.
  - No general or out-of-domain retention numbers are reported.
  
  — [UMoE HTML](https://arxiv.org/html/2607.11444)

**Empirical PEFT study on native MoEs [19 Oct 2026]**
- Setup: Qwen3-30B-A3B-Base, Gemma-4-26B-A4B and GPT-OSS-20B. Tasks were math (MATH-500) and intent classification, with general retention measured on MMLU-Pro and IFEval — [PEFT-for-MoE study](https://arxiv.org/html/2610.26370)
- Where to adapt:
  - Experts only: 68.6% downstream, MMLU-Pro −3.3, IFEval −3.6.
  - Attention only: 61.7%, with the largest general loss (−10.4 / −7.9).
  - Attention + experts: 67.0%, with the best retention.
  
  — [study](https://arxiv.org/html/2610.26370)
- Router tuning: experts+router scored 68.1% vs 68.6% for experts only. At the higher budget it was 64.9% vs 64.8%. The authors recommend adapting the router only when the task needs routing specialization — [study](https://arxiv.org/html/2610.26370)
- Load-balancing auxiliary loss during PEFT: coefficient 0 gave 68.5%. Coefficients 1e-4, 1e-3, 1e-2 and 1e-1 gave 68.1%, 66.8%, 64.6% and 58.6% — [study](https://arxiv.org/html/2610.26370)
- Granularity and method:
  - Weight-level LoRA inside experts (69.1% at r=2, 68.5% at r=1) beat module-level bypass (68.4%) and expert-level bypass (64.7%).
  - LoRA and DoRA tied at 68.5%, with DoRA keeping more IFEval.
  - Among MoE-specific variants: layer-level LoRA 66.7%, MoE-LoRA 65.7%, DAS-LoRA (domain-score expert selection) 64.8%.
  - Rank 1 had the best general-retention profile.
  
  — [study](https://arxiv.org/html/2610.26370)

**Other 2026 MoE-PEFT preprints**
- MoE²-LoRA [24 Jul 2026]: reuses base-router activations to route LoRA experts (Routing-Conditioned Projection) and uses a single global LoRA expert pool shared across layers. It argues static expert selection limits per-token capacity. The abstract gives no numbers — [MoE²-LoRA](https://arxiv.org/abs/2607.21978)
- ACE [5 Sep 2026, EMNLP 2026]: per-expert LoRAs become functionally redundant during fine-tuning. ACE groups redundant experts and gives each group one shared, higher-rank LoRA at equal budget. It reports the best mean accuracy among parameter-matched PEFT on 3 backbones and 1.31–1.48× faster training than expert-wise LoRA — [ACE](https://arxiv.org/abs/2609.06072)
- TEXAS [31 Jul 2026]: finds task experts by contrasting activations on instances the base model solves vs fails, then up-weights answer tokens in failed instances that hit those experts. It does not restrict training to a subset. Best or tied-best in 17/18 settings, +1.3–1.5 points over the strongest baseline — [TEXAS](https://arxiv.org/abs/2608.06396)
- DR-LoRA [Jan 2026]: dynamic rank allocation per expert. Evaluated on OLMoE, Qwen1.5-MoE and LLaMA-MoE, not Qwen3 — [DR-LoRA](https://arxiv.org/pdf/2601.04823)

**MoE-of-LoRA designs (mostly built on dense bases)**
- LoRAMoE [Dec 2023]: LoRA experts plus a router as a plug-in, with a localized balancing constraint. Aimed at reducing world-knowledge forgetting in large-scale SFT — [LoRAMoE](https://arxiv.org/abs/2312.09979)
- MixLoRA [Apr 2024]: top-k-routed LoRA experts inside the FFN with an auxiliary balance loss. Per the abstract, about 9% accuracy gain over state-of-the-art PEFT in multi-task settings, and its framework cuts GPU memory by about 40% and latency by about 30% — [MixLoRA](https://arxiv.org/abs/2404.15159)
- HydraLoRA [Apr 2024]: asymmetric design with one shared A and several routed B matrices — [HydraLoRA](https://arxiv.org/abs/2404.19245)
- MoSLoRA [Jun 2024]: a learnable mixer between LoRA subspaces — [MoSLoRA](https://arxiv.org/abs/2406.11909)
- MoLE [Apr 2024]: learned gating over pre-trained LoRAs — [MoLE](https://arxiv.org/abs/2404.13628)
- HMoRA [ICLR 2025]: hierarchical hybrid token-level plus task-level routing of LoRA experts, with an auxiliary loss that sharpens routing certainty and task separation. The authors note task-level routing alone generalizes poorly to unseen tasks and that load-balancing loss weakens specialization. Outperforms full fine-tuning while training 3.9% of parameters — [HMoRA (ICLR proceedings)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e43a33994a28f746dcfd53eb51ed3c2d-Abstract-Conference.html)
- MoE-LPR [Aug 2024] compared LoRAMoE on a Qwen1.5-1.8B expansion task: expanded languages 42.37, original-language retention about 94.9%. MoE-LPR reached 45.07 and about 96.6% — [MoE-LPR HTML](https://arxiv.org/html/2408.11396)

**Domain-specific expert pruning for serving (related)**
- REAP works best with domain calibration data (see above). Releasing the low-saliency experts after task specialization is a serving-cost lever; UMoE notes direct-SFT models spend substantial routed compute on low-saliency experts that can be removed post hoc with little average degradation — [UMoE abstract](https://arxiv.org/abs/2607.11444)

### Inferences
- The baseline's hot set (~97/256 ≈ 38% of experts per layer, from 80%/50% mass unions) is far looser than ESFT's optimum (5–15% of experts, p≈0.1–0.2). Two cheaper selections deserve an A/B: ESFT-Token at p=0.2–0.5, and a REAP-saliency ranking (gate × output norm). That "all 256" added nothing fits ESFT's random-expert result: cold experts receive almost no gradient-carrying tokens.
- The Oct-2026 PEFT study and ESFT both point at shared parameters as the main source of forgetting. The baseline trains rank-32 LoRA on attention, linear attention and the shared expert. Worth ablating: (a) freeze the shared-expert LoRA, (b) drop attention LoRA rank to 8–16, and (c) DoRA in place of LoRA on experts, if IFEval-style instruction retention matters for voice agents.
- Keep the load-balancing auxiliary loss at 0 during PEFT and distillation. Qwen3.6 ships router_aux_loss_coef=0.001, and HF Trainer paths may add it if output_router_logits is turned on. With the router frozen this mostly matters only if router LoRA or bias is trained.
- UMoE is the most direct "beyond the baseline" step: same 256-slot serving footprint, but low-saliency slots reassigned to copies of voice-agent-salient experts. It is cheap to try: 1 calibration pass plus SFT. Key risks:
  - Unknown general-ability retention, since none was reported.
  - Smaller gains on Qwen3.5 (+1.67) than on Qwen3 (+3.4).
  - It conflicts with LoRA-only deployment: it needs full expert weights and one merged model.
- Pruning results show multiple-choice knowledge (MMLU-like) is the first casualty of expert removal. Any prune/regrow or merge must be gated on knowledge benchmarks, not only task metrics. Tool-calling (BFCL) also degrades faster than code.

### Gaps
- No published ESFT / hot-expert-LoRA results on Qwen3.5/3.6 with distillation from a dense teacher, and none on speech/voice-agent data.
- I found no head-to-head of union-of-mass-set selection (the baseline's heuristic) vs ESFT-Token/Gate vs REAP saliency on the same model.
- The PEFT study adapts all experts by default and has only one selection variant (DAS-LoRA, lower at 64.8%). No dedicated "all vs selected experts" comparison exists on Qwen3.x.
- MoE²-LoRA and ACE numbers on Qwen3 backbones were not visible in the abstracts.
- Not re-fetched this session: the LoRAMoE, MixLoRA, HydraLoRA, MoSLoRA and MoLE descriptions (including MixLoRA's ~9%, 40% and 30% figures) are from their arXiv abstracts as recalled. Verify them before quoting numbers.

---

## 2. Adding NEW experts / capacity to a trained MoE (expansion, continual learning, router-row init, load balancing, freezing, forgetting)

### Takeaway
The well-evidenced recipe has three parts. (1) Initialize new experts by copying (upcycling) existing FFNs or experts, not randomly. (2) Freeze the old experts, plus attention and embeddings when forgetting matters. (3) Initialize the new router rows deliberately: from domain embeddings (FlexOlmo, Nexus), from the parent expert's row plus noise (UMoE, Expert Upcycling), or behind a learned task-prior loss (MoE-LPR). Then run a short router-only or "review" phase on <1% replay data to pull old-domain tokens back to old experts. The main failure modes are routing drift (old-domain tokens migrate to new experts even with old experts frozen), new experts starving, and expert collapse when load balancing is absent. Keeping top-k fixed keeps active FLOPs constant; the cost is memory.

### Cited Findings
**MoE-LPR: freeze old, add copied experts, router-only review [21 Aug 2024]**
- Setup: upcycle Qwen1.5-1.8B into 6 experts (original + 5 copies), top-2. Stage 1 freezes all original parameters and trains only the new experts and router on 24B tokens of new-language data, with LBL weight 0.01 to avoid routing collapse — [MoE-LPR HTML](https://arxiv.org/html/2408.11396)
- Stage 2 ("review"): train only the router on about 0.138B replay tokens (about 0.6% of stage 1). Language Priors Routing loss (weight 0.1) pushes original-language tokens' top-1 to frozen expert 0, and LBL is removed — [MoE-LPR HTML](https://arxiv.org/html/2408.11396)
- Results:

  | Method | Expanded-language avg | Original-language avg | Retention |
  |---|---|---|---|
  | Base | 34.36 | 53.97 | 100% |
  | LoRA | 33.02 | 44.43 | ~82% |
  | Full FT | 41.12 | 49.73 | ~92% |
  | LLaMA-Pro | 41.10 | 50.99 | ~94.5% |
  | LoRAMoE | 42.37 | 51.24 | ~94.9% |
  | MoE-LPR | 45.07 | 52.12 | ~96.6% |

  Activated parameters stay constant (2.6B) — [MoE-LPR HTML](https://arxiv.org/html/2408.11396)
- Ablations:
  - Without review: original-language average 47.14, i.e., severe forgetting.
  - Without the LPR loss: 51.32.
  - Random-init (not copied) experts: expanded-language performance drops significantly.
  - Full FT at higher LR collapses to near-random MC.
  - Replay gains flatten after about 50K documents.
  
  — [MoE-LPR HTML](https://arxiv.org/html/2408.11396)
- LayerMoE [28 May 2025, ACL 2025]: allocates fewer new experts to layers with high cross-lingual representation similarity, with a classifier in front of the router on those layers that sends old-language tokens to the original experts. Uses 60% fewer experts (single expansion) and 33.3% fewer (lifelong expansion) than the prior SOTA — [LayerMoE](https://arxiv.org/abs/2505.22582)

**FlexOlmo: independently trained experts with domain-embedding router rows [v4 23 Aug 2025]**
- Each new expert is trained in a 2-expert MoE alongside the frozen public FFN expert, with frozen shared attention, for about 50B tokens. Without this frozen anchor, the experts diverged and merged poorly — [FlexOlmo HTML](https://arxiv.org/html/2507.07024)
- Router-row init: each expert's router embedding starts as the mean GRIT-embedder domain embedding of its data and is then fine-tuned along with the expert while the public row stays frozen. A per-expert negative bias makes experts compete with each other, not only with the public expert. Merging just concatenates the rows; optional router tuning uses about 5B tokens of proxy data — [FlexOlmo HTML](https://arxiv.org/html/2507.07024)
- Ablations (Table 1 average):
  - Full FlexOlmo (no router tuning): 46.7.
  - Without coordinated training: 38.8.
  - Without domain-embedding init and bias: 44.4.
  - Without bias only: 45.8.
  - Init from public-model hidden states: 43.5.
  - Random router init makes expert embeddings too similar.
  
  — [FlexOlmo HTML](https://arxiv.org/html/2507.07024)
- Result: about 10% better than the strongest merging baseline (BTM). Removing one expert (opt-out) mainly hurts that domain. The paper's headline "41% relative" gain over the public model is not clearly reproducible from its tables (Tables 1–2 imply about 24–30%) — [FlexOlmo HTML](https://arxiv.org/html/2507.07024)

**Nexus: router independent of expert count [28 Aug 2024; EMNLP 2025]**
- The router is a learned MLP that maps a domain embedding to an expert embedding, so a new expert's router row is P_r(d_new) with no retraining from scratch. Routing is top-1 plus a shared expert. Adding the new code expert used ≤1B tokens of light fine-tuning (50% new domain, 50% old) — [Nexus HTML](https://arxiv.org/html/2408.15901); [EMNLP talk](https://underline.io/lecture/133210-nexus-adaptive-upcycling-to-efficiently-pretrain-mixture-of-experts)
- Results: on the new domain, +18.4% (200M tokens) and +18.8% (1B tokens) relative over a linear-router MoE. General performance dropped 1.9% relative after extension — [Nexus HTML](https://arxiv.org/html/2408.15901)

**BTX, BTS, Lifelong-MoE, Expert Upcycling, EMO**
- BTX [12 Mar 2024]:
  - Setup: Llama-2-7B seed branched into math, code and Wikipedia experts plus the original model (4 experts, top-2, 11.1B active). Router learned during 80B tokens of MoE fine-tuning with LBL α=0.01; attention is averaged — [BTX HTML](https://arxiv.org/html/2403.07816)
  - Average 47.9 vs Llama-2 40.7, sparse upcycling 46.3 and BTM 43.4.
  - Without LBL the math expert dominates and the code expert dies.
  - Freezing the expert FFNs during MoE fine-tuning changed little (34.7 vs 34.7 in a 10B-token ablation). Blending experts dropped to 22.2.
  
  — [BTX HTML](https://arxiv.org/html/2403.07816)
- Branch-Train-Stitch [31 Jan 2025]: frozen expert LLMs connected to a frozen seed model through small trainable stitch layers. Experts can be added or removed with a small amount of extra training. The abstract gives no numbers — [BTS](https://arxiv.org/abs/2502.00075)
- Lifelong-MoE [20 May 2023, ICML 2023]: adds experts for each new data distribution with regularized pretraining at constant compute. Better few-shot results on 19 tasks than prior lifelong methods — [Lifelong-MoE](https://arxiv.org/abs/2305.12281). Search-snippet summaries say it freezes old experts and gates and that random-init new experts and gates did poorly; I did not verify this in the full text — [search result PDF](https://arxiv.org/pdf/2305.12281)
- Expert Upcycling [21 Apr 2026]: grows E → mE experts during continued pretraining by duplicating experts and extending the router, with top-K fixed. At 7–13B it matches the fixed-size baseline loss with 32% fewer GPU hours. Gradient-based utility selection of which experts to duplicate more than triples gap closure under limited CPT — [Expert Upcycling](https://arxiv.org/abs/2604.19835)
- EMO [13 May 2026]: progressive expert-pool growth during pretraining with a sparsity-aware scaling law. Matches fixed-expert training and improves wall-clock — [EMO](https://arxiv.org/abs/2605.13247). A search snippet says the init strategies for new experts and router rows differ only marginally given enough tokens; not verified in full text — [EMO PDF](https://arxiv.org/pdf/2605.13247)

**LoRA-expert continual learning**
- Self-MoE (MiXSE) [17 Jun 2024]: self-specialized experts trained on self-generated data, plus a router. About +6.5 points average over the base across knowledge, reasoning, math and code, beating instance and weight merging. The authors note non-specialty tasks can degrade — [Self-MoE](https://arxiv.org/abs/2406.12034)
- MoE-CL [Sep 2025]: one dedicated LoRA expert per task plus a shared LoRA expert whose transfer is filtered by a GAN task-discriminator. A/B test on Tencent Video content review cut manual review cost by 15.3% — [MoE-CL](https://arxiv.org/html/2509.18133v3)
- GoD-MoE (AAAI 2026) grows LoRA experts on demand for continual instruction tuning — [GoD-MoE](https://ojs.aaai.org/index.php/AAAI/article/view/40077)
- DynMoE [May 2024, ICLR 2025]: "top-any" gating plus automatic adding and removing of experts during training — [DynMoE](https://arxiv.org/abs/2405.14297)

**Router drift when experts are added (failure mode)**
- StaR-MoE [17 May 2026, class-incremental vision]: after expansion, the router reassigns old-class samples to new experts even though the old experts are frozen ("routing drift"). The fix is sensitivity-aware routing alignment to historical routing distributions plus asymmetric capacity regularization — [StaR-MoE](https://arxiv.org/abs/2605.17571)
- RestoreMore [Aug 2026, image restoration]: avoids touching old routers by using per-expert query-key routing keys, so adding an expert only appends a key — [RestoreMore](https://arxiv.org/pdf/2608.30305)
- Continual pretraining of MoEs [6 Mar 2025, v2 Nov 2025]: MoEs (500M active / 2B total, 600B tokens) with Sinkhorn or z+aux balancing were "surprisingly robust" to distribution shift, even without replay, and matched full re-training at a fraction of the cost — [Thérien et al.](https://arxiv.org/abs/2503.05029)

### Inferences
- Parameter accounting for Qwen3.6-35B-A3B, computed from its config:
  - One routed expert is 3×2048×512 ≈ 3.15M parameters per layer.
  - Adding k experts per layer in all 40 layers costs k×126M parameters (k×≈252 MB in bf16). k=8 gives about 1.0B (+~3%); k=32 gives about 4.0B (+~11.5%).
  - Each new router row is 2,048 parameters per layer, which is negligible.
  - With top-8 unchanged, active FLOPs are unchanged.
  - For comparison, rank-16 LoRA on one expert is about 0.09–0.12M parameters, so the baseline's ~97 hot-expert LoRAs per layer equal about 3–4 full new experts per layer.
- Recommended expansion recipe for task capacity (synthesis of MoE-LPR, FlexOlmo, UMoE and StaR-MoE):
  1. Pick parents by task saliency: REAP or ESFT-Token on voice-agent traffic, per layer.
  2. Append k copies per layer. Initialize each new router row as parent row + noise, perturbing the parent and the copy symmetrically, as in UMoE. Alternatively use a domain/goal-embedding projection, as in FlexOlmo or Nexus.
  3. Freeze all 256 original experts, the shared expert, attention and GDN. Train new experts plus the router (or only the new router rows plus a bias) with a small LBL computed only over the new experts, or with aux-loss-free bias balancing.
  4. Run a router-only "review" on ~0.5–1% replay of general data with an LPR-style loss that keeps non-target tokens on original experts.
  5. Track routing drift with the per-layer KL between base-router and new-router distributions on general data.
- One-sided initialization (only the new copy perturbed) starved the copies in UMoE. A new expert whose router row is random or low-norm will likely be ignored unless the router is trained or biased. Conversely, a row identical to its parent's splits traffic 50/50 and duplicates function until training breaks the symmetry.
- For a voice-agent product, the cleaner alternative to growing the expert count is UMoE-style in-place reallocation (Section 1). It keeps num_experts=256, so EP layouts, kernels, quantization recipes and vLLM configs stay untouched.

### Gaps
- No published expert-expansion result on a Gated-DeltaNet hybrid (Qwen3-Next/3.5/3.6) or on any 256-expert, top-8, width-512 MoE. All expansion evidence is on coarse or dense upcycled models (1.8B–13B) or vision.
- No published head-to-head of router-row init schemes (parent copy+noise vs domain embedding vs zero/negative bias) on a native fine-grained MoE. The EMO and Expert Upcycling init details were not visible in the abstracts.
- Forgetting numbers for Expert Upcycling, EMO and Lifelong-MoE were not available from the pages read.

---

## 3. Adding NEW layers (block expansion, depth up-scaling, MoE-layer insertion, adapters as layers)

### Takeaway
Identity-initialized block expansion (LLaMA Pro) is the best-documented way to add depth while freezing the original network. Interleaved zero-output blocks trained alone on domain data gave large code and math gains with roughly flat general scores, and it beat full FT, LoRA and MoE-FFN expansion on the gain-versus-forgetting balance. But it raises per-token latency and FLOPs, unlike adding experts. At equal parameter budget, MoE-LPR (new experts) beat LLaMA-Pro on both new-domain gain and retention. For a latency-bound voice agent on a hybrid GDN stack, new layers are the weaker option.

### Cited Findings
- LLaMA Pro [v2 30 May 2024]:
  - Expanded LLaMA2-7B from 32 to 40 blocks (8 new, interleaved), zero-initializing each new block's output projections (W^O, W3) so it starts as identity. RMSNorm is not zeroed, because it would get zero gradient.
  - Only new blocks were trained, on about 80B tokens of code and math (Proof-Pile-2 + Python), at about 2,830 H800 GPU-hours.
  
  — [LLaMA Pro HTML](https://arxiv.org/html/2401.02415)
- Gains: average 44.23 vs LLaMA2-7B 39.62. HumanEval 28.66 vs 13.05; GSM8K 17.89 vs 14.48. General scores barely moved (MMLU 47.88 vs 46.87, HellaSwag 77.94 vs 78.59). Mistral-Pro: GSM8K 39.2 → 50.6 but MMLU 62.7 → 60.6 — [LLaMA Pro HTML](https://arxiv.org/html/2401.02415)
- Law-domain comparison (general avg / domain Unfair-ToS):
  - 8 blocks: 56.70 / 75.11
  - LoRA r=1024: 58.15 / 65.34
  - Full FT: 55.01 / 70.63
  - MoE expansion: 56.56 / 67.27
  - Prefix (bottom) stacking: 29.36 / 0.81, i.e., collapse
  
  Top stacking preserved general ability but gained less. Domain gain plateaus at about 8 blocks (61.71 with 1 block, 65.91 with 8, 65.23 with 32) — [LLaMA Pro HTML](https://arxiv.org/html/2401.02415)
- Same-budget comparison against new experts (Qwen1.5-1.8B language expansion): LLaMA-Pro scored 41.10 expanded / 50.99 original vs MoE-LPR's 45.07 / 52.12. Both LLaMA-Pro and LoRAMoE add inference cost as parameters are added, while MoE-LPR keeps active parameters fixed — [MoE-LPR HTML](https://arxiv.org/html/2408.11396)
- SOLAR 10.7B depth up-scaling [Dec 2023]: duplicate the base layer stack, remove overlapping layers, then continue pretraining. The reported Instruct model beat Mixtral-8x7B-Instruct on its eval suite — [SOLAR](https://arxiv.org/abs/2312.15166)
- Branch-Train-Stitch [Jan 2025]: adds stitch layers that read frozen expert models' representations into a frozen seed model. This is a "new layer" route to composing specialists — [BTS](https://arxiv.org/abs/2502.00075)

### Inferences
- On Qwen3.6, a new block must also preserve the 3×GDN : 1×full-attention pattern and the MTP head's expectations. It needs its own MoE FFN (256×3.15M ≈ 0.8B for a full copy of a layer's experts) or a dense FFN. Each added block adds per-token latency on every decode step, including GDN recurrent-state memory for linear blocks and KV cache for full-attention blocks. This compounds the latency budget of a voice agent, whereas extra experts with top-k fixed add zero FLOPs.
- If depth is tried anyway:
  - Interleave identity blocks (copy of the preceding layer with the attention or GDN output projection and the MoE down-projections zeroed), train only the new blocks, and never prefix-stack.
  - The MTP head reads the last hidden state, so it may need re-tuning if a block is added near the top.
- "Adapters as new layers" (bottleneck adapters, stitch layers) are a middle ground. They have small latency and are freeze-friendly, but I found no 2025–2026 evidence that they beat expert-level LoRA on native MoEs.

### Gaps
- No published block-expansion or depth-up-scaling results on MoE bases, or on hybrid linear-attention (GDN/Mamba) stacks. In particular, I found no guidance for initializing GDN state parameters (A_log, dt_bias, conv) for identity behavior.
- I found no 2025–2026 study of "MoE-layer insertion", i.e., inserting a fresh sparse FFN layer into a trained MoE.

---

## 4. Router training and task/goal-conditioned routing (router-only FT, z-loss, aux-loss-free balancing in FT, R3 routing replay in RL, task-level routing, goal embeddings, routing steering)

### Takeaway
Evidence through Oct 2026 converges on three points. (a) Do not re-impose uniform load balancing in post-training: it flattens the inherited routing and hurts. (b) Either freeze the router or hold it softly near the base router. Router Prior Bias, an anchor toward the base router's logits, beat both LBL and unanchored FT by more than 14 points in-domain. Router-only tuning with a task-embedding regularizer (RoMA) targets a reported 10–20% gap to optimal routing. (c) For RL, Rollout Routing Replay (R3) is now the default for MoE stability and is wired into verl and slime. Goal or task-conditioned routing has real evidence for both specialization and efficiency:
- Task-level routing gave 1.9–2.6× peak throughput by extracting per-task sub-networks.
- Logit steering of behavior-linked experts shifted safety and faithfulness by up to +20–27% with no training.
- Hybrid token+task routing (HMoRA) generalized better than pure task routing.

The main risk is dense or naive router manipulation degrading utility.

### Cited Findings
**Load balancing and anchoring during post-training**
- Router Prior Bias (NAVER) [8 Sep 2026]: adds a soft training-time logit bias derived from the frozen base router, computed separately for the prompt and answer spans. Moonlight-16B-A3B math, in-domain: RPB 45.77, re-applied LBL 31.91, unanchored FT 29.44. Replicated on Qwen3-30B-A3B-Base. Weight-, logit- and output-distribution anchors work about equally well. A hard prior assignment preserves structure but hurts performance — [RPB](https://arxiv.org/abs/2609.08115)
- PEFT study [Oct 2026]: auxiliary loss 0 was best (68.5%), with monotone decline up to 58.6% at 1e-1 — [study](https://arxiv.org/html/2610.26370)
- Aux-loss-free balancing (DeepSeek) [Aug 2024]: a per-expert bias is added to routing scores for top-k selection only and nudged by each expert's load. It avoids interference gradients from an auxiliary loss — [Loss-Free Balancing](https://arxiv.org/abs/2408.15664)
- Qwen's global-batch LBL [Jan 2025]: computing LBL over the global batch instead of the micro-batch allows domain-level specialization and improves quality — [Global-batch LBL](https://arxiv.org/abs/2501.11873). Qwen3.6 config: router_aux_loss_coef=0.001 — [Qwen3.6 config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)
- Router z-loss (ST-MoE) [Feb 2022]: penalizes large router logits to stabilize sparse training without quality loss — [ST-MoE](https://arxiv.org/abs/2202.08906)
- DenseMixer [NeurIPS 2025]: a straight-through estimator gives the top-K router gradients from all experts. Forward needs all experts' outputs during training; inference is unchanged.
  - Reported +3.7 points GPQA-Diamond on Qwen3-30B MoE with 1k SFT examples.
  - The authors cite 1.46× theoretical FLOPs.
  - Pip package densemixer; works with LoRA.
  
  — [DenseMixer NeurIPS page](https://neurips.cc/virtual/2025/125501); [OpenReview PDF](https://openreview.net/pdf/8e0f963e9ac617ea12089c96b60be5d830ca21a8.pdf)
- Unsloth disables router-layer fine-tuning by default for Qwen3 and Qwen3.5 MoEs "for stability" — [Unsloth Qwen3.5 FT docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

**Router-only fine-tuning**
- RoMA [10 Nov 2025; ICLR 2026 per citing work]: fine-tunes only the routers (all else frozen) of OLMoE, DeepSeekMoE and Qwen3-MoE. A manifold regularizer pulls each sample's routing weights toward those of successful neighbours in a task-embedding space. Motivation: pretrained routers sit 10–20% below optimal routing accuracy. The abstract gives no per-benchmark numbers — [RoMA](https://arxiv.org/abs/2511.07419); [search listing](https://arxiv.org/pdf/2511.07419)
- C3PO [10 Apr 2025]: test-time re-mixing of core experts in critical layers using similar reference samples. +7–15% accuracy, and 1–3B-active MoEs then beat 7–9B dense LLMs — [C3PO](https://arxiv.org/abs/2504.07964)
- MoE-LPR stage 2 is router-only training on <1% replay and recovered original-language ability from 47.14 to 52.12 — [MoE-LPR HTML](https://arxiv.org/html/2408.11396)

**Routing replay in RL**
- R3 [Oct 2025]:
  - Problem on Qwen3-30B-A3B: about 10% of router decisions differ between SGLang inference and Megatron training, and 94% of tokens differ in at least one layer. Training–inference KL is 1.535e-3 vs 6.4e-4 for dense Qwen3-8B.
  - Fix: cache rollout routing masks and replay them in both old-policy recompute and update. The softmax is over training logits restricted to the replayed experts, so the router still gets gradients.
  
  — [R3 HTML](https://arxiv.org/html/2510.11370)
- R3 results: prevented GRPO collapse (runs collapsed at steps 60 and 105 without it). SWE-bench Verified 31.80 → 38.60 for multi-turn agents. Under 3% rollout overhead. Beat GSPO and TIS in its settings; combining with TIS gave no clear gain. Implemented in verl with Megatron and SGLang — [R3 HTML](https://arxiv.org/html/2510.11370)
- Applied to Qwen3.5-35B-A3B (Agent Lightning + verl + Megatron + vLLM, 4×B200, EP=4, R3 flags `actor_rollout_ref.rollout.enable_rollout_routing_replay=true`, `actor_rollout_ref.actor.megatron.router_replay.mode=R3`): SWE-bench Verified rose from 47.8% to 61.6% after 1,792 examples, then 60.6% at 2,816 [29 Sep 2026] — [Agent Lightning MoE example](https://microsoft.github.io/agent-lightning/latest/76-example-coding-agent-moe/)
- ESRL [11 Sep 2026]: deliberately perturbs routing for exploration. High-confidence experts are anchored, sampling is restricted to a candidate pool, perturbation strength scales with router entropy, and paths are replayed in training. +3.2 Pass@1 and +4.5 Pass@8 over GRPO on Qwen3-30B-A3B, using slime — [ESRL](https://arxiv.org/abs/2609.13058); [search listing](https://arxiv.org/pdf/2609.13058)

**Task/goal-conditioned routing**
- Task-MoE [Sep 2021, EMNLP Findings]: routing by task (language pair) instead of by token lets per-task sub-networks be extracted. +1.0 BLEU over token-MoE (32 experts, 30 pairs) and 1.9× peak inference throughput. At 200 pairs (128 experts, 13B) it is competitive with token-MoE at 2.6× throughput. Distilling a token-MoE kept only 32% of its gains — [Task-MoE](https://arxiv.org/abs/2110.03742)
- DEMix [Aug 2021]: domain-conditioned expert FFNs, one per domain, with domain labels at training time and mixing at test time — [DEMix](https://arxiv.org/abs/2108.05036)
- HMoRA [ICLR 2025]: pure task-level routing generalizes poorly to unseen tasks. A hierarchical token+task hybrid fixes this — [HMoRA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e43a33994a28f746dcfd53eb51ed3c2d-Abstract-Conference.html)
- FlexOlmo and Nexus condition router rows on domain embeddings (see Section 2). FlexOlmo found prompt-classifier routing (pick the expert by classifying the prompt) unstable: good only when the classifier picked the right expert — [FlexOlmo HTML](https://arxiv.org/html/2507.07024)
- SteerMoE [Sep 2025; ICLR 2026]: detects behavior-linked experts by activation-frequency risk difference on paired inputs, then at inference raises or lowers their router log-softmax scores. Up to +20% safety and +27% faithfulness across 11 benchmarks and 6 LLMs. Adversarial steering cut safety by 41%, and by 100% combined with jailbreaks. Code at adobe-research/SteerMoE — [SteerMoE (alphaXiv)](https://alphaxiv.org/abs/2509.09660)
- MASCing [30 Apr 2026]: an LSTM surrogate maps routing logits to behaviors and optimizes sparse steering masks over expert gates. Multi-turn jailbreak defense rose from 52.5% to 83.9%. The authors motivate sparse masks by noting dense router manipulation degrades utility — [MASCing](https://arxiv.org/abs/2604.27818)
- RoMA (above) explicitly ties routing to task embeddings. Safety-expert analysis [May 2026] finds routing in aligned MoEs is largely topic-driven — [search listing](https://arxiv.org/html/2605.29708v1)
- MoTE [Jun 2025]: instruction-to-dedicated-expert routing for multi-task embedding models; argues task-pure training data specializes better — [MoTE](https://arxiv.org/pdf/2506.17781)

### Inferences
- Router policy for the target model: keep the router frozen for SFT and distillation (as now), or train it under an RPB-style soft anchor (KL or logit bias to the frozen base router) with LBL at 0. Never re-enable the 0.001 LBL during post-training.
- If DPO or RL ever touches the router, or rollouts use vLLM/SGLang while training uses Megatron/HF, use R3. Even with a frozen router, numerical routing mismatch between engines exists (about 10% of decisions differ in R3's measurements), so R3 is worth enabling for decision-token DPO with on-policy samples too.
- Cheapest "goal-conditioned routing" design for voice agents:
  - Learn a per-goal (or per-task-slice) additive bias vector b_goal ∈ R^256 per layer (≈10K parameters per goal across 40 layers) on the frozen router logits.
  - Select the goal from the system prompt or agent state, much like aux-loss-free bias but task-indexed.
  - Train it with RPB-style anchoring on general data, or with SteerMoE-style promote/suppress sets derived from hot-expert statistics.
  
  This is my synthesis of the aux-loss-free bias, RPB and SteerMoE mechanisms; I found no paper that tests exactly this.
- Efficiency upside: if per-goal routing concentrates traffic on a subset (e.g., the ~97 hot experts), Task-MoE-style sub-network extraction or REAP-style per-deployment pruning could cut resident expert memory for single-purpose deployments. Expect MC-knowledge loss at 50% pruning (REAP).
- Risks:
  - Hard masks or "enforce absolutely" priors hurt (RPB hard-assignment variant, MASCing's dense-manipulation note, Task-MoE's poor generalization to unseen tasks per HMoRA).
  - Prefer soft biases with token-level routing kept intact.

### Gaps
- I found no paper that tests "routing tokens" (special prompt tokens whose sole job is to bias the router) on a production MoE. Nor did I find evidence that goal-conditioned routing on a native fine-grained MoE improves end-task quality beyond what router-frozen PEFT achieves.
- RoMA's and RPB's per-benchmark numbers and RPB's Qwen3-30B results were not visible in the abstracts.
- DenseMixer's 1.46× FLOPs figure was reported for its own test models. For a 256-expert top-8 model, computing all experts in forward is likely much more expensive (inference). I found no DenseMixer numbers on Qwen3.5/3.6.
- No published R3 results specifically for DPO (as opposed to GRPO-family RL).
- Not re-fetched this session: Loss-Free Balancing (2408.15664), Global-batch LBL (2501.11873), ST-MoE z-loss (2202.08906), DEMix (2108.05036), SOLAR (2312.15166) and DynMoE (2405.14297) are characterized from their abstracts as recalled. I give no quantitative claims for them.

---

## 5. Practicalities on hybrid Gated-DeltaNet MoEs (training/serving support, LoRA-on-experts serving, EP of extra experts, memory/latency, 2026 tooling)

### Takeaway
Training support for Qwen3.5/3.6 MoE is mature across Unsloth, ms-swift Megatron, Megatron-Bridge, slime and verl as of Oct 2026, including EP and R3. Serving LoRA on routed experts is the weak link:
- vLLM has fused-MoE LoRA (v0.12+) but only with the Triton experts backend. It asserted out on Qwen3.5-122B when FlashInfer was auto-selected.
- SGLang supports a specific "shared-outer" expert-LoRA layout.
- TensorRT-LLM requires the CUTLASS MoE backend.

Merging LoRA into expert weights (or UMoE-style in-place reallocation) avoids all of this. Adding experts changes num_experts and every EP and kernel layout downstream.

### Cited Findings
**Architecture facts (Qwen3.6-35B-A3B, config.json)**
- hidden 2048; 40 layers with full_attention_interval=4 (3 linear-attention GDN + 1 full attention, repeated).
- 256 experts, top-8, moe_intermediate_size 512, shared_expert_intermediate_size 512.
- router_aux_loss_coef 0.001, mtp_num_hidden_layers 1.
- GDN: 16 key heads × 128, 32 value heads × 128, conv kernel 4, mamba_ssm_dtype float32.
- Vocab 248,320; model_type qwen3_5_moe_text.

— [Qwen3.6-35B-A3B config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)

- Third-party architecture breakdown: about 3.15M parameters per expert, 806M expert parameters per layer, 25.2M active per layer. Softmax router, then top-8, renormalized. Shared expert gated by sigmoid(Linear(x,1)). Fused gate_up tensor [256, 1024, 2048] [29 Jun 2026] — [HF community blog](https://huggingface.co/blog/EXDai/qwen36-35b-a3b-architecture-overview). Release reported as 15 Apr 2026 by third-party aggregators; not verified on the official card — [vLLM recipes page](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)

**Training tooling**
- Unsloth (Qwen3.5 docs):
  - MoE kernels are claimed about 12× faster with more than 35% less VRAM for 35B-A3B, 122B-A10B and 397B-A17B.
  - bf16 LoRA on Qwen3.5-35B-A3B fits in 74 GB. MoE QLoRA (4-bit) is not recommended.
  - Router fine-tuning is off by default.
  - Custom Mamba (GDN) Triton kernels compile slowly on first run.
  
  — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- ms-swift Megatron-SWIFT (v4.1/v4.3 docs): lists Qwen3.5 with LoRA and MoE support for CPT, SFT, GRPO, DPO, KTO and RM, with TP/PP/SP/CP/ETP/EP — [Megatron-SWIFT quick start v4.3](https://swift.readthedocs.io/en/v4.3/Megatron-SWIFT/Quick-start.html)
- Megatron-Bridge 0.5.0: Qwen3.6-35B-A3B auto-detected as Qwen3_5MoeForConditionalGeneration; 1 node of 8 GPUs with EP=8 recommended; needs transformers ≥ 5.2.0 — [Megatron-Bridge Qwen3.5/3.6](https://docs.nvidia.com/nemo/megatron-bridge/0.5.0/models/qwen/qwen35-vl.html)
- Megatron-Bridge 0.6.0 LoRA options:
  - `share_expert_adapters` (default True) shares one adapter across local experts per EP rank. Set it to False for per-expert LoRA.
  - `normalize_moe_lora` uses rank dim//topk on expert layers.
  - `experts_shared_outer_loras` shares gate_up lora_A and down lora_B across experts, matching SGLang's `experts_shared_outer_loras=True` serving contract (SGLang PR #21466).
  
  — [Megatron-Bridge PEFT LoRA API](https://docs.nvidia.com/nemo/megatron-bridge/0.6.0/apidocs/bridge/bridge.peft.lora.html)
- slime: the README lists Qwen3.6, Qwen3.5, Qwen3-Next and Qwen3MoE support (per search summary) — [slime fork README](https://github.com/HJSang/slime). UMoE's Qwen3.5-35B-A3B experiments were built on slime — [UMoE HTML](https://arxiv.org/html/2607.11444)
- verl: R3 for Qwen3.5-35B-A3B with Megatron actor and vLLM rollout, EP=4 on 4×B200 with parameter, optimizer and gradient offload — [Agent Lightning MoE example](https://microsoft.github.io/agent-lightning/latest/76-example-coding-agent-moe/)
- PEFT ≥0.18 `target_parameters` can target batched or fused expert weights in Qwen3 MoE — [Friendli demo card](https://friendli.ai/models/tugot17/qwen3-30b-a3b-peft18-moe-lora-demo)

**Serving LoRA on experts**
- vLLM:
  - Fused-MoE LoRA layer classes (FusedMoE3DWithLoRA) appear in v0.12–v0.16 API docs — [vLLM v0.12 fused_moe LoRA](https://docs.vllm.ai/en/v0.12.0/api/vllm/lora/layers/fused_moe/)
  - On vLLM 0.17.0 [18 Mar 2026], Qwen3.5-122B LoRA failed on the assertion `fused_experts is TritonExperts` because FlashInfer experts were auto-selected. A reply points to `--moe-backend` to force Triton; replies conflict on whether that works — [vLLM forum](https://discuss.vllm.ai/t/lora-integration-for-qwen3-5-122b-fails-during-deployment-on-vllm-0-17-0/2466)
  - A per-expert Qwen3 MoE adapter layout reportedly loads in vLLM 0.19.0 when `experts` is added to target modules — [OpenPipe ART PR](https://github.com/OpenPipe/ART/pull/688)
- TensorRT-LLM: routed-expert LoRA modules `moe_gate`, `moe_h_to_4h` and `moe_4h_to_h` require the CUTLASS MoE backend — [TensorRT-LLM LoRA](https://nvidia.github.io/TensorRT-LLM/1.3.0rc27/features/lora.html)
- SGLang's general LoRA serving docs list `lora_paths`, `max_loras_per_batch` and `lora_backend` (triton or flashinfer) with no expert detail — [SGLang LoRA serving](https://docs.sglang.ai/backend/lora.html)

### Inferences
- Serving options, ranked by risk for the voice-agent product:
  1. Merge hot-expert and attention LoRA into the base weights and serve a plain checkpoint. This has zero runtime cost, but it is one model per adapter set.
  2. UMoE-style reallocation, again with a merged checkpoint and identical shapes.
  3. Multi-LoRA serving on vLLM with `--moe-backend triton`. This is slower than FlashInfer or CUTLASS experts on Blackwell or Hopper, and per-expert LoRA adds a grouped GEMM per routed expert.
  4. Added experts (num_experts 256 → 256+k). This needs config edits, re-sharding, and re-quantization (FP8/NVFP4 scales for the new experts).
- When adding experts, pick k so that 256+k divides cleanly by the EP degree (e.g., k=8 or 16 for EP=8). New task experts that receive concentrated goal-routed traffic will be EP hot-spots, so plan for expert replication or load balancing in the serving engine. I did not verify engine-level EPLB support for Qwen3.6 here.
- Memory budget: k new experts per layer adds k × 252 MB in bf16 across 40 layers. Top-8 fixed keeps decode FLOPs and latency unchanged apart from the extra expert weights in HBM. Raising top-k to 10 would add 25% routed-FFN FLOPs per token.
- GDN-specific: none of the methods in Sections 1–2 touch GDN state parameters, so they are architecture-agnostic. Block expansion (Section 3) would need GDN-aware identity init, which is undocumented. Unsloth's note on slow GDN kernel compilation and Megatron-Bridge's transformers ≥5.2 requirement are the only GDN-specific training caveats found.

### Gaps
- I could not confirm the exact vLLM release that made fused-MoE LoRA production-ready for Qwen3.5/3.6, nor whether FlashInfer or CUTLASS expert backends support LoRA as of v0.2x. SGLang PR #21466 details were not fetched.
- I found no published latency or throughput numbers for multi-LoRA-on-experts serving of Qwen3.5/3.6, nor for serving models with added experts.
- Official Qwen3.6 model card and release date: confirmed only via third-party pages and config.json.
- Whether ms-swift or Megatron-Bridge support selective (hot-subset) expert LoRA natively, as opposed to per-expert or shared adapters, was not confirmed. Megatron-Bridge wildcards apply to module names, not expert indices.

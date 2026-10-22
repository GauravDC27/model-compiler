# Numerical and System-Level Determinism of LLM Inference (dense + MoE), as of October 2026

Research date: 2026-10-21. Each finding is dated by the source's publication or last-update date where known. Findings from before about mid-2025 are flagged where later work may have superseded them.

Terms used throughout (they follow the assignment's distinction):
- **Bitwise determinism (same stack):** same model bytes, engine version, kernel set, GPU SKU, driver, and deployment config give byte-identical logits and tokens for the same request, whatever the batch composition, load, chunking or caching. This is the property Vosti formalizes as "bitwise-identical logits at corresponding output positions" ([Vosti, arXiv 2609.38981, 30 Sep 2026](https://arxiv.org/html/2609.38981v1)).
- **Cross-stack reproducibility:** byte-identical outputs across GPU SKUs or architectures, vendors, engine versions, or parallelism layouts (TP size, EP layout).
- **Semantic determinism:** same answer, accuracy or meaning, but not the same bytes. Tolerance-based verification (for example TOPLOC) sits between this and bitwise.

---

## Q1. What are the root causes of nondeterminism at temperature 0?

### Takeaway
The main cause on a fixed stack is not "GPU randomness" or atomics. It is a **lack of batch invariance (and, more generally, position/schedule invariance)**: kernels choose a different reduction strategy depending on batch size, chunk size, prefill vs decode path, KV layout or TP size, and floating-point non-associativity then turns that into different bits. Those bit differences flip argmax on low-margin tokens and then grow over long generations. MoE adds routing sensitivity (a near-tie in the router changes which experts run), historical capacity-factor token dropping (couples requests in a batch), non-deterministic top-k and finalize/unpermute kernels, and train/inference router mismatch. Cross-stack sources include library and kernel autotuning, SM count, GPU architecture, collective algorithm choice, and template or version drift.

### Cited Findings

**Floating point and batch invariance (the core mechanism)**
- (Sep 10, 2025) Thinking Machines Lab (Horace He et al.) argues that the common "concurrency + floating point" explanation is incomplete. Atomic adds are largely absent from LLM forward passes, so a single forward pass is run-to-run deterministic. The real cause is that server load changes the batch size, which changes each kernel's reduction order and therefore each request's numerics ("lack of batch invariance") — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- (Sep 10, 2025) The batch-dependent ops it identifies: **RMSNorm** (split reductions used at small batch), **matmul** (split-K at small M), and **attention** (split-KV / FlashDecoding, where the number of splits depends on batch and the KV/page-table layout). Data-parallel strategies (one batch element per core, no split-K) are invariant — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- (Sep 10, 2025) Measured effect: Qwen3-235B-A22B-Instruct-2507 (an MoE), temperature 0, prompt "Tell me about Richard Feynman", 1,000 completions of 1,000 tokens gave **80 unique completions**. The most common appeared 78 times. All were identical for 102 tokens and diverged at token 103 ("Queens, New York" in 992 runs, "New York City" in 8) — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- (Sep 2025, LMSYS) SGLang agrees: at temperature 0 the main source is varying batch size changing how kernels split reductions. Qwen3-8B in normal mode gave **2–18 unique outputs per 50 trials**, depending on backend and test — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/); [SGLang docs](https://docs.sglang.io/advanced_features/deterministic_inference.html)
- (Jan 2026) LLM-42's classification of default kernels in vLLM and SGLang:
  - **Not batch-invariant:** cuBLAS GEMM, ring-based AllReduce (which is also not position-invariant), RMSNorm, and fused RMSNorm+residual.
  - **Batch- and position-invariant:** FlashAttention-3 with `num_splits=1`, multimem AllReduce, and tree-based AllReduce with fixed NCCL settings.
  — [LLM-42, arXiv 2601.17768v2](https://arxiv.org/html/2601.17768v2)
- (Aug 2026) CoRun's key observation is that "most kernels are not batch-invariant, they are position-invariant", so fixed-shape (padded) execution is enough to be deterministic — [CoRun, arXiv 2608.14376](https://arxiv.org/abs/2608.14376)

**Scheduler and serving features (beyond batch size)**
- (Sep 30, 2026) Vosti lists the execution variations that must not change outputs: **batch composition, prompt chunking (chunked prefill), prefill vs decode execution, and KV-cache reuse/eviction/recomputation (prefix caching)** — [Vosti abstract](https://arxiv.org/abs/2609.38981)
- (Sep 30, 2026) Vosti shows that production deterministic modes still fail on some of these variations:
  - FA3 uses a batch-wide K-length bound that affects local-window (sliding-window) masking and kernel specialization.
  - SGLang decode CUDA graphs use wider KV page tables.
  - SGLang's Triton prefill and decode use different numerical algorithms. Triton passed only **3/384 prefill–decode pairs** on Llama-3.1-8B.
  - In vLLM's invariant Triton path, query partitioning changes which fully-masked softmax tiles are visited.
  - On Gemma3-4B (sliding window), **all four production deterministic configurations had mismatches in every category**.
  — [Vosti HTML](https://arxiv.org/html/2609.38981v1)
- (Sep 2025) Chunked prefill: SGLang's default "best-effort" chunking gives inconsistent truncation points. Deterministic mode aligns chunk boundaries to integer multiples of the split-KV size — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- (Sep 21, 2026) The TensorRT-LLM RFC lists the batch-dependent sources in TRT-LLM's PyTorch backend:
  1. Attention multi-block/split-KV, where the split count depends on SM count and batch×heads.
  2. FMHA library selection by batch bucket.
  3. Dense-GEMM tactic selection by M (autotuner), plus profiling noise when no cache is persisted.
  4. MoE grouped-GEMM tactic selection by token count.
  5. MoE finalize/unpermute reduction (CUTLASS, top-k>2), which is non-deterministic per its docstring.
  6. All-reduce strategy chosen by message size.
  7. The CUDA-graph batch-size padding ladder.
  8. Chunked-prefill and KV-reuse boundary placement.
  — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)

**Library, compiler and autotuning (cross-run and cross-build)**
- cuBLAS guarantees bitwise-identical results run to run only on GPUs with the **same architecture and same SM count**, within one toolkit version. "Bit-wise reproducibility is not guaranteed" across toolkit versions. The guarantee also fails with multiple concurrent streams unless each stream has its own workspace or handle, or `CUBLAS_WORKSPACE_CONFIG=:16:8` (may limit performance) or `:4096:8` (about 24 MiB more memory) is set. Atomics mode (`cublasSetAtomicsMode`) is explicitly not bit-reproducible — [cuBLAS docs §2.1.4](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)
- (Updated Jul 22, 2026) XLA on GPU is nondeterministic for scatter and select-and-scatter, which use atomics, and for autotuned GEMM, conv and attention kernels, where candidates differ in accumulation order. "Compiling the same HLO twice can yield different results" unless autotuning is persisted or disabled — [OpenXLA determinism](https://openxla.org/xla/determinism)
- TensorRT-LLM maintainers have said determinism across engine builds isn't possible because "kernels are picked by dynamic tuning results" (older issue, likely 2024; may be superseded) — [TRT-LLM issue #2474](https://github.com/NVIDIA/TensorRT-LLM/issues/2474)
- PyTorch: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms". CPU and GPU runs may differ even with identical seeds — [PyTorch 2.14 reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

**Hardware**
- (Jan 30, 2026) EigenAI measured **0% bitwise match between A100 and H100** (logit differences around 1e-7). Different hosts with the same GPU SKU matched 100%. They also traced some nondeterminism to **faulty memory, fixed by enabling ECC** — [EigenAI HTML](https://arxiv.org/html/2602.00182v1)
- (Sep 22, 2026) Cross-GPU divergence also comes from **hardware-dependent kernel selection** (different reduction orders and tensor-core arithmetic). Unmitigated BF16 greedy outputs differed across A100, L40S and H100 on **30.8–100% of problems** depending on model and task — [arXiv 2609.25624 HTML](https://arxiv.org/html/2609.25624v1)

**MoE-specific**
- (Aug 5, 2023; older, but the mechanism still holds for capacity-limited MoE) Sherman Chann's hypothesis: with sparse MoE, tokens from different requests compete for limited expert capacity, so a request's output depends on what else is in the batch. He quotes the Soft MoE paper: "the model is no longer deterministic at the sequence-level, but only at the batch-level". GPT-4 at temperature 0 gave about 11.67 unique completions per 30 runs versus 3.67 for gpt-3.5-turbo. Applying this to GPT-4 is unconfirmed speculation — [152334h blog](https://152334h.github.io/blog/non-determinism-in-gpt-4/)
- (Oct 2025; ICML 2026) MoE routers can select different experts across repeated forward passes "even under identical conditions", and routing differs between training and inference engines — [R3, arXiv 2510.11370](https://arxiv.org/abs/2510.11370); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/63177)
- (Sep 2026) About **10% of per-layer routing decisions differ** between engines on Qwen3-30B-A3B (R3 paper, as cited by the TRT-LLM RFC — secondary citation) — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)
- (Oct 25, 2025) SGLang notes that the DeepSeek-V3.2 custom top-k kernel (sparse attention indexer) is non-deterministic — [SGLang issue #10278](https://github.com/sgl-project/sglang/issues/10278)
- (Oct 14, 2026) The vLLM DeepSeek-V4-Flash work had to make DSA Top-K deterministic. It breaks ties by position using a unique 64-bit key, which adds about 5.9 µs per call — [vLLM blog, Oct 2026](https://vllm.ai/blog/2026-10-14-train-inference-consistency)
- (Sep 2026) MiniMax-M1 (via the TRT-LLM RFC) reports train/inference token-probability correlation of about 0.9 with a BF16 LM head and router, and about 0.99 with FP32 (secondary citation) — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)

**Precision and amplification**
- (Jun 2025; v2 Oct 24, 2025) BF16 greedy decoding of DeepSeek-R1-Distill-Qwen-7B:
  - Up to **9.15% std in AIME'24 accuracy** and **~9,000 tokens difference in response length** across 12 configurations (2 GPU types × 2 GPU counts × 3 batch sizes). Both are 0 under FP32.
  - On MATH500, over 90% of examples diverge under BF16.
  - The mean divergence index (first differing token) is about 65 tokens in BF16, about 395 in FP16, and about 1,826 in FP32, tracking mantissa width (7/10/23 bits).
  - The cause is small top-1/top-2 logit margins in reasoning models (example: 49.75% vs 46.65%).
  — [LayerCast paper, arXiv 2506.09501v2](https://arxiv.org/html/2506.09501v2)
- (May 28, 2026) Batch-induced token flips are rare per step: **0.3–1.3% of synchronous decode steps** across MATH500, GSM8K and HumanEval (Llama-3.1-8B: 0.48%). Low top-1/top-2 logit margin predicts much of the flip risk, and K/V perturbations stay flat until a flip happens — [MarginGate, arXiv 2605.30218](https://arxiv.org/abs/2605.30218)

**Sampling, templates and versions**
- (Sep 2025) Temperature>0 sampling with `torch.multinomial` is batch-dependent. SGLang replaced it with `multinomial_with_seed`, which uses Gumbel noise from a seeded hash, so the same inputs and seed always give the same sample — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- (Sep 2026) Template drift: the 2609.25624 authors had to pin a date string in the Llama chat template, which otherwise caused artifactual divergence — [arXiv 2609.25624 HTML](https://arxiv.org/html/2609.25624v1)
- (Archived OpenAI cookbook, about late 2023) Even with the same `seed`, parameters and `system_fingerprint`, "There is a small chance that responses differ". The fingerprint changes when OpenAI updates serving infrastructure — [OpenAI cookbook](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter)
- (Aug 2024 – Apr 2025; older) Five "deterministic" LLM configurations over 10 runs showed accuracy variation up to 15% and a best–worst gap up to 70%. No LLM gave repeatable accuracy across all tasks — [Atil et al., arXiv 2408.04667](https://arxiv.org/abs/2408.04667)

### Inferences
- A "model compiler" must treat determinism as a **whole-system invariant**, not a kernel property. Batch composition, chunk boundaries, prefill/decode path, cache hit or miss, preemption/recompute, CUDA-graph padding, autotuning, collective algorithm, and template or version bytes all have to be pinned or made invariant. Vosti's results show that even "deterministic modes" leak through sliding-window masking and prefill/decode algorithm differences.
- Because flips are rare per step (0.3–1.3%) but compound (divergence often within the first ~65–100 tokens in BF16), a long-generation system without determinism will almost always diverge at the sequence level. Reasoning and agentic workloads are where this matters most.
- For MoE, the dominant extra risks are: (a) router near-ties amplifying tiny upstream differences into a different expert set, (b) capacity-based dropping coupling requests (avoid by using dropless routing), (c) non-deterministic top-k, permute/unpermute and finalize kernels, and (d) cross-rank combine order under EP/TP.

### Gaps
- I found no primary NCCL documentation stating which algorithm/protocol combinations (Ring/Tree/NVLS/CollNet/PAT, LL/LL128/Simple) give identical bits for a given rank count. The NCCL env-var page I fetched says nothing on determinism ([NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)). The SGLang/slime recipe uses `NCCL_ALGO=Ring` for training reproducibility ([LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)).
- I found no primary measurement of how often current dropless MoE serving (DeepSeek/Qwen-style) actually drops or reroutes tokens because of batch coupling. Chann's 2023 capacity-factor analysis applies to capacity-limited designs.
- Tokenizer-version drift was not researched beyond the chat-template example. The web-search budget ran out.

---

## Q2. Thinking Machines Lab "Defeating Nondeterminism in LLM Inference" and batch_invariant_ops: what exactly was done, what was the slowdown, and how was it adopted?

### Takeaway
On Sep 10, 2025, Horace He and colleagues at Thinking Machines showed that batch invariance, not concurrency, is the cause. They made RMSNorm, matmul and attention batch-invariant: data-parallel reductions with no split-K, and **fixed split size rather than a fixed number of splits** for split-KV decode. Swapped into vLLM via `torch.Library`, this gave 1,000/1,000 identical completions at a 1.6–2.1× slowdown. It also enabled "true on-policy" RL with KL exactly 0. Within weeks it was adopted by SGLang (Sep 22, 2025) and vLLM (`VLLM_BATCH_INVARIANT=1`, Oct 22, 2025), and it has become the reference design that 2026 papers build on or argue against.

### Cited Findings
- **Author and date:** Horace He with others at Thinking Machines Lab, Sep 10, 2025 — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **RMSNorm:** use a data-parallel strategy (one batch element per core). Use one reduction strategy with enough parallelism for all batch sizes, rather than switching to split reductions at small batch — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **Matmul:** data-parallel tiling with no split-K. The simplest fix is to compile one kernel configuration for all shapes, which costs about 20% versus cuBLAS — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **Attention:** FA2-style parallelism over queries. Update the KV cache and page table before the attention kernel so K/V layout is consistent between prefill and decode. For decode split-KV, use a **fixed split size**: for example, a 1,000-length KV becomes 3×256 + 1×232, so the reduction order depends only on position. The FlexAttention changes needed were not released — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **Implementation:** built on vLLM's FlexAttention backend, using `torch.Library` to substitute PyTorch operators — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **Library contents:** `batch_invariant_ops` (MIT license) provides batch-invariant `torch.mm`, `torch.addmm`, `torch.log_softmax` and `torch.mean`, enabled with a `set_batch_invariant_mode()` context manager. The vLLM demo (`deterministic_vllm_inference.py`) needs a small upstream vLLM change. Without it, 1,000 completions of length 100 gave 18 unique outputs; with it, all were identical. At fetch time the repo had about 1.1k stars and only 7 commits — [GitHub batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops)
- **Performance** (Qwen3-8B, 1 GPU, 1,000 sequences, 90–110 output tokens):

  | Configuration | Time | Slowdown |
  |---|---|---|
  | vLLM default | 26 s | 1.0× |
  | Unoptimized deterministic | 55 s | ~2.1× |
  | With improved attention kernel | 42 s | ~1.6× |

  — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **RL result** (RLVR on BigMath, Qwen2.5-VL-8B-Instruct init, 4,096 max rollout length):
  - Without importance weighting, reward collapsed (loss spike around step 318).
  - With importance weighting, KL stayed around 0.001 with spikes.
  - With bitwise-identical sampler and trainer ("True On-Policy RL"), KL stayed at exactly 0.
  — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

**Adoption**
- SGLang integrated the TML kernels on Sep 22, 2025 — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- vLLM shipped `VLLM_BATCH_INVARIANT=1` on Oct 22, 2025, credited to @bwasti — [vLLM on X](https://x.com/vllm_project/status/1981088861506982041); [vLLM on X (2)](https://x.com/vllm_project/status/1981088876082188450)
- The vLLM + TorchTitan bitwise-consistent RL work followed on Nov 10, 2025 — [vLLM blog](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference)
- TML's 61.5% overhead (42 s / 26 s) is the baseline SGLang compares itself against — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)

### Inferences
- The core recipe has three parts: (1) choose reduction strategy and tile config independent of runtime M, batch, chunk or position; (2) no split-K, or split-K with fixed chunk size; (3) fixed split size for split-KV. It is simple enough to re-implement per kernel family. For a model compiler, this suggests a **"shape-class-pinned kernel configuration" pass**: tile/K-loop chosen from (N, K, dtype) only, never from runtime M.
- TML's numbers (Sep 2025) are now superseded by optimized implementations (see Q3). Treat 1.6–2.1× as an upper bound for naive implementations, not the current cost.

### Gaps
- The FlexAttention fixed-split-size changes were not released, so TML's 42 s configuration is not reproducible from public code (as of the blog).
- I found no TML follow-up publication with updated numbers in 2026.

---

## Q3. Current state (2026) of deterministic / batch-invariant modes in vLLM, SGLang, TensorRT-LLM, llama.cpp, HF, PyTorch, JAX/XLA and vendors

### Takeaway
As of October 2026, **vLLM is the most mature**. `VLLM_BATCH_INVARIANT=1` is still labeled beta, but covers NVIDIA SM80+, AMD gfx90a/942/950, Intel XPU and Ascend. It validates many MoE models (DeepSeek-V3/R1/V3.1, Qwen3-30B-A3B, GPT-OSS, Granite MoE), TP with a pinned custom all-reduce, and DP+EP on GB300. A separate Oct 2026 vLLM path achieves bitwise rollout/training parity on DeepSeek-V4-Flash with DP8/EP8 at about 91% of baseline throughput. **SGLang** offers `--enable-deterministic-inference` (FA3, FlashInfer, Triton) with about 24–46% overhead, MoE examples, and deterministic seeded sampling; EP is still pending. **TensorRT-LLM** has only a partial `FORCE_DETERMINISTIC`, with an open RFC (Sep 2026) for a full mode. **llama.cpp, HF TGI/transformers and JAX/TPU** have no batch-invariant serving mode that I found. PyTorch, cuBLAS and XLA give same-stack determinism controls only.

### Cited Findings

#### vLLM (`VLLM_BATCH_INVARIANT=1`)
- **Status and enabling:** beta; tracking issue #27433. Set `VLLM_BATCH_INVARIANT=1` for `vllm serve`, or via `os.environ` before importing vLLM. Opt out of breakable CUDA graphs with `VLLM_USE_BREAKABLE_CUDAGRAPH=0`. On XPU, `--attention-config.backend TRITON_ATTN` is required — [vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance/)
- **Hardware:** NVIDIA compute capability ≥8.0; AMD Instinct gfx90a, gfx942, gfx950; Intel XPU with Triton. Ascend requires setting `VLLM_BATCH_INVARIANT=1` *before building* vllm-ascend so a custom op library is installed — [vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance/); [vLLM Ascend docs](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/batch_invariance.html)
  - Flag: one search summary of an older docs version said the requirement was compute capability ≥9.0. The current page says ≥8.0, so the older statement is superseded.
- **Tested MoE models:**
  - DeepSeek-V3, V3-0324, R1, V3.1
  - Qwen3-30B-A3B, Qwen3-Next-80B-A3B-Instruct, Qwen3-30B-A3B-Thinking-2507-FP8
  - GPT-OSS-20b and 120b
  - Granite 3.1 1b-a400m and 3b-a800m, and granite-swash-3b-a600m (SWA MoE)
  - Mixtral is not listed. Many dense models are also listed (Qwen3, Qwen2.5, Llama 3.1/3.2, Mistral-7B, GLM-4-9B, Phi, Gemma 2, EXAONE 4.0, OLMo 2, and others), including AWQ and FP8 variants.
  — [vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance/)
- **Mechanisms:**
  - Deterministic attention kernels: FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION.
  - On Ada/Hopper/Blackwell with tuned matmul entries, BF16 unquantized linear layers skip `torch.compile` and use breakable CUDA graphs "so tile configs follow batch size".
  - LoRA shrink is split into 8 fixed K chunks summed in fixed order instead of with atomics.
  - **TP:** custom all-reduce stays on with a fixed reduction order (the 1-stage kernel is pinned; large inputs are reduced in fixed-size chunks). FlashInfer, AITER and QuickReduce all-reduce are disabled.
  - **Limitation:** all-reduces whose size isn't a multiple of 16 bytes are unsupported, because they can switch between NCCL and custom AR with batch size.
  — [vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance/)
- **FlashInfer backend internals:** the vLLM FlashInfer backend has a batch-invariant workspace constant (2048×1024×1024), and `plan()` accepts `prefill_fixed_split_size` and `disable_split_kv` — [vLLM FlashInfer backend API](https://docs.vllm.ai/en/v0.12.0/api/vllm/v1/attention/backends/flashinfer/)
- **Roadmap status** (issue #27433; opened Oct 23, 2025; comments through Oct 20, 2026):
  - DeepSeek-V3 on 8×H100 (#26609), DeepGEMM on Blackwell (#27127), R1 TP8 on Blackwell (#27229), TRITON_MLA (#29125), and CI tests (#27842) are listed.
  - torch.compile and CUDA-graph support is tracked in #27660.
  - FLASHINFER_MLA is pending ("help needed").
  - **DP+EP** was originally out of scope. #45683 (merged) made the **MoE combine deterministic**, and **DP=2×TP=2 with EP was validated on GB300**.
  - **AMD** (#52231): CI green on MI250/MI300/MI355, including TP=4 and DP=4 with EP (Oct 9, 2026), still under review.
  - **Speculative decoding:** GLM-4.5-Air-FP8 with MTP was bitwise equal on B300 (Oct 9, 2026; #52522).
  - **Prefix caching:** no stated support. On Sep 27, 2026 the maintainer said it "might work in some certain cases".
  - **Known gaps:** GPTQ/AWQ MoE experts on Marlin; FA4 on Blackwell forced to FA2; Qwen3-Next stopped starting in BI mode after #40193 (reported Oct 4, 2026); chunked-prefill TRITON_MLA fix pending (#60122).
  — [vLLM issue #27433](https://github.com/vllm-project/vllm/issues/27433)
- **Performance PRs on the BI path** (as reported in #27433):

  | PR | Change | Reported gain |
  |---|---|---|
  | #29345 | BMM | +18.1% throughput |
  | #40408 | CUTLASS FP8 | −28.9% E2E latency |
  | #40413 | Fused RMSNorm | −2.1% E2E latency |
  | #53247 | Per-arch persistent matmul | ~3× faster decode kernels |
  | #53649 | Blackwell Triton autotune | −33.6% E2E latency |
  | #39912, #40372 | NVFP4 linear / MoE via CUTLASS | not stated |

  The docs give **no official overhead number**; they say only that performance may drop as an "intentional trade-off" — [vLLM issue #27433](https://github.com/vllm-project/vllm/issues/27433); [vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance/)
- **DeepSeek-V4-Flash train/inference consistency path** (Oct 14, 2026; Yan Bai of NVIDIA and Ao Shen of Inferact):
  - Model: block-FP8 MoE with MLA, DSA sparse attention, 43 MoE layers.
  - Layouts: rollout uses 4 vLLM engines at DP8/EP8, TP=1; training uses EP32/CP4; 8 GB200 machines (32 GPUs).
  - **DeepGEMM batch-invariant mode** for dense and expert GEMMs (no switch to FlashInfer below M=32). The router gate uses a single FP32-output kernel.
  - FP8 KV cache kept, with decode routed through the prefill sparse-MLA read path so both phases share one algorithm.
  - Deterministic Top-K with a position tie-break.
  - **Expert merge done locally in FP32 in top-k slot order.**
  - TP>1 is rejected at startup.
  - CUDA graph FULL_AND_PIECEWISE is made bitwise equal to eager.
  - Result: KL and max diff 0 at every step for 100 steps, at **91.3% of baseline rollout throughput** (24.22 vs 26.53 tok/ms). On a pure inference benchmark, after transport matching, the consistent build was 3.2% *faster* (16,834 vs 16,313 tok/s).
  - Not supported: speculative decoding/MTP, TP>1, decode CP.
  — [vLLM blog, Oct 2026](https://vllm.ai/blog/2026-10-14-train-inference-consistency)

#### SGLang (`--enable-deterministic-inference`)
- **Flag and backend:** `--enable-deterministic-inference` with `--attention-backend` `fa3` (default), `flashinfer` or `triton`.

  | Backend | CUDA graph | Chunked prefill | Radix cache | Temp > 0 |
  |---|---|---|---|---|
  | FA3 | Yes | Yes | Yes | Yes |
  | Triton | Yes | Yes | Yes | Yes |
  | FlashInfer | Yes | Yes | **No** | Yes |

  Other docs details:
  - Default sampling seed is 42; per-request `sampling_seed` is supported.
  - Test tool: `python3 -m sglang.test.test_deterministic --test-mode single|prefix|radix_cache`.
  - Qwen3-30B-A3B (MoE) on FA3 is a documented example.
  — [SGLang docs](https://docs.sglang.io/advanced_features/deterministic_inference.html)
- **Mechanisms** (Sep 22, 2025):
  - TML batch-invariant mean, log-softmax and matmul.
  - FlashInfer uses `fixed_split_size` and `disable_kv_split`.
  - FA3 fixes num-splits to 1.
  - Triton uses a fixed decode split size and also runs on AMD.
  - Chunked prefill is aligned to split-KV multiples.
  — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- **Overhead** (H200, end-to-end, for 1024/4096/8192 in/out lengths):

  | Backend | 1024 | 4096 | 8192 |
  |---|---|---|---|
  | FlashInfer | +42.6% | +46.0% | +24.4% |
  | FA3 | +27.2% | +30.2% | +35.7% |
  | Triton | +55.1% | +44.6% | +44.8% |

  FA3/FlashInfer average **34.35%**, versus TML's 61.5%. CUDA graphs gave about 2.8× speedup in deterministic mode. The target is under 20% overhead — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- **Superseded statement:** the Sep 2025 blog said "dense models only; MoE planned" and that TP1/TP2 were deterministic while larger TP needed new reduce kernels. Later docs show a Qwen3-30B-A3B example and the roadmap lists DeepSeek-V3 (#12095, 8×H200), a large-TP deterministic all-reduce (#10930), an AMD all-reduce (#15340) and blockwise FP8 (#11491) — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/); [SGLang docs](https://docs.sglang.io/advanced_features/deterministic_inference.html); [SGLang issue #10278](https://github.com/sgl-project/sglang/issues/10278)
- **Roadmap gaps** (#10278; closed Feb 21, 2026):
  - **Pending:** expert parallel (no PR), FP8 fused MoE, NVFP4 GEMM/MoE, per-channel FP8, speculative decoding, Blackwell TP4 bug (#11513).
  - **FlashInfer radix cache:** reverted (#11728) after non-determinism was detected.
  - **User report:** DeepSeek-V3.1 on 8×H200 dropped from about 100 to 60 tok/s with deterministic mode, FA3 and EAGLE. The cause is confounded by EAGLE.
  — [SGLang issue #10278](https://github.com/sgl-project/sglang/issues/10278)
- **Independent overhead measurement** (Jan 2026): SGLang-Deterministic throughput is **24% (in=2048, out=512) to 36% (ShareGPT) lower** offline. Mixing one deterministic request into 10 non-deterministic ones drops throughput about 56% (845 to about 415 tok/s), because the mode is server-wide. Batch-invariant GEMM peaks at 194 TFLOPS versus cuBLAS 527 TFLOPS on H100 (−63%). Triton RMSNorm is up to 50% slower than the fused CUDA kernel — [LLM-42 HTML](https://arxiv.org/html/2601.17768v2)
- **Bug reports** (from search listings; dates not verified): the flag does not make hybrid Mamba2 granite-4.0-h batch-invariant ([#43413](https://github.com/sgl-project/sglang/issues/43413)), and it crashes at startup on L40S with Triton shared-memory OutOfResources ([#29149](https://github.com/sgl-project/sglang/issues/29149))
- **Vosti finding** (Sep 2026): SGLang Det. with Triton matched only 3/384 prefill–decode pairs and 13/14 prefix pairs on Llama-3.1-8B. FA3 matched all pairs on Llama, but all modes failed on Gemma3-4B — [Vosti HTML](https://arxiv.org/html/2609.38981v1)

#### TensorRT-LLM
- There is no complete deterministic mode. `FORCE_DETERMINISTIC=1` covers attention multi-block/split-KV, and chunked prefill/KV reuse are disabled under it. However:
  - A shared-memory guard can silently re-enable multi-block.
  - `getBoolEnv` only matches the exact string `"1"`, so `=true` is ignored.
  - Not covered: FMHA library selection (pin manually with `TLLM_FMHA_LIBS`), GEMM autotuner tactics (`enable_autotuner=False` falls back to `tactic=-1`), MoE grouped-GEMM tactics, all-reduce strategy, and CUDA-graph padding.
  - MoE finalize is fixable only with `disable_finalize_fusion=True` (CUTLASS path); TRTLLM-Gen has no unfused path.
  — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)
- **RFC #19501 plan** (opened Sep 21, 2026 by a community contributor; no NVIDIA staff reply shown):
  - Phase 1: `enable_deterministic_inference` flag plus Router Replay; "modest" overhead.
  - Phase 2: FP32 `lm_head_dtype` and `router_dtype`, and sampling-mask replay.
  - Phase 3: fixed-split-size attention, no-split-K dense GEMM, fixed-order MoE finalize; expected overhead about 20–35%.
  - Backlog: TP-size invariance.
  — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)
- **Older guidance** (likely 2024; probably superseded): run with `max_batch_size=1` for deterministic output — [TRT-LLM issue #2474](https://github.com/NVIDIA/TensorRT-LLM/issues/2474)

#### llama.cpp
- I found no batch-invariance flag. Maintainer ggerganov noted "subtle effects that lead to differences when processing the same input with different batch sizes" (see #7745) and suspected CUDA kernels and numerical stability — [llama.cpp PR #11684](https://github.com/ggml-org/llama.cpp/pull/11684) (from search snippet)
- EigenAI built its deterministic engine on llama.cpp, inheriting its Q4/Q5 quantized matmuls and adding deterministic GEMM/reduction kernels (see Q5) — [EigenAI HTML](https://arxiv.org/html/2602.00182v1)

#### PyTorch
- `torch.use_deterministic_algorithms(True)` selects deterministic implementations and raises on ops without one (for example CUDA `index_add_`).
  - `warn_only=True` lets SDPA Flash/Efficient/cuDNN backward run non-deterministically with a warning. With `warn_only=False`, Flash/Efficient backward is deterministic and cuDNN attention is disabled; the MATH backend is always deterministic.
  - `torch.backends.cudnn.deterministic` affects only cuDNN convolutions. Set `cudnn.benchmark=False`.
  - `torch.utils.deterministic.fill_uninitialized_memory` defaults to True under deterministic mode.
  - No guarantees across releases, commits, platforms, or CPU vs GPU.
  — [PyTorch 2.14 notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
- **`CUBLAS_WORKSPACE_CONFIG`:** `:16:8` (may limit performance) or `:4096:8` (about 24 MiB extra) is needed for cuBLAS reproducibility with multiple streams — [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)
- **Training-side recipe used with SGLang/slime:**
  - FA2 instead of FA3 for deterministic backward
  - Megatron `--deterministic-mode`
  - `NCCL_ALGO=Ring`
  - `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  - `torch.use_deterministic_algorithms(True, warn_only=False)`

  Two Qwen3-8B training runs gave identical curves — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- **These flags give run-to-run determinism for a fixed shape. They do not give batch invariance.** On CPU, FP32 logprobs differed between batch 1 and batch 2 on about 54.5% of vocab entries (max 7.6e-6) — community comment, Oct 2026 — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)

#### JAX/XLA
- **GPU:** `--xla_gpu_exclude_nondeterministic_ops` filters out non-deterministic algorithm variants, rewrites select-and-scatter, and disables live autotuning. `--xla_gpu_require_complete_aot_autotune_results` or `--xla_gpu_autotune_level=0` pin autotuning. The page has no TPU statements (last updated Jul 22, 2026) — [OpenXLA determinism](https://openxla.org/xla/determinism)
- **TPU RNG:** the newer hardware RNG path is deterministic only for a given sharding. This came from a search-result summary of a Flax commit and was not verified first-hand — [Flax commit via semanticdiff](https://app.semanticdiff.com/gh/google/flax/commit/7b20ed2d3bcd6679306fee8fb0cb5790b344c3c8)

#### Hardware vendors
- **NVIDIA:** cuBLAS same-architecture, same-SM-count guarantee ([cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)). An NVIDIA DevTech engineer co-authored the vLLM V4-Flash bitwise work using DeepGEMM batch-invariant mode ([vLLM blog, Oct 2026](https://vllm.ai/blog/2026-10-14-train-inference-consistency)).
- **AMD:** vLLM BI supports gfx90a/942/950, and the AMD TP/EP PR has green CI ([vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance/); [vLLM issue #27433](https://github.com/vllm-project/vllm/issues/27433)). SGLang Triton deterministic attention runs on AMD, and SGLang has an AMD deterministic all-reduce (#15340) ([LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/); [SGLang issue #10278](https://github.com/sgl-project/sglang/issues/10278)).
- **Huawei Ascend:** vllm-ascend has a batch-invariance custom op library ([vLLM Ascend docs](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/batch_invariance.html)).

### Inferences
- **Practical 2026 overhead bands** for "deterministic mode on" versus default, same hardware:
  - Generic batch-invariant kernels: roughly **+20–60% latency, −24–36% throughput**. Pathological cases reach about 2× latency and up to 74% throughput loss (per CoRun's characterization; see Q4).
  - Carefully co-designed MoE stack (vLLM DeepSeek-V4-Flash): **about 9% throughput cost**, and roughly 0% on a tuned inference benchmark.
  - The gap is mostly kernel engineering (DeepGEMM/CUTLASS batch-invariant variants, CUDA graphs), not a fundamental cost.
- **MoE determinism recipe common to vLLM, SGLang and TRT-LLM:**
  - Batch-invariant grouped GEMM (no token-count-dependent tactic)
  - FP32 router GEMM with one kernel
  - Deterministic top-k tie-break
  - Fixed-order, FP32 local expert combine (no atomics in unpermute/finalize)
  - EP all-to-all used only for data movement, never reduction
  - TP=1 per expert, or a pinned-order all-reduce
- Server-wide modes (SGLang, vLLM) charge every request for determinism. Per-request determinism (LLM-42, MarginGate) is an architectural alternative a model compiler could expose as a "determinism SLA" flag.

### Gaps
- **HF transformers/TGI:** the web-search budget ran out before I could verify any 2026 deterministic or batch-invariant mode. I found no evidence of one.
- **llama.cpp:** no 2026 primary source found on determinism flags.
- **vLLM:** no official end-to-end overhead benchmark for `VLLM_BATCH_INVARIANT=1` in the docs.
- **SGLang:** current (Oct 2026) EP status and MoE overhead not verified.
- **TPU:** no source on reduction determinism, and Google publishes nothing on TPU batch-invariant serving that I found.

---

## Q4. Research on reproducibility of reasoning (LayerCast and follow-ups), divergence amplification, and 2025–2026 systems papers

### Takeaway
LayerCast (Jun 2025) showed that BF16 reasoning evaluations are not reproducible across GPU type, count and batch size (up to 9% accuracy std, about 9k-token length differences) and that FP32 compute fixes most of it. Follow-up work splits into four strands:
1. Numerics: FP32 compute (LayerCast), fixed-config fused-upcast GEMMs (2609.25624), power-of-two INT8 scales (2609.00363).
2. Kernel invariance: TBIK for TP-size invariance (Nov 2025, ICML 2026).
3. Scheduling instead of kernel invariance: LLM-42 verified speculation (Jan 2026), MarginGate (May 2026), CoRun padding (Aug 2026).
4. Formal verification: Vosti (Sep 2026), which found that production deterministic modes still leak.

### Cited Findings

**LayerCast ("Give Me FP32 or Give Me Death?", Yuan et al.)**
- Setup: DeepSeek-R1-Distill-Qwen-7B and -Llama-8B, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct; AIME'24, MATH500, LiveCodeBench. 12 configurations: A100 vs L40S, 2 vs 4 GPUs, batch 8/16/32 — [arXiv 2506.09501v2](https://arxiv.org/html/2506.09501v2)
- BF16 AIME'24 accuracy std (all 0 under FP32):

  | Model | BF16 std |
  |---|---|
  | R1-Distill-Qwen-7B | 9.15% |
  | R1-Distill-Llama-8B | 4.60% |
  | Llama-3.1-8B | 1.92% |
  | Qwen2.5-7B | 1.71% |

  Length std is about 9,190 tokens for R1-Distill-Qwen-7B (FP16: about 5,990) — [arXiv 2506.09501v2](https://arxiv.org/html/2506.09501v2)
- Under BF16 greedy decoding, over 90% of MATH500 examples diverge. The FP32 divergence rate is reported as 2.2% in the text but 5.8% in Table 9 (internal inconsistency) — [arXiv 2506.09501v2](https://arxiv.org/html/2506.09501v2)
- More GPUs (4 vs 2) generally add variation, and smaller batch adds more. A100 varies slightly more than L40S, but this is not consistent — [arXiv 2506.09501v2](https://arxiv.org/html/2506.09501v2)
- **LayerCast method:** store weights in BF16, upcast each to FP32 just before its matmul, and compute everything in FP32. It uses 34% less memory than full FP32 and matches FP32 stability (for example, 0% AIME std on R1-Distill-Llama-8B versus 5.87% A100 and 3.44% L40S for BF16). Its divergence rate is below 3.4%. It is released as a vLLM patch. No speed cost was measured, but FP32 roughly doubles inference time versus BF16 — [arXiv 2506.09501v2](https://arxiv.org/html/2506.09501v2)
- **Recommendations:** report the mean ± error bars over multiple sampled runs; for a single greedy run, use FP32 — [arXiv 2506.09501v2](https://arxiv.org/html/2506.09501v2)
- **Code:** [github.com/nanomaoli/llm_reproducibility](https://github.com/nanomaoli/llm_reproducibility) (per [TBIK abstract](https://arxiv.org/abs/2511.17826))

**TBIK — Tree-Based Invariant Kernels** (Zhang, Ding, Yuan et al.; v1 Nov 21, 2025, v2 May 29, 2026; ICML 2026)
- **Method:** TP-invariant matmul and reduction that align intra-GPU and inter-GPU reduction orders with one hierarchical binary tree. It uses a custom tree all-reduce (reduce up the tree, then broadcast), not NCCL's tree, which uses chain reduction inside a node. Written in Triton and integrated into vLLM and FSDP — [arXiv 2511.17826](https://arxiv.org/abs/2511.17826); [HTML](https://arxiv.org/html/2511.17826); [ICML 2026](https://icml.cc/virtual/2026/poster/66224)
- **Results:** on Qwen3-8B, AIME24, L40S, average unique outputs were 12.00 (BF16), 7.87 (batch-invariant ops alone), and **1** (BIO+TBIK). Max probability divergence across TP 1/2/4/8 was 0 for all four models — [TBIK HTML](https://arxiv.org/html/2511.17826)
- **Cost:**
  - Tree MatMul reaches about 63% of cuBLAS at large batch (about 120 vs 190 TFLOPS BF16).
  - End-to-end on Qwen3-8B, TP=4, H20: **BIO +10–33%; BIO+TBIK +22–63% over BF16**. Tree MatMul accounts for 2–25% and tree all-reduce up to about 10%.
  - Required eager mode, with no CUDA graphs, prefix caching or chunked prefill.
  - MoE and quantization not tested.
  — [TBIK HTML](https://arxiv.org/html/2511.17826)

**LLM-42** (Gond, Kamath, Ramjee, Panwar; code under github.com/microsoft; Jan 25, 2026; ACM publication listed)
- **Method:** decode on the fast non-deterministic path, then a verifier replays candidate tokens under a fixed-shape reduction schedule. Consistent tokens are committed and the rest rolled back. Existing kernels are reused — [arXiv 2601.17768](https://arxiv.org/abs/2601.17768); [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3830418.3843901)
- **Results** (Llama-3.1-8B, 4×H100 PCIe):
  - With 1 of 11 requests deterministic: 911 tok/s, 2.2× SGLang-Deterministic and within 3% of non-deterministic mode.
  - At 100% deterministic traffic, recomputed tokens were 10.97% (ArXiv dataset) and 0.32% (ShareGPT).
  - TTFT P90 101 ms vs 172 ms for SGLang-Det.
  - Correctness checked on Qwen3-30B-A3B (MoE) with TP. Multi-GPU performance was not evaluated.
  — [LLM-42 HTML](https://arxiv.org/html/2601.17768v2)

**MarginGate** (May 28, 2026)
- **Method:** verify only low top-1/top-2-margin decode steps and repair a mismatch by replacing the current K/V column.
- **Results:** 100% sequence-level determinism on Llama-3.1-8B and Qwen2.5-14B with verifier trigger rates of 18.56% and 15.05%, cutting latency overhead 2.23× and 1.99× versus always-on (LLM-42-style) verification. DSR1-Distill-Qwen-7B needs a 49.5% trigger rate.
- [arXiv 2605.30218](https://arxiv.org/abs/2605.30218)

**CoRun** (Aug 14, 2026)
- **Method:** isolated prefill plus fixed-shape padded decode with CUDA graphs.
- **Characterization of batch-invariant kernels:** they increase latency by more than 2× and reduce serving throughput by up to 74%.
- **Results:** 15–324% higher throughput than batch-invariant approaches, with 51.8% lower TTFT and 48.6% lower TPOT on average. Tested on Qwen and DeepSeek.
- [arXiv 2608.14376](https://arxiv.org/abs/2608.14376)

**Vosti** (Sep 30, 2026; Qin, Du, Zhang, Lentz, Zhuo)
- **Formal spec:** bitwise-identical logits at corresponding positions for a fixed model and config.
- **Verification:** the scheduler and paged prefix-sharing KV cache are proven in Verus. A Triton analyzer proves kernels give bitwise-equal output across batch size, query length and KV layout.
- **Kernel design:** configs fixed at init and independent of M, chunk or phase; zero-init FP32 accumulator; no runtime autotuning.
- **Results:** passed **5,488/5,488** variation pairs across 7 models (Llama 3.x, Gemma 3/4). Decode throughput was 1.36–3.19× vLLM's batch-invariant modes, but prefill was lower and the system is slower than default and SGLang deterministic modes.
- [arXiv 2609.38981](https://arxiv.org/abs/2609.38981); [HTML](https://arxiv.org/html/2609.38981v1)

**Divergence amplification**
- The divergence index rises from about 65 tokens (BF16) to about 1,826 (FP32), so lower precision diverges earlier and the effect compounds over long chains of thought — [arXiv 2506.09501v2](https://arxiv.org/html/2506.09501v2)
- TML's 235B MoE example diverged at token 103 of 1,000 — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- In LLM-42, the first consistent span is often hundreds of tokens and the second is often near zero. Once a sequence diverges, it stays divergent — [LLM-42 HTML](https://arxiv.org/html/2601.17768v2)
- (Jan 2026; search-snippet only, not verified first-hand) A drift study reports a 120B-parameter model gave identical outputs in only about 12.5% of runs — [alphaXiv 2601.19934](https://www.alphaxiv.org/abs/2601.19934)

### Inferences
- **Two design families exist:**
  - *Invariant kernels everywhere* (TML, vLLM, SGLang, TBIK, Vosti): simplest semantics, fixed overhead on all traffic.
  - *Fast path plus verify/repair* (LLM-42, MarginGate): overhead proportional to deterministic traffic and to low-margin steps. It still needs a canonical reference schedule, so it is "deterministic with respect to a canonical fixed-shape execution".
  - CoRun shows padding to fixed shapes is a cheap third option when kernels are position-invariant.
- For a compiler-like system, a **formal contract plus proof-carrying kernel set (Vosti-style)** is the strongest available assurance. Testing alone missed leaks in vLLM and SGLang modes (sliding window, prefill/decode).
- FP32 compute (LayerCast) reduces but does not eliminate divergence, because reduction order still varies. It should be combined with fixed reduction order, not used instead of it.

### Gaps
- No 2026 replication of LayerCast's reasoning-variance numbers on MoE reasoning models (for example DeepSeek-R1 full or Qwen3-235B-Thinking) was found.
- The LLM-42, MarginGate, CoRun and Vosti evaluations are mostly dense ≤14B models on 1–4 GPUs. MoE plus multi-node performance under these schemes is untested in the papers I found.

---

## Q5. Cross-hardware / cross-vendor bitwise reproducibility and verifiable inference

### Takeaway
**Same-SKU bitwise reproducibility is solved in practice** (EigenAI 100% across hosts; vLLM, SGLang and TBIK). **Cross-architecture within NVIDIA is achievable only by owning the kernels**: fixed reduction order independent of SM count and tile heuristics, IEEE FP32 accumulation in registers. 2609.25624 achieved bitwise-identical *linear layers* across Ampere, Ada and Hopper, but attention, norms and the sampler still leaked. **Cross-vendor (NVIDIA vs AMD vs TPU) bitwise is an open problem** with no published demonstration for LLM inference that I found. Gensyn's RepOps/REE claims cross-processor reproducibility with correctly-rounded math, but publishes no overhead or hardware list. Integer arithmetic (power-of-two INT8 scales) is the most promising route to exact cross-kernel agreement. Verification therefore splits into bit-exact replay (EigenAI, Gensyn) and tolerance-based schemes (TOPLOC), and tolerance-based conformance has measurable blind spots.

### Cited Findings

**Cross-architecture**
- (Sep 22, 2026) Fixed-configuration fused-upcast GEMMs load BF16 weights, upcast to FP32 in registers and accumulate in IEEE-754 FP32 with a reduction order that depends only on problem shape. This gives **bitwise-identical linear-layer outputs across A100 (SM80), L40S (SM89) and H100 (SM90)** — [arXiv 2609.25624](https://arxiv.org/abs/2609.25624)
- Cross-GPU greedy divergence:

  | Setting | Divergence |
  |---|---|
  | Unmitigated BF16 | 30.8–100% |
  | LayerCast-style baseline (Yuan et al.) | 0–0.51% |
  | Proposed method | 0% in 11/12 cells; 0.08% residual traced to unpinned H100 attention |

  — [arXiv 2609.25624 HTML](https://arxiv.org/html/2609.25624v1)
- Speed: 1.17–1.43× faster than the LayerCast-style baseline on A100/H100 and 1.6–3.1× on L40S; prefill runs at about 0.85× cuBLAS. Batch invariance holds only for M ≤ 64 and breaks at M=65. AMD is called "an open problem"; Blackwell and MoE are not covered — [arXiv 2609.25624 HTML](https://arxiv.org/html/2609.25624v1)
- (Jan 30, 2026) EigenAI: A100 vs H100 gave **0% bitwise match**. The authors plan per-architecture verifier pools — [EigenAI HTML](https://arxiv.org/html/2602.00182v1)
- Community data (Oct 2026): the BF16 batch effect differs by 0.1–0.7 points between two Ada cards and is 2.4–10.9 points lower on Volta — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)

**Integer and fixed-point arithmetic**
- (Aug 25, 2026) Rounding every INT8 weight scale to a power of two made CUTLASS and Triton INT8 kernels agree bitwise at every linear layer of Qwen3 (196/196 and 252/252, versus 8/196 and 10/252 with original scales). Generated token sequences became byte-identical at 1.7B, 8B and 14B (8/8 prompts vs 0/8). Perplexity changed by +0.32%, −0.28% and +0.48% — [arXiv 2609.00363](https://arxiv.org/abs/2609.00363)
- The same paper's fault injection (8,232 cells) showed that tolerance-based conformance of one BF16 ULP is blind to a whole class of epilogue faults (scale precision, double rounding, multiplication order, truncation, fused ordering). Four of five went undetected — [arXiv 2609.00363](https://arxiv.org/abs/2609.00363)
- A community proposal suggests exact fixed-point (256-bit) accumulation as a *reference oracle* for small models, not a serving path. The commenter disclosed working at Anomly — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)

**Gensyn RepOps / REE / Verde**
- (Mar 17, 2026) REE (Reproducible Execution Environment) is a containerized inference runtime. Its RepOps layer "fixes reduction order", "uses correctly rounded maths for key functions" and does extra-precision work so outputs stay stable across supported hardware. Every run emits a receipt binding model, prompt, config and output, which a verifier re-checks by re-running. It supports 40+ open models. The page lists no specific hardware and no overhead numbers — [Gensyn REE blog](https://www.gensyn.ai/blog/ree)
- Verde (Feb 2025) uses refereed delegation: it bisects to the first disagreeing operator and recomputes only that operator. This requires bitwise agreement among honest nodes, which RepOps supplies — [Verde, arXiv 2502.19405](https://arxiv.org/pdf/2502.19405)
- OPEN-1B (Sep 2026) reportedly says RepOps gives bitwise-reproducible results across CPUs, NVIDIA GPUs and Apple GPUs at some speed cost (from search snippet, not verified first-hand) — [OPEN-1B, arXiv 2609.17380](https://arxiv.org/pdf/2609.17380)
- A secondary blog reports RepOps matmul overhead under 30% versus cuBLAS. **Unverified; no primary source found.**

**EigenAI (bit-exact plus optimistic verification)**
- (Jan 30, 2026) An untrusted operator runs inference on a fixed GPU architecture, signs and encrypts the request and response, and posts the log to EigenDA. Watchers can challenge via EigenVerify, with re-execution in a TEE under threshold-released keys. "Because inference itself is bit-exact, verification reduces to a byte-equality check"; one honest replica suffices — [arXiv 2602.00182](https://arxiv.org/abs/2602.00182)
- **Implementation:** H100/GH200 only; CUDA 12.4 with R550 driver as an example; containers pinned by digest; ECC and persistence mode required.
  - Deterministic GEMM with fixed block-to-tile mapping
  - Warp-level `__shfl_down_sync` binary-tree reduction
  - No FP atomics
  - Q4/Q5 matmul inherited from llama.cpp
  - Fixed-seed PRNG with canonical iteration order for top-k/nucleus sampling
  — [EigenAI HTML](https://arxiv.org/html/2602.00182v1)
- **Overhead:** deterministic GEMM at 0.97× cuBLAS and end-to-end at 0.98× (+1.8%). The paper is internally inconsistent: it cites 95–98% and 97–99% in different places, and both 10,000 runs and 10 runs per configuration — [EigenAI HTML](https://arxiv.org/html/2602.00182v1)
- **Vendor claims:** 100% reproducibility across 10,000 runs, under 2% overhead, "live on mainnet". No independent verification found — [EigenCloud blog](https://blog.eigencloud.xyz/deterministic-ai-inference-eigenai/)

**TOPLOC (Prime Intellect; tolerance-based)**
- (Jan 2025; ICML 2025) Locality-sensitive hashing of intermediate activations detects unauthorized changes to model, prompt or precision with "no false positives or negatives in our empirical evaluations". It is robust "across diverse hardware configurations, GPU types, and algebraic reorderings". Proofs take 258 bytes per 32 tokens versus 262 KB for raw embeddings (Llama-3.1-8B) — [arXiv 2501.16007](https://arxiv.org/abs/2501.16007); [ICML 2025 PMLR](https://proceedings.mlr.press/v267/ong25a.html); [Prime Intellect blog](https://primeintellect.ai/blog/toploc)

### Inferences
- **Recommended tiering for a model compiler:**
  - **Tier A** — bitwise on a pinned (SKU, driver, container, kernel set) "target triple". Achievable now at about 0–10% (EigenAI, vLLM V4-Flash) to 20–60% (generic BI) overhead.
  - **Tier B** — bitwise across NVIDIA architectures. Requires owning every kernel (GEMM, attention, norm, softmax, sampler) with shape-only reduction order and no tensor-core-path differences. GEMM is demonstrated; attention and norms are not yet.
  - **Tier C** — cross-vendor. Only plausible via integer/fixed-point paths (power-of-two-scaled INT8) or correctly-rounded software reductions (RepOps-like), at unknown cost.
- Verification should be bit-exact where Tier A holds (byte equality, cheap disputes). Tolerance-based hashing (TOPLOC) is for heterogeneous fleets, accepting that sub-ULP faults are invisible.

### Gaps
- No published cross-vendor (NVIDIA vs AMD vs TPU) bitwise LLM inference result found.
- No primary RepOps overhead numbers or hardware list found.
- zkML (for example zkLLM) and opML were not researched because the web-search budget was exhausted. I found no primary sources on their 2026 overheads.
- TOPLOC thresholds and behavior under TP/attention-kernel changes were not extracted (abstract only).

---

## Q6. Determinism in RL: train–inference mismatch, MoE routing, importance-sampling corrections, and what it implies for serving

### Takeaway
RL post-training is the main *commercial* driver of deterministic inference. Rollout engines (vLLM, SGLang) and trainers (FSDP, Megatron, TorchTitan) compute different logprobs for the same weights and tokens, which makes "on-policy" RL silently off-policy and can collapse training, especially for MoE. The fixes form a ladder:
1. Algorithmic correction: TIS/MIS importance sampling.
2. Precision fixes: FP16 instead of BF16; FP32 LM head and router.
3. MoE routing replay (R3).
4. Full bitwise train/inference parity: KL = 0. TML in Sep 2025, vLLM + TorchTitan in Nov 2025 (2.4× slower), TBIK across TP sizes, and vLLM DeepSeek-V4-Flash MoE in Oct 2026 at about 91% throughput.

### Cited Findings
- **Off-policy RL** (Aug 2025; NeurIPS 2025): the vLLM/FSDP implementation gap "implicitly turns the on-policy RL to be off-policy". Truncated importance sampling (TIS) caps the ratio with min(ρ, C), a one-sided cap, and is integrated into VeRL and slime — [Feng Yao et al. blog](https://fengyao.notion.site/off-policy-rl); [NeurIPS 2025 "On the Rollout-Training Mismatch"](https://neurips.cc/virtual/2025/124259)
  - Note: the blog body could not be fetched (Notion rendering). These details come from search-result summaries.
- **TML** (Sep 2025): importance weighting keeps KL around 0.001 with spikes. Bitwise-identical sampler and trainer keeps KL at 0 and avoids collapse — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **FP16** (Oct 2025, Sea AI Lab/NUS): BF16 rounding error is the main cause of mismatch. Switching to FP16 (a few lines of code) stabilizes RL. Plain importance-weighted policy gradient in FP16 beat all BF16 baselines across GRPO, GSPO, TIS, MIS and PG, including 14B dense and MoE models on VeRL and Oat — [arXiv 2510.26788](https://arxiv.org/abs/2510.26788)
- **R3 — Rollout Routing Replay** (Oct 2025; ICML 2026): records routing from the inference engine and replays it in training. It significantly reduces train-inference KL, cuts extreme discrepancies by about an order of magnitude, prevents collapse, and beats GSPO and TIS — [arXiv 2510.11370](https://arxiv.org/abs/2510.11370); [PMLR v306](https://proceedings.mlr.press/v306/ma26aw.html)
- **ByteDance** (via the RFC, secondary): KL of about 5e-4 to 1e-3 on one GPU type and 1e-2 to 1e-1 on another, so mismatch severity is hardware-dependent — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)
- **Router Replay memory:** about 35 GB per actor (about 280 GB across 8 actors) for a 96-layer MoE at long context, which motivates a `uint8` expert-ID format — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)
- **Sampling-support mismatch:** when top-p/top-k truncate the distribution, trainer-side normalization over the full vocabulary biases the ratio, and TIS/MIS cannot fix a mismatched support. Community data showed renormalizing with the sampling mask moved the mean ratio from 0.896 to 1.001 and out-of-band tokens from 51.8% to 4.3% — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)
- **vLLM + TorchTitan** (Nov 10, 2025): Qwen3-1.7B GRPO on GSM8K. Every forward-pass kernel was audited; vLLM BI forward kernels were reused with custom backward passes. KL was always 0, with faster and higher reward, but about **2.4× slower**. Limits: two model code copies, no `torch.compile`, eager vLLM. Code is at `torchtitan/experiments/deterministic_vllm_rl` — [vLLM blog](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference)
- **TBIK** (Nov 2025): vLLM at TP=4 and FSDP at TP=1 bitwise identical. Final GSM8K Pass@1 was 0.73 (TBIK), 0.68 (BIO) and 0.60 (BF16) — [TBIK HTML](https://arxiv.org/html/2511.17826)
- **vLLM DeepSeek-V4-Flash** (Oct 14, 2026):
  - The baseline's train-inference KL rose from about 2e-3 to **29.9** while reward fell back. The consistent run had KL and max diff exactly 0 for 100 steps, at 91.3% throughput.
  - R3 was unnecessary because the router GEMM is deterministic on identical inputs.
  - It requires 1 update per step; with 4 updates per step, about 7% of tokens switched expert combination (policy staleness, not nondeterminism).
  - A separate verl PR preview reported about 10% end-to-end overhead on 32 GB200.
  - DeepSeek-V4.1 also reached 0 diff.
  — [vLLM blog, Oct 2026](https://vllm.ai/blog/2026-10-14-train-inference-consistency)
- **SGLang + slime:** GRPO runs with deterministic rollouts (temp > 0 with seeds) gave identical rollouts and losses. With the training-side flags (FA2, Megatron deterministic mode, `NCCL_ALGO=Ring`, and others), two Qwen3-8B runs had identical curves — [LMSYS blog](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- **Community GRPO data:** six GRPO arms on GSM8K (150 steps, two seeds) ended within 0.013 reward of default (SE about 0.026). At small scale, the mismatch fixes did not change final reward measurably — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)

### Inferences
- **Requirements this places on a deterministic serving stack:**
  - Batch-invariant *and* parallelism-layout-invariant numerics, because rollout and trainer use different TP/EP/CP layouts.
  - An FP32 router and LM head, or at least a single deterministic kernel.
  - Deterministic top-k and expert combine.
  - Seeded, counter-based sampling with the sampling mask exportable.
  - Routing-ID export (for R3 fallback).
  - A weight-version stamp on outputs.
- The Oct 2026 vLLM result is the strongest evidence that **bitwise MoE parity at production scale (32× GB200, EP, FP8, sparse attention) costs under 10%** when kernels are co-designed. That is a much better trade than the 2.4× of the Nov 2025 dense prototype.
- A model compiler that wants "same prompt → same bytes" gets RL-grade train/inference parity almost for free if the same kernel set is used for the training forward pass.

### Gaps
- Exact TIS cap values and mismatch magnitudes from the Feng Yao blog could not be verified first-hand (Notion page did not render).
- R3's exact routing-disagreement fraction and KL reduction numbers were not extracted from the full paper; the ~10% figure is secondary via the TRT-LLM RFC.

---

## Q7. Production systems and startups selling "deterministic LLM inference" and their claims

### Takeaway
As of October 2026, "deterministic inference" is sold mainly in **verifiable / crypto-adjacent compute** (EigenAI / EigenCloud, Gensyn REE, Prime Intellect TOPLOC) and offered as **open-source engine modes** (vLLM, SGLang). Mainstream closed APIs still offer only best-effort seeds. Vendor claims (100% reproducibility, under 2% overhead) are self-reported and in EigenAI's case internally inconsistent. All of them are scoped to a pinned hardware SKU or a "supported hardware" list.

### Cited Findings
- **EigenAI / EigenCloud** (Jan 2026 paper; blog):
  - Claims 100% reproducibility across 10,000 test runs, under 2% overhead, and "live on mainnet".
  - Pitched for prediction-market judges, trading bots and scientific assistants.
  - Verification is a byte-equality check through optimistic re-execution in a TEE.
  - Scope is H100/GH200 only; determinism fails across A100 vs H100.
  — [EigenCloud blog](https://blog.eigencloud.xyz/deterministic-ai-inference-eigenai/); [arXiv 2602.00182](https://arxiv.org/abs/2602.00182); [HTML](https://arxiv.org/html/2602.00182v1)
- **Gensyn REE** (Mar 17, 2026): reproducible inference with signed receipts that are validated and verified by re-run. Supports 40+ open models; RepOps provides the reproducible operators. No performance numbers are published on the announcement page — [Gensyn REE blog](https://www.gensyn.ai/blog/ree)
- **Prime Intellect TOPLOC** (Jan 2025; ICML 2025): tolerance-based proof of inference for decentralized inference, not bitwise determinism. Pip package `toploc` — [Prime Intellect blog](https://primeintellect.ai/blog/toploc); [arXiv 2501.16007](https://arxiv.org/abs/2501.16007)
- **Inferact** (an author affiliation on the Oct 2026 vLLM bitwise-consistent RL post) appears to be a vLLM-associated company working on this. I did not verify its commercial offering — [vLLM blog, Oct 2026](https://vllm.ai/blog/2026-10-14-train-inference-consistency)
- **Microsoft Research** released LLM-42 as code ([github.com/microsoft/llm-42](https://github.com/microsoft/llm-42), per the [arXiv abstract](https://arxiv.org/abs/2601.17768)). This is research, not a product.
- **OpenAI** (archived cookbook, about late 2023): `seed` plus `system_fingerprint` is best-effort; "Determinism is not guaranteed" — [OpenAI cookbook](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter). This is old and may be superseded; I did not check current API docs.
- **Anomly:** a commenter on the TRT-LLM RFC disclosed working there while proposing fixed-point oracles. I could not verify what Anomly sells — [TRT-LLM RFC #19501](https://github.com/NVIDIA/TensorRT-LLM/issues/19501)

### Inferences
- There appears to be no mainstream hosted API (OpenAI, Anthropic, Google, Together, Fireworks) with a *contractual* bitwise-determinism guarantee. I did not verify this for 2026 because the search budget ran out. A model-compiler product with a formal, versioned determinism contract (Vosti-style spec plus pinned target triple plus receipts) would be differentiated.
- Every credible bitwise claim is tied to a pinned hardware SKU and software digest. Cross-hardware claims (Gensyn) lack published evidence or overhead data.

### Gaps
- The web-search budget ran out before I could survey other startups and hosted providers (Together, Fireworks, Baseten, Ritual, Hyperbolic, Nesa, Inference Labs, ORA/opML, zkML vendors) for deterministic-inference claims. This should be followed up.
- No independent audit of EigenAI's or Gensyn's reproducibility or overhead claims was found.

# Groq LPU / TSP compile-time (static) scheduling and determinism, and how the idea transfers to LLM agent systems, inference serving and token budgeting (state as of 21 Oct 2026)

Note on sources. Groq's own copies of the ISCA 2020 and ISCA 2022 papers at groq.com/wp-content/... now return HTTP 404, and the groq.humain.ai mirror returned a proxy 502 during this session, so I could not read either paper in full. Single-chip and multi-chip details below come mainly from the primary Groq Hot Chips 34 deck (Aug 2022, Dennis Abts, Groq chief architect), the ISC 2020 ML-hardware workshop deck (Abts), and the paper abstracts as surfaced by search. All dates are publication dates of the cited source.

## 1. How the Groq TSP/LPU achieves deterministic, statically scheduled execution on a single chip

### Takeaway
The TSP turns a many-core chip "inside out". It removes every reactive or speculative hardware mechanism: caches, arbiters, crossbars, replay, reordering, branch prediction and speculation. It exposes all architecturally visible state, including physical SRAM banks and stream registers, to the compiler. Every functional unit is fixed-latency and runs in lockstep, so the compiler can place every tensor in space and time and compute operand arrival to the cycle ("a single subtract"). The result is cycle-accurate, repeatable batch-1 execution. Correctness and timing are guaranteed by the compiler rather than by hardware arbitration.

### Cited Findings
- **Design-for-determinism rules (Hot Chips 34, Aug 2022):**
  - The memory consistency model "disallow[s] reordering of memory refs".
  - There are "No 'reactive components' like arbiters, crossbars, replay mechanisms, caches".
  - Software must have access to all architecturally visible machine state.
  - The "Compiler 'knows' the exact location of every tensor on-chip" and "is orchestrating the arrival of operands and the instructions which use them".
  - Source: [Groq Hot Chips 34 deck, Abts, 2022](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Two explicitly managed interfaces (HC34, 2022):**
  - A *static–dynamic interface*: what is done at compile time versus execution time. The runtime layer manages it.
  - A *hardware–software interface*: what state is visible to the compiler so it can "reason about correctness and provid[e] predictable performance".
  - Groq states: "Machine learning models are a good fit for this static analysis and deterministic execution."
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **What the TSP drops compared with CPUs (HC34, 2022):** speculative and out-of-order execution ("increases tail latency"), and implicit data flow through DRAM→L3→L2→L1 cache hierarchies ("introduce complexity and non-determinism"). These are replaced by "A large, single-level scratchpad SRAM, fixed, deterministic latency" with tensors explicitly allocated by device, hemisphere, memory slice, bank and address offset. — [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **The paper's own description (ISCA 2020, "Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads"):** a functionally sliced microarchitecture in which memory units are interleaved with vector and matrix functional units. It avoids reactive elements "for example, arbiters and caches" to keep execution deterministic. — [Groq ISCA 2020 page](https://groq.com/groq-isca-paper-2020/); [PDF mirror](https://groq.humain.ai/wp-content/uploads/2024/02/2020-Isca.pdf)
- **Wikipedia's summary of the TSP:** a single-core deterministic design that omits branch predictors, arbiters, reorder buffers and caches, with the compiler controlling all execution. — [Wikipedia: Groq](https://en.wikipedia.org/wiki/Groq). The page carries an "LLM-generated, unverified" maintenance banner, so treat it as secondary.
- **Functional slices (HC34, 2022):**
  - ICU (instruction control units), MEM (SRAM), VXM (vector), MXM (matrix) and SXM (switch: reshapes, permutes, transposes and chip-to-chip I/O).
  - Instructions flow vertically and execute SIMD-style. Tensor operands and results flow horizontally on "streams".
  - The chip is split into East and West hemispheres with the VXM in the middle.
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Instruction issue (HC34, 2022):**
  - 144 independent instruction dispatch paths (ICUs), each issuing one instruction per cycle. Groq describes this as "fully-pipelined 144-wide VLIW instructions".
  - Instruction dispatch logic takes "< 3% area overhead".
  - The compiler performs "Cycle-accurate instruction scheduling", and functional units "execute in lockstep".
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Stream registers as the on-chip network (HC34, 2022):**
  - The interconnect is a "Simple, one-dimensional" east/west array of stream registers, where each "Stream register = one-cycle hop".
  - "No arbiters / queues = software can easily reason about exact data movement without simulation. Travel time calculation as simple as a single subtract."
  - There are 64 logical 320-byte streams (32 east, 32 west) and 45 stream-register-file stations per superlane, all SECDED-protected.
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Synchronisation instructions (HC34, 2022):**
  - SYNC parks an ICU until a NOTIFY. Any of the 144 ICUs can NOTIFY the others "to provide a lock-step execution and allow the compiler to reason about time of execution".
  - NOP N delays by N cycles. Repeat n,d repeats the previous instruction n times, d cycles apart.
  - MEM instructions are Read/Write/Gather/Scatter on streams. C2C instructions are Deskew, Send and Receive (320-byte vector).
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Memory (HC34, 2022):**
  - There is "No dynamic hardware caching". The "Flat memory hierarchy (no L1, L2, L3)" is exposed "as a set of physical banks that are directly addressed".
  - 88 independent MEM slices × 8,192 addresses = 220 MB, at 80 TB/s. One slide also says "220 MB … (55TBps)", which is an internal inconsistency in the deck.
  - Groq states that large on-chip capacity "reduces need to spill to non-deterministic DRAM".
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **SRAM capacity in other sources:** the ISCA 2020 paper gives "220 MiBytes of globally shared SRAM" with 32 bytes per lane of stream bandwidth. — [ISCA 2020 PDF mirror (via search abstract)](https://groq.humain.ai/wp-content/uploads/2024/02/2020-Isca.pdf). Groq marketing says 230 MB. A secondary wiki notes that 230 million bytes ≈ 219.3 MiB, so these are the same capacity in different units. — [aiwiki Groq hardware](https://aiwiki.ai/wiki/groq_hardware)
- **Compute (HC34, 2022):**
  - MXM: 320 lanes × 320, with 409,600 MACCs on-chip. Supports INT8 and FP16 inputs with INT32 or FP32 accumulate.
  - VXM: 5,120 ALUs (16 per lane).
  - "Peak of ~750 Tops (Int8Acc32) (900 MHz)", "> 1 TeraOps/sec per mm²".
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **ISCA 2020 paper figures (as surfaced by search):**
  - 14 nm, 25×29 mm die, 900 MHz nominal clock.
  - 320 lanes in 16-lane "superlanes", where a superlane is "the minimum granularity of computation".
  - A headline "peak performance of 820 TeraOps/sec from 26.8B [transistors]".
  - ResNet-50: "20.4K processed images per second (IPS) with a batch-size of one", which the paper calls "a 4× improvement" over other modern GPUs and accelerators.
  - The 820 TOPS figure differs from the later 750 TOPS at 900 MHz.
  - Source: [ISCA 2020 PDF mirror](https://groq.humain.ai/wp-content/uploads/2024/02/2020-Isca.pdf); [Groq ISCA 2020 page](https://groq.com/groq-isca-paper-2020/)
- **Yield redundancy (HC34, 2022):** the chip has 21 superlanes. One is a spare, and the compiler never sees it. This is "the only place where we 'hide'" anything from the compiler. — [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Exceptions and faults are decided ahead of time (HC34, 2022):**
  - Arithmetic overflow semantics are chosen at compile time (ADD_SAT/ADD_MOD, SUB_*, MUL_*).
  - SECDED ECC protects all SRAM, datapaths, stream registers and instruction buffers.
  - The runtime does a "One-shot replay on an error" to tell transient faults from persistent ones, and replays persistent faults on alternate hardware.
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Determinism simplifies hardware verification (HC34, 2022):** there are "No 'liveness' properties across the chip or system", no cooperating FSMs or coherence, "No arbiters or link-layer replay", and stateless, side-effect-free functional units checked through assume-guarantee reasoning. — [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Groq's later framing (blog, 7 Mar 2025):**
  - Four principles: "Software-first", "Programmable assembly line architecture", "Deterministic compute and networking", "On-chip memory".
  - "Data flow is statically scheduled by the software during compilation." The compiler produces "a program encompassing all data movement information throughout execution".
  - SRAM bandwidth is "upwards of 80 terabytes/second", against about 8 TB/s for GPU HBM.
  - Source: [Groq blog, "What is a Language Processing Unit?"](https://groq.com/blog/the-groq-lpu-explained)
- **Third-party summary (Zellic, 24 Apr 2024):** instruction pointers act as implicit timestamps for sending and receiving data, and "many optimizations improve average-case performance but not worst-case performance". — [Zellic](https://www.zellic.io/blog/groq-tsp-whitepapers)

### Inferences
- The essential contract is: (1) a fully static dataflow graph; (2) fixed-latency, side-effect-free operators; (3) an explicitly addressed, non-caching memory; and (4) a compiler-owned global clock. Remove any one and the compiler can no longer prove timing. For an agent compiler, the analogues are: a static task DAG; operators (models, tools) with bounded and predictable latency and token cost; explicit, compiler-placed context (KV/prefix) storage instead of LRU caches; and a global plan clock.
- Groq keeps a thin dynamic layer even in hardware: one-shot replay on faults, plus a runtime that manages the static–dynamic interface. A "goal-compiled" agent system should copy this split: a static plan plus a narrow, pre-specified runtime fallback (replay or reroute), not open-ended dynamic re-planning.

### Gaps
- I could not read the ISCA 2020 full text, so the exact compiler algorithms and the full results tables beyond the abstract figures were not verified this session.
- No public source found explains how the Groq compiler handles data-dependent control flow on chip. For example, it is not documented whether MoE expert selection is compiled as compute-all-and-mask, as predication, or as host-driven program switching.

## 2. Multi-chip deterministic networking ("software-defined Tensor Streaming Multiprocessor", ISCA 2022 / RealScale) and how LLM inference is mapped

### Takeaway
Groq extends single-chip lockstep to hundreds or thousands of chips. Each chip has free-running hardware-aligned counters, plus software deskew instructions that compensate for clock drift between independently clocked ("plesiochronous") chips. Links are paced in software. The compiler schedules every 320-byte tensor "packet" on specific links and paths, so there are no routers, no hardware flow control, no congestion sensing and no retries. FEC replaces link-level replay. LLMs are pipelined and tensor-parallelised across hundreds of chips because each chip holds only about 230 MB.

### Cited Findings
- **ISCA 2022 paper, "A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning" (pp. 567–580, DOI 10.1145/3470496.3527405):** a software-controlled network on a software-defined Dragonfly. It "avoids the latency variation introduced by dynamic contention for network links". It scales to "up to 10,440 TSPs and more than 2 TeraBytes of global memory accessible in less than 3 microseconds of end-to-end system latency". — [KAIST record](https://koasas.kaist.ac.kr/handle/10203/301182?mode=full); [Groq ISCA 2022 page](https://groq.com/?p=3478)
- **Software-scheduled versus conventional scale-out network (HC34, 2022):**
  - "No hardware-arbitration for dynamic contention", "No deep input queues required", "No hardware routing".
  - Software "scheduling" provides load balancing, with "No congestion sensing: tensors are statically scheduled".
  - "Route Tensors, Not Packets": the packet is one 320-byte vector with "Only 2.5% encoding overhead" and "No hardware flow control, no virtual channel information".
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Routing (HC34, 2022):** load balancing replaces global adaptive routing with "deterministic, load-balanced path scheduling" that uses non-minimal routes for path diversity. Groq cites adaptive-routing pathologies ("phantom congestion") as the reason. — [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Topology and packaging (HC34, 2022):**
  - A GroqNode has 8 TSPs. A GroqRack has 9 nodes, and "Any node can be the 'spare' node".
  - A 32-port "group" acts as a virtual high-radix router. The system scales "up to 33 groups in a single global hop".
  - Each TSP has 16 directly connected C2C links.
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Global synchronisation (HC34, 2022):**
  - The problems are link-latency variation, the lack of a global clock, and clock drift.
  - The solution has hardware and ISA parts. Each TSP has a free-running Hardware Aligned Counter (HAC) and a Software Aligned Counter (SAC). The DESKEW instruction pauses until the HAC overflows. TRANSMIT sends a sync notification across C2C. "Runtime Deskew" delays a TSP by "t ± dt deterministically".
  - Links are "paced using software to avoid overflow / underflow". "FEC … corrects most transmission errors."
  - Source: [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **RealScale communication model (HC34, 2022):** an RDMA read needs a request/reply round trip through NICs. RealScale is a one-way scheduled Send(X)/Recv(X), giving "logically shared access to global SRAM which is physically distributed among the TSPs". — [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **AllReduce versus an 8×A100 NCCL system (HC34, 2022):** the A100 system has about 3× higher network channel bandwidth. Normalised for that, Groq "matches the bandwidth at large tensor size while significantly improving bandwidth at intermediate tensor size", because Read(vector)→Send(vector) takes "Only a handful of cycles". — [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Earlier scale figures (ISC 2020 ML-hardware workshop):** a "Worst-case system latency of 2.5 µsec", and example systems scaling from 1 to 320 TSPs across 40 nodes. — [Abts ISC 2020 deck](https://mlhardware.github.io/2020/groq.pdf)
- **Zellic summary (Apr 2024):** directly connected TSPs calibrate one-byte counters into a shared clock. Compile-time scheduling lets more links be used at once with "no runtime request/response messages". — [Zellic](https://www.zellic.io/blog/groq-tsp-whitepapers)
- **Workload mapping in HC34 (2022):** BERT layers are spread across the TSPs of a node (model parallelism), with mini-batches across nodes. The stated goal is to "Minimize latency variance to provide predictable throughput". — [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Llama 2 70B demo (The Next Platform, 27 Nov 2023):**
  - 576 GroqChips (8 racks × 9 nodes × 8 cards, all nine nodes used for compute), INT8, 512 input and 1,024 output tokens, more than 300 tokens/s.
  - Groq claimed 10× the speed at one-tenth the cost of 576 H100s, and about 1–3 J/token against 10–30 J/token. The author doubted the system-cost claim.
  - Pods scaled to 264 chips at the time, with 4,128 chips planned on one fabric.
  - Source: [The Next Platform](https://www.nextplatform.com/ai/2023/11/27/groq-says-it-can-deploy-1-million-ai-inference-chips-in-two-years/1661216)
- **SemiAnalysis (Feb 2024):** "each chip has only 230MB of SRAM", so Mixtral 8x7B was served on 576 chips (8 racks × 9 servers × 8 chips). The article is mostly paywalled. — [SemiAnalysis, "Groq Inference Tokenomics: Speed, But At What Cost?"](https://newsletter.semianalysis.com/p/groq-inference-tokenomics-speed-but)
- **Groq's own description of LLM mapping:** its "Inside the LPU" post covers "SRAM-centric design", "statically scheduled networks for predictable performance", "tensor parallelism for faster single-user latency", TruePoint numerics, speculative decoding and pipeline parallelism. Only the meta description was retrievable. — [Groq blog](https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed)
- **MoE serving (Groq blog, 27 May 2025):** chips interconnect into "one shared resource fabric" via the Groq Compiler and "RealScale™ chip-to-chip interconnect". Llama 4 Maverick (a 400B MoE) ran on day one. No chip counts or tokens/s were given. — [Groq blog](https://groq.com/from-speed-to-scale-how-groq-is-optimized-for-moe-other-large-models/)

### Inferences
- Static networking works because the communication pattern of a fixed model graph is known exactly, so link contention can be solved offline like register allocation. The agent-system analogue is to pre-assign each step's calls to specific model replicas or GPU pools and schedule them in time, instead of letting a load balancer react. That only works if call durations are predictable, which is why token budgets (Section 7) are the "fixed latency" equivalent.
- Groq's "plesiochronous plus periodic software deskew" pattern maps to plan checkpoints. Execute statically between sync points, and resynchronise (re-estimate remaining budget and time) at a small number of compiler-inserted barriers.

### Gaps
- I could not verify the exact per-model chip counts and parallelism split (tensor versus pipeline) for current GroqCloud models (Llama 3.x/4, Kimi K2, gpt-oss and others). Groq has not published them.
- The 576-chip Llama 2 70B figure is from Nov 2023 press. It probably shrank on later software, but I found no primary update.

## 3. Performance consequences, cost and limits of the static/SRAM-only approach

### Takeaway
Determinism, SRAM bandwidth and the absence of contention buy very high single-stream decode speed: about 300 tokens/s on Llama 2 70B in 2023, and about 1,665 tokens/s on Llama 3.3 70B with speculative decoding in late 2024, both on the 14 nm chip. The price is very low memory capacity per chip (about 230 MB), so a 70B model needs hundreds of chips. That raises capital cost, rack count and power per deployment, and it favours small-batch, latency-first operation over throughput per dollar. Independent analysts questioned the economics.

### Cited Findings
- **Llama 2 70B speed (late 2023 / early 2024):** Groq's Anyscale LLMPerf write-up reports an average of 185 output tokens/s, "3–18× faster" than other providers on that leaderboard. — [Groq blog](https://groq.com/groq-lpu-inference-engine-crushes-first-public-llm-benchmark/). Artificial Analysis measured a median of 241 tokens/s for Groq against about 18 tokens/s for Azure. — [DeepLearning.AI The Batch](https://www.deeplearning.ai/the-batch/groq-elevates-ai-processing-speeds-with-advanced-chips)
- **Speculative decoding on the 14 nm LPU (late 2024):** Llama 3.3 70B rose from about 250 to 1,660 tokens/s "with software updates alone". Artificial Analysis measured 1,665 output tokens/s, which Groq says is ">20X faster than the median of other providers". — [Groq blog](https://groq.com/groq-first-generation-14nm-chip-just-got-a-6x-speed-boost-introducing-llama-3-1-70b-speculative-decoding-on-groqcloud). Groq's console lists about 1,600 tokens/s, an 8K context, and $0.59/$0.99 per million input/output tokens. — [GroqCloud model docs](https://console.groq.com/docs/model/llama-3.3-70b-specdec)
- **Memory-capacity limit:** a 230 MB SRAM-only chip forces large models across many chips (576 for Mixtral). — [SemiAnalysis, Feb 2024](https://newsletter.semianalysis.com/p/groq-inference-tokenomics-speed-but). The Next Platform's author doubted Groq's "1/10th the cost" claim, given system-level costs. — [The Next Platform, Nov 2023](https://www.nextplatform.com/ai/2023/11/27/groq-says-it-can-deploy-1-million-ai-inference-chips-in-two-years/1661216)
- **Groq's stated rationale for batch size (ISC 2020 deck):** Groq optimises for batch-size 1 and "maximize[s] the number of INT … per floor tile". — [Abts ISC 2020 deck](https://mlhardware.github.io/2020/groq.pdf). HC34 says "Batch-1 inference is important for responsiveness and delivering quality-of-service (QoS) that is impossible to do with more traditional microarchitectures using crossbars, cache hierarchies". — [HC34 deck](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **Supply-chain angle (Nov 2023):** Ross said Groq's supply chain avoids HBM and CoWoS packaging. — [The Next Platform](https://www.nextplatform.com/ai/2023/11/27/groq-says-it-can-deploy-1-million-ai-inference-chips-in-two-years/1661216)
- **Process roadmap:** the first-generation chip is GlobalFoundries 14 nm. The next generation was announced for Samsung 4 nm, with a projected 15–20× power-efficiency gain. — [The Next Platform](https://www.nextplatform.com/ai/2023/11/27/groq-says-it-can-deploy-1-million-ai-inference-chips-in-two-years/1661216); [Groq blog, Mar 2025](https://groq.com/blog/the-groq-lpu-explained)
- **The newer NVIDIA-built Groq 3 LPU (Mar 2026):** 500 MB SRAM per chip at 150 TB/s, against about 230 MB at 80 TB/s on v1. It adds external DRAM via fabric expansion: up to 512 GB DRAM per tray via fabric logic plus 256 GB via the host, and 12 TB DDR5 per rack on the product page. This is an implicit admission that SRAM-only capacity was a limit. — [NVIDIA Technical Blog, 16 Mar 2026](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform); [NVIDIA LPX page](https://nvidia.com/en-us/data-center/lpx)
- **Independent critique of static abstractions (STeP, Stanford, Nov 2025 / Jan 2026):** prior spatial-dataflow abstractions force dynamic behaviour to be "implemented statically". Adding dynamic tiling, parallelisation and time-multiplexing gave 2.18× lower on-chip memory, 1.5× lower latency and 2.57× higher compute utilisation on LLM layers in simulation. — [arXiv 2511.07776](https://arxiv.org/abs/2511.07776)

### Inferences
- The static approach wins hardest when (a) the graph is fixed, (b) the workload is latency-bound (decode at small batch), and (c) the work is memory-bandwidth-bound. It loses when capacity or dynamic shapes dominate: MoE routing, variable sequence lengths, long contexts. The 2026 NVIDIA design puts attention and KV on GPUs and FFN/MoE experts on LPUs (Section 4). This suggests the industry answer is a hybrid: static where the shape is static, dynamic where it is not.
- For an agent compiler, the analogue to "SRAM capacity" is the context and KV budget each specialist can hold. Static plans are cheapest when each step's context fits a pre-allocated slot.

### Gaps
- I found no independent, current (2026) cost-per-token comparison of GroqCloud versus GPU clouds. The SemiAnalysis 2024 analysis is paywalled.
- There is no public tail-latency distribution (p99/p50) for GroqCloud that would quantify "determinism" at the service level, which includes queuing.

## 4. Groq corporate status as of October 2026 (verified with primary sources where possible)

### Takeaway
On 24 Dec 2025 Groq announced a non-exclusive licence of its inference technology to NVIDIA. Founder and CEO Jonathan Ross, President Sunny Madra and other staff moved to NVIDIA, and the price was reported (not disclosed) at about $20B. NVIDIA launched the "Groq 3 LPU" / "Groq 3 LPX" rack at GTC (16–17 Mar 2026) as a decode/FFN co-processor for Vera Rubin. The remaining independent Groq is now an inference cloud (GroqCloud). It raised $650M on 22 Jun 2026 under CEO Adam Winter, plans to deploy NVIDIA LPX, and was still adding models to GroqCloud in October 2026.

### Cited Findings
- **The licence (Groq newsroom, 24 Dec 2025):**
  - Groq "entered into a non-exclusive licensing agreement with Nvidia for Groq's inference technology".
  - "Jonathan Ross, Groq's Founder, Sunny Madra, Groq's President, and other members of the Groq team will join Nvidia."
  - "Simon Edwards stepping into the role of Chief Executive Officer". Groq "will continue to operate as an independent company", and "GroqCloud will continue to operate without interruption".
  - No price was disclosed.
  - Source: [Groq newsroom](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale)
- **Reported price:** about $20B (CNBC estimate, reported Dec 2025). — [Yahoo Finance](https://finance.yahoo.com/news/nvidia-license-groq-technology-20-123939149.html); [DCD](https://www.datacenterdynamics.com/en/news/nvidia-to-license-tech-from-ai-inference-chip-company-groq-hire-its-leadership/). Analysts debated whether it is an "acquisition, acquihire or creative licensing deal". — [Constellation Research](https://www.constellationr.com/blog-news/insights/nvidias-groq-deal-acquisition-acquihire-or-creative-licensing-deal)
- **Groq 3 LPX (NVIDIA Technical Blog, 16 Mar 2026):**
  - The chip is the "Groq 3 LPU", designated LP30. It has "500 MB of compiler-managed on-chip SRAM", 150 TB/s SRAM bandwidth, and 96 C2C links at 112 Gbps (about 2.5 TB/s).
  - The rack holds 256 LPUs with 128 GB SRAM, 40 PB/s SRAM bandwidth, 640 TB/s scale-up and 315 PFLOPS FP8.
  - Rubin GPUs do prefill and decode attention over the KV cache. LPUs do "FFN and sparse MoE expert execution": attention–FFN disaggregation (AFD), exchanging activations every token. LPX can also act as a speculative-decoding draft generator.
  - "The compiler explicitly schedules computation, data movement, and synchronization", with no dynamic hardware schedulers.
  - A "plesiosynchronous" C2C protocol cancels clock drift across hundreds of LPUs.
  - Claims: up to 35× higher throughput per MW at 400 tokens/s per user (a 2T-parameter MoE with 400K context, against GB200 NVL72). Orchestration is through NVIDIA Dynamo.
  - Source: [NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform)
- **Rack configuration conflict:** the NVIDIA blog fetch described 16 × 2U trays with 16 LPUs each. Another summary of the same blog and The Decoder (17 Mar 2026) describe 32 trays × 8 LPUs. All agree on 256 LPUs per rack and H2 2026 availability. — [The Decoder](https://the-decoder.com/gtc-2026-with-groq-3-lpx-nvidia-adds-dedicated-inference-hardware-to-its-platform-for-the-first-time/); [NVIDIA LPX page](https://nvidia.com/en-us/data-center/lpx); [DCD](https://www.datacenterdynamics.com/en/news/nvidia-announces-groq-3-lpu-ai-inference-chip-plans-256-lpu-rack/)
- **The $650M raise (Groq press release, 22 Jun 2026):**
  - Led by Disruptive and Infinitum.
  - CEO Adam Winter, Chairman Alex Davis, CFO Matt Eng, COO Alan Rice.
  - 13 data centres across North America, Europe, the Middle East and APAC, targeting about 200 MW by end-2027.
  - Funds will fit out capacity with "Groq's latest inference technology, including NVIDIA's new LPX system".
  - More than 5M developers.
  - Source: [Groq newsroom](https://groq.com/newsroom/groq-raises-usd650m-to-scale-its-ai-inference-cloud-business)
- **Secondary framing of the raise:** Groq is pivoting "From Chip Maker to AI Inference Neocloud". — [AI Weekly](https://aiweekly.co/alerts/groq-raises-650m-to-pivot-from-chip-maker-to-ai-inference-neocloud-targeting). Wikipedia dates the raise to May 2026, which conflicts with Groq's 22 Jun 2026 release. — [Wikipedia](https://en.wikipedia.org/wiki/Groq)
- **GroqCloud still active (blog index, 15 Oct 2026):** the latest post is "Meta's Muse Spark open-weight release Is Now in Limited Preview on GroqCloud". — [Groq blog index](https://groq.com/blog)
- **Earlier milestones:**
  - Series D of $640M at a $2.8B valuation (Aug 2024).
  - $1.5B Saudi commitment for a Dammam GroqCloud data centre (10 Feb 2025).
  - Samsung 4 nm foundry selection (Aug 2023).
  - Source: [Wikipedia](https://en.wikipedia.org/wiki/Groq) (secondary)

### Inferences
- NVIDIA licensing Groq's statically scheduled design and deploying it as a decode/FFN engine next to dynamically scheduled GPUs is the strongest market validation to date (2026) of "static where possible". It also shows that the winning system is heterogeneous and orchestrated (Dynamo), not purely static. This is directly analogous to a goal compiler driving a heterogeneous pool of specialist models.

### Gaps
- No source found gives the date or reason Simon Edwards (CEO from Dec 2025) was succeeded by Adam Winter (CEO by Jun 2026).
- It is unconfirmed whether independent Groq still designs its own silicon. The June 2026 release neither confirms nor denies this. The "abandoning chip manufacturing" framing is secondary.
- The final deal price and structure were not disclosed by either party.

## 5. Which properties of static scheduling generalise, and the contrast with dynamic GPU scheduling (including determinism of results)

### Takeaway
Four properties transfer to LLM and agent systems:
- (a) Latency predictability from eliminating runtime contention.
- (b) A single authoritative plan that owns placement, timing and resources ("the compiler as source of truth").
- (c) Ahead-of-time resource allocation: memory, links, model replicas, token budgets.
- (d) Bit-level result determinism.

All four require the program's control flow and operator costs to be known before execution. LLM agents violate this in two places: output lengths and tool results are data-dependent. The 2024–2026 literature therefore converges on "compile a plan, plus a narrow adaptive runtime". Result determinism on GPUs is achievable but costs throughput.

### Cited Findings
- **Groq's own scoping:** "Machine learning models are a good fit for this static analysis and deterministic execution". Their nodes are operators that "fire only when all their input operands are available". — [HC34 deck, 2022](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- **SGLang's compile mode has the same limit (paper v2, Jun 2024, Appendix D):** SGLang programs can be traced "with abstract arguments" into an IR graph and run by a graph executor. Tracing "is limited to programs without data-dependent control flow". A compile-time "prefix-aware code movement" pass, which uses GPT-4 to reorder nodes and lengthen the constant prefix, succeeded on 12 of 15 templates and added about 60 shareable tokens. The paper calls it "aggressive" because it can change semantics. — [arXiv 2312.07104 (HTML)](https://arxiv.org/html/2312.07104v2)
- **Determinism of results on GPUs (Thinking Machines Lab, Horace He, 10 Sep 2025):**
  - The main cause of temperature-0 nondeterminism is a lack of *batch invariance*: kernel results depend on batch size, which varies with server load. Atomic-add concurrency is not the main driver.
  - Qwen3-235B-A22B produced 80 unique completions out of 1,000 at temperature 0, diverging at token 103. With batch-invariant kernels, all 1,000 were identical.
  - Cost on Qwen3-8B: 26 s (vLLM default), 55 s (deterministic, unoptimised), 42 s (with an improved attention kernel).
  - Source: [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **The LPU's claimed benefits (NVIDIA, Mar 2026):** "reduced execution jitter" and stable TTFT and per-token latency "even at small batch sizes". No measured distributions were given. — [NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform)
- **GPU-side dynamic scheduling wins on throughput (Anyscale, 22 Jun 2023):** continuous (iteration-level) batching beats static batching by up to 23× (vLLM), about 8× (TGI, Ray Serve) and about 4× over FasterTransformer's optimised static batching. Setup: OPT-13B on one A100-40GB. Static batching idles the GPU until the longest sequence finishes. — [Anyscale blog](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- **Dynamism helps on dataflow hardware too:** see the STeP result in Section 3 (2.18× memory, 1.5× latency, 2.57× utilisation from dynamic features). — [arXiv 2511.07776](https://arxiv.org/abs/2511.07776)
- **Static control flow also buys security (CaMeL, Google, Mar 2025):** CaMeL extracts control and data flow from the trusted query so that "untrusted data … can never impact the program flow". It solved 77% of AgentDojo tasks with provable security, against 84% undefended. — [arXiv 2503.18813](https://arxiv.org/abs/2503.18813)
- **Pre-planned branches open a new attack surface (2 Oct 2026):** "branch steering attacks" reached 94.4% and 89.5% success on standard and Dual-LLM computer-use agents. COBRA adds ahead-of-time capability constraints per branch, bringing attack success to 0% with 97% benign utility. — [arXiv 2610.03089](https://arxiv.org/abs/2610.03089)

### Inferences
- **What generalises cleanly:**
  - (1) Plan-level dataflow analysis: parallelism, pipelining and prefix sharing are visible only if the whole DAG is known.
  - (2) Ahead-of-time placement and reservation: model replica, KV/prefix slot, token budget.
  - (3) Deterministic decoding (batch-invariant kernels, fixed seeds) where reproducibility matters, at an observed throughput cost of about 1.6–2×.
  - (4) Security and auditability: the plan is an inspectable artefact.
- **What does not generalise:**
  - Cycle-exact timing, because LLM output length is stochastic.
  - "No arbitration at all", because multi-tenant serving needs admission control.
  - The design implication is to compile *bounds* (token budgets, timeouts, max fan-out) rather than exact times, and to treat budget overruns like Groq's "one-shot replay": a pre-specified recovery path.

### Gaps
- I found no study that directly measures end-to-end agent-task latency variance (p99/p50) under a fully static plan versus a dynamic ReAct loop on the same workload.

## 6. LLM-system work that transfers "compile-time planning" to agents and LLM programs (DAG compilers, app-level schedulers, prefix/KV planning, plan caching, workflow compilers), with measured gains

### Takeaway
Since late 2023 a clear lineage has grown along the same path. Planners compile tool and LLM calls into DAGs (ReWOO, LLMCompiler, LLM-Tool Compiler). Serving systems ingest that application structure: Parrot, SGLang, Teola/Ayo, Alto, Autellix, KVFlow, Halo. By 2025–2026, "physical-plan compilers" choose model, backend and verifier per node under SLOs (Murakkab, Dyserve, HexAGenT). Gains are typically 1.5–4× latency, 2–15× throughput, and 2–7× (up to about 50×) cost. The newest systems pair an offline or per-request compiler with an adaptive runtime rather than relying on purely static schedules.

### Cited Findings
**Planner / DAG compilers (agent level)**
- **ReWOO (23 May 2023):** plans all tool calls up front ("Reasoning WithOut Observation") with placeholders. Results: 5× token efficiency and +4% accuracy on HotpotQA, plus reasoning offloaded from 175B GPT-3.5 to a 7B LLaMA. — [arXiv 2305.18323](https://arxiv.org/abs/2305.18323)
- **LLMCompiler (SqueezeAILab, ICML 2024, arXiv 7 Dec 2023):** three parts "drawing inspiration from the principles of classical compilers": a Function Calling Planner that emits a DAG, a Task Fetching Unit and a parallel Executor. Results: up to 3.7× latency speedup, 6.7× cost savings and about 9% accuracy gain over ReAct. — [arXiv 2312.04511](https://arxiv.org/abs/2312.04511); [GitHub](https://github.com/SqueezeAILab/LLMCompiler)
- **LLM-Tool Compiler (7 May 2024):** fuses similar tool operations into one call, inspired by multiply-add fusion. Results: up to 4× more parallel calls, with up to 40% lower token cost and 12% lower latency on a large Copilot platform. — [arXiv 2405.17438](https://arxiv.org/abs/2405.17438)
- **AsyncLM (9 Dec 2024):** asynchronous function calling with interrupts cuts end-to-end task latency by 1.6×–5.4× on BFCL. This is the *dynamic* alternative to static DAG planning. — [arXiv 2412.07017](https://arxiv.org/abs/2412.07017)

**Application-aware serving (exposing the program to the scheduler)**
- **Parrot (OSDI 2024, arXiv 30 May 2024):** "Semantic Variables" annotate prompt inputs and outputs so the service can do "conventional data flow analysis" across requests, giving "up to an order-of-magnitude improvement". — [arXiv 2405.19888](https://arxiv.org/abs/2405.19888)
- **SGLang (arXiv 12 Dec 2023, v2 Jun 2024):**
  - Frontend primitives plus a runtime with RadixAttention (radix-tree KV reuse) and compressed-FSM constrained decoding: up to 6.4× throughput and up to 3.7× lower latency.
  - Cache-aware "longest-shared-prefix-first" scheduling is provably optimal offline (equivalent to DFS of the radix tree) and reaches about 96% of the optimal hit rate. Hit rates were 50–99% on benchmarks, and 52.4%/74.1% in Chatbot Arena deployments.
  - Source: [arXiv 2312.07104](https://arxiv.org/abs/2312.07104); [HTML](https://arxiv.org/html/2312.07104v2); [GitHub](https://github.com/sgl-project/sglang)
- **Teola (arXiv 29 Jun 2024, v3 Mar 2025), published as Ayo at ASPLOS '25:** represents each query as a "primitive-level dataflow graph" to parallelise and pipeline across modules, with up to 2.09× speedup. Repo examples on 4×RTX 3090: naive RAG 3.72 s→1.97 s, advanced RAG 7.40 s→3.17 s. — [arXiv 2407.00326](https://arxiv.org/abs/2407.00326); [GitHub NetX-lab/Ayo](https://github.com/NetX-lab/Ayo)
- **Alto (arXiv 7 Mar 2024, v3 22 Jul 2025):** streaming and parallelism for compound AI via "nested ancestry" metadata inferred from the programming model. It matches or improves latency by 10–30% against LangGraph implementations. — [arXiv 2403.04311](https://arxiv.org/abs/2403.04311)
- **Autellix (19 Feb 2025):** treats agent programs as "first-class citizens". It preempts and prioritises LLM calls using program-level history to cut head-of-line blocking, giving 4–15× throughput at the same latency compared with vLLM. — [arXiv 2502.13965](https://arxiv.org/abs/2502.13965)
- **AIOS (COLM 2025, arXiv Mar 2024):** an agent OS kernel (scheduling, context, memory, access control) that runs agents up to 2.1× faster. — [arXiv 2403.16971](https://arxiv.org/abs/2403.16971)
- **Conveyor (29 May 2024):** partial tool execution overlapped with decoding cuts request completion latency by up to 38.8%. — [arXiv 2406.00059](https://arxiv.org/abs/2406.00059)

**Prefix/KV reuse planned from the workflow graph**
- **KVFlow (10 Jul 2025):**
  - Abstracts the agent schedule as an "Agent Step Graph" and assigns each agent a "steps-to-execution" value, used for eviction instead of LRU.
  - Prefetches the next step's KV from CPU in the background.
  - Up to 1.83× speedup for single workflows and 2.19× with many concurrent workflows, compared with SGLang's hierarchical radix cache.
  - Source: [arXiv 2507.07400](https://arxiv.org/abs/2507.07400)
- **Halo (2 Sep 2025, v2 Jan 2026):** represents each agentic workflow as a "structured query plan DAG", consolidates batched queries to expose shared computation, and optimises with a cost model over prefill/decode cost, cache reuse and GPU placement. Up to 3.6× speedup for batch inference and 2.6× throughput online. — [arXiv 2509.02121](https://arxiv.org/abs/2509.02121)

**Physical-plan compilers and SLO-aware workflow orchestration (2025–2026)**
- **Murakkab (22 Aug 2025):** a declarative workflow spec decoupled from execution configuration, with a "profile-guided optimizer and adaptive runtime" that maps components to models and hardware under SLOs. Up to 2.8× less GPU use, 3.7× less energy and 4.3× lower cost while maintaining SLOs. — [arXiv 2508.18298](https://arxiv.org/abs/2508.18298)
- **Dyserve, "Serving Agentic Workflows with a Physical-Plan Compiler and Adaptive Runtime" (3 Jul 2026, v2 10 Sep 2026):**
  - Per request, a profile-guided compiler jointly picks each node's model, verification policy and backend, and pre-builds "pressure-specialized variants".
  - The runtime switches variants on live backend pressure "without invoking the optimizer on the load-change path".
  - Results: +3–9 accuracy points with 1.1–6.8× mean-latency speedups over the most accurate baseline. On a burst trace, correct on-time completions rose from 18.1% to 67.2%.
  - This is the closest published analogue to a "goal-compiled agentic MoE".
  - Source: [arXiv 2607.02942](https://arxiv.org/abs/2607.02942)
- **HexAGenT (15 May 2026):** models each request as an "online-revealed DAG". It prioritises ready calls by risk of missing the workflow's completion horizon and jointly places prefill and decode on heterogeneous A100/H100/H200 prefill-decode clusters. The SLO scale needed is 20.1% lower on average at 95% attainment and 33.0% lower at 99% (maximum 45.0% and 80.5%). — [arXiv 2605.16637](https://arxiv.org/abs/2605.16637)
- **PipeSwift (15 Sep 2026, v4 12 Oct 2026):**
  - For agentic serving, job completion time (JCT), not TTFT/TPOT, is the objective, and it is governed by the balance between prefill and decode efficiency.
  - Pipeline parallelism with a co-designed micro-batch partitioning cuts JCT by up to 1.45× against SGLang wide-EP, 2.33× against vLLM PP2 and 1.54× against open-source PD disaggregation.
  - Setup: 360B+ MoE models on 64 H800s with real coding and web-search agent traces.
  - Source: [arXiv 2609.16491](https://arxiv.org/abs/2609.16491)

**Program / workflow optimisers ("compilers" over prompts, structure and model choice)**
- **DSPy (5 Oct 2023):** "a compiler that will optimize any DSPy pipeline to maximize a given metric". It beat standard few-shot prompting by more than 25% (GPT-3.5) and 65% (llama2-13b-chat), and expert demonstrations by 5–46% and 16–40%. — [arXiv 2310.03714](https://arxiv.org/abs/2310.03714)
- **AFlow (ICLR 2025, arXiv Oct 2024):** MCTS search over code-represented workflows. +5.7% over state-of-the-art baselines, and smaller models beat GPT-4o "at 4.55% of its inference cost". — [arXiv 2410.10762](https://arxiv.org/abs/2410.10762)
- **Cognify (12 Feb 2025):** hierarchical autotuning of structure, operators and prompts. Up to 2.8× quality, up to 10× lower cost and 2.7× lower latency. — [arXiv 2502.08056](https://arxiv.org/abs/2502.08056)
- **Archon (ICML 2025):** searches combinations of inference-time techniques and LLMs "Given a compute budget", outperforming o1, GPT-4o and Claude 3.5 Sonnet by 15.1% on average. — [arXiv 2409.15254](https://arxiv.org/abs/2409.15254)
- **Routing and cascades (per-step model selection):** FrugalGPT (May 2023) matches GPT-4 with up to 98% cost reduction. — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176). RouteLLM (Jun 2024) cuts cost by more than 2× without quality loss. — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665)

**Plan caching (amortising the compile step)**
- **Agentic Plan Caching (NeurIPS 2025, arXiv 17 Jun 2025, v2 26 Jan 2026):** extracts "structured plan templates" from completed runs, matches new requests by keyword, and adapts templates with a lightweight model. v2 reports 50.31% lower cost and 27.28% lower latency on average with performance maintained. v1 reported 46.62% cost reduction, so the figures changed between versions. — [arXiv 2506.14852](https://arxiv.org/abs/2506.14852)

### Inferences
- The measured gains decompose into four sources:
  - (i) Parallelism exposed by the DAG: LLMCompiler 3.7×, Teola 2.09×.
  - (ii) Prefix/KV reuse planned from structure: SGLang up to 6.4×, KVFlow 1.8–2.2×, Halo 2.6–3.6×.
  - (iii) Program-aware scheduling across tenants: Autellix 4–15×, HexAGenT 20–33% tighter SLO.
  - (iv) Per-node model, backend and budget selection: Murakkab 4.3× cost, Dyserve 1.1–6.8× latency with higher accuracy, FrugalGPT/AFlow 20–50× cost.
  - A goal-compiled agentic MoE can stack (i)+(ii)+(iv) at plan time and leave (iii) to the serving layer.
- Dyserve's "precompile variants, select at runtime without re-optimising" is the agent-world equivalent of Groq's static schedule with pre-specified replay. It keeps the hot path free of planner calls.
- Plan caching suggests the compile step itself can be amortised: compile once per task *class*, not per task.

### Gaps
- No direct head-to-head benchmark exists across these systems; each uses different baselines (ReAct, vLLM, SGLang, LangGraph).
- I did not verify LangGraph's static-graph (`compile()`) versus dynamic (Send/Command) semantics from primary docs this session; the docs URL redirected.
- I did not find a 2025–2026 paper explicitly titled as an "AI agent compiler" that targets heterogeneous expert-model pools with token budgets, beyond Dyserve and Murakkab.

## 7. Token-budget research: planning compute per step

### Takeaway
Token length is the agent-world equivalent of instruction latency. Since late 2024 it has become controllable: by prompting (TALE), by decode-time intervention (s1 budget forcing), and by training (L1/LCPO). That makes per-step budgets plannable. Accuracy degrades gracefully at about 60–70% token savings on math. However, 2026 evaluations (TRIAGE) show LLMs are still poor at *prospective* allocation across a queue of tasks, which argues for a separate cost-model-driven planner rather than self-budgeting.

### Cited Findings
- **TALE, "Token-Budget-Aware LLM Reasoning" (arXiv 24 Dec 2024, v6 16 Oct 2026):**
  - A zero-shot estimator asks the LLM to predict a token budget, which is then inserted into the CoT prompt.
  - "On average, TALE achieves a 68.64% reduction in token usage while maintaining accuracy with less than 5% decrease". Accuracy went from 83.75% to 81.03%, and cost from 289.78 to 118.46 ×10⁻⁵ $/sample.
  - Budget-estimate in-range accuracy was only 60.61%.
  - The conclusion and table captions also cite 68.9% and 67%.
  - Source: [arXiv 2412.18547 (HTML v1)](https://arxiv.org/html/2412.18547v1); [abstract](https://arxiv.org/abs/2412.18547)
- **s1 / budget forcing (31 Jan 2025):** "forcefully terminating the model's thinking process or lengthening it by appending 'Wait'". s1-32B beats o1-preview on competition math by up to 27%, and budget forcing extrapolates AIME24 from 50% to 57%. — [arXiv 2501.19393](https://arxiv.org/abs/2501.19393)
- **L1 / LCPO (COLM 2025, arXiv 6 Mar 2025):** RL training to satisfy a length constraint given in the prompt. It outperforms s1 at length control, and the 1.5B L1 "surpasses GPT-4o at equal reasoning lengths". — [arXiv 2503.04697](https://arxiv.org/abs/2503.04697)
- **TRIAGE (13 May 2026):** given a task pool and a token budget, a model must commit to one ordered plan covering selection, sequencing and per-problem allocation. Current LLMs "exhibit substantial gaps in prospective metacognitive control" relative to an oracle. — [arXiv 2605.13414](https://arxiv.org/abs/2605.13414)
- **HEART (24 Jun 2026):** decomposes robotic task planning into atomic reasoning tasks allocated to "role-specialized expert agents under a token budget". It improves plan success over single-LLM and rule-based planners. — [arXiv 2606.25404](https://arxiv.org/abs/2606.25404)
- **Archon (ICML 2025):** treats "maximum token budget" as the x-axis of an accuracy Pareto frontier when composing inference-time techniques. — [arXiv 2409.15254](https://arxiv.org/abs/2409.15254)
- **Plan-level token savings:**
  - ReWOO: 5× token efficiency. — [arXiv 2305.18323](https://arxiv.org/abs/2305.18323)
  - LLM-Tool Compiler: up to 40% lower token cost. — [arXiv 2405.17438](https://arxiv.org/abs/2405.17438)
  - Agentic Plan Caching: 50.31% cost reduction. — [arXiv 2506.14852](https://arxiv.org/abs/2506.14852)
  - Latent-DARM (ICLR 2026 workshop): approaches reasoning-model results using "less than 2.2% of its token budget" via a diffusion planner plus autoregressive executors. — [arXiv 2603.09184](https://arxiv.org/abs/2603.09184)

### Inferences
- For a goal compiler, per-step budgets should be (a) predicted by a cost model trained on execution traces rather than by the LLM itself (TALE's estimator is in range only about 61% of the time, and TRIAGE shows weak prospective allocation), (b) enforced at decode time (budget forcing, or LCPO-trained models), and (c) treated as the "fixed latency" contract that makes static scheduling of downstream steps possible.
- Budget enforcement plus deterministic decoding (Section 5) gives bounded-latency, reproducible steps. That is the closest achievable analogue to Groq's fixed-latency functional units.

### Gaps
- I found no study measuring how well per-step token-budget *predictions* compose into accurate end-to-end latency or cost predictions for multi-step agent plans.

## 8. Deterministic latency / SLO scheduling in LLM serving: what is lost (utilisation) versus gained (predictability)

### Takeaway
GPU serving moved from static to dynamic (continuous) batching for throughput: about 4–23× in 2023. It then reintroduced *planning* to recover predictability: chunked prefills (Sarathi-Serve), prefill/decode disaggregation sized offline to TTFT/TPOT SLOs (DistServe), and workflow-level SLO scheduling for agents (HexAGenT, PipeSwift, Dyserve). Groq/LPX represents the opposite extreme: small-batch, jitter-free decode at the cost of per-chip capacity and utilisation. For agents, the relevant metric is shifting from per-token SLOs to job completion time.

### Cited Findings
- **Static versus continuous batching (Anyscale, Jun 2023):** up to 23× throughput with continuous batching plus vLLM memory management, and about 4× even over optimised static batching. Static batching wastes GPU time waiting on the longest sequence. — [Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- **Sarathi-Serve (OSDI 2024, arXiv Mar 2024):** chunked prefills and "stall-free schedules" under tail-latency constraints give 2.6× serving capacity (Mistral-7B, 1×A100), up to 3.7× (Yi-34B, 2×A100) and up to 5.6× (Falcon-180B with pipeline parallelism) against vLLM. — [arXiv 2403.02310](https://arxiv.org/abs/2403.02310)
- **DistServe (OSDI 2024, arXiv Jan 2024):** disaggregates prefill and decode and "co-optimizes the resource allocation and parallelism strategy" per phase *given* TTFT and TPOT SLOs, which is an ahead-of-time placement plan. It serves 7.4× more requests or meets a 12.6× tighter SLO with more than 90% of requests in SLO. — [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)
- **Agentic, workflow-level SLOs (2025–2026):**
  - Autellix: 4–15× program throughput at the same latency. — [arXiv 2502.13965](https://arxiv.org/abs/2502.13965)
  - HexAGenT: 20.1–33.0% smaller SLO scale at P95/P99. — [arXiv 2605.16637](https://arxiv.org/abs/2605.16637)
  - PipeSwift: JCT is the right objective for agents, and pipeline parallelism cuts JCT 1.45–2.33×. — [arXiv 2609.16491](https://arxiv.org/abs/2609.16491)
  - Dyserve: on-time correct completions 18.1%→67.2% under bursts. — [arXiv 2607.02942](https://arxiv.org/abs/2607.02942)
  - Murakkab: SLOs maintained with 2.8× fewer GPUs. — [arXiv 2508.18298](https://arxiv.org/abs/2508.18298)
- **The LPU end of the spectrum:** a design goal of minimising "latency variance to provide predictable throughput" with batch-1 QoS. — [HC34, 2022](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf). In 2026 NVIDIA positions LPX for "latency-target scheduling" via Dynamo, with stable TTFT/TPOT at small batch. — [NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform)
- **Cost of result determinism on GPUs:** about 1.6–2.1× slower (26 s → 42–55 s) with batch-invariant kernels. — [Thinking Machines, Sep 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

### Inferences
- The utilisation-versus-predictability trade is real but negotiable through planning. DistServe and Murakkab show that ahead-of-time resource plans can meet SLOs *and* reduce hardware. Pure static batching (the closest GPU analogue of Groq's fixed schedule) is the worst throughput option.
- A goal compiler should therefore plan at the *workflow* level (DAG, budgets, placement, prefix reuse) and hand the serving layer a deadline and a priority (HexAGenT/Autellix-style), not a cycle-exact schedule.
- For latency-critical decode-heavy steps (for example, short tool-selection or routing calls inside an agent loop), LPU-class hardware is the natural target. NVIDIA's AFD split (GPUs for attention/KV, LPUs for FFN/MoE experts) suggests a heterogeneous "MoE of hardware" that the planner can target explicitly.

### Gaps
- I found no published measurement of GroqCloud or LPX p99/p50 latency ratios under multi-tenant load, so the "determinism" advantage at service level, including queuing, is unquantified.
- I found no study comparing a fully static agent schedule against dynamic (Autellix/HexAGenT-style) scheduling on the same agent workload and hardware.
